from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from localpilot.engines.registry import EngineRegistry, EngineSpec
from localpilot.runtime.base import RuntimeProvider, RuntimeUnavailable
from localpilot.schemas import BenchmarkMetrics, CandidatePlan


DEFAULT_PORT = 8_100
HEALTH_TIMEOUT_SECONDS = float(os.environ.get("LOCALPILOT_ENGINE_TIMEOUT", "900"))


@dataclass
class StreamSample:
    ttft_ms: float
    total_ms: float
    output_tokens: int
    ok: bool
    error: Optional[str] = None

    @property
    def decode_ms(self) -> float:
        return max(0.0, self.total_ms - self.ttft_ms)

    @property
    def tokens_per_second(self) -> Optional[float]:
        if self.output_tokens <= 1 or self.decode_ms <= 0:
            return None
        return (self.output_tokens - 1) / (self.decode_ms / 1000)


def _distinct_prompts(
    prompts: List[Dict[str, Any]], concurrency: int
) -> List[str]:
    """Builds `concurrency` prompts that do not share a full prefix.

    Rotating the task's own prompts keeps the load realistic. When more
    streams are asked for than there are prompts, a trailing marker
    differentiates the reused ones: concurrent users share a system prefix
    but not an entire request, and a benchmark that lets them share
    everything measures the prefix cache.
    """
    texts = [prompt["text"] for prompt in prompts] or [""]
    built = []
    for index in range(max(1, concurrency)):
        text = texts[index % len(texts)]
        repeat = index // len(texts)
        if repeat:
            text = f"{text}\n\n(request variant {repeat + 1})"
        built.append(text)
    return built


class MemorySampler:
    """Samples memory during generation and keeps the peak.

    The engine runs in its own process, so this process's RSS is
    irrelevant; what matters is the pool the weights and cache live in.
    Which pool that is depends on the platform, so the sampler picks its
    own source rather than being told:

    - **accelerator** -- NVML reports per-device used memory. Correct for a
      discrete board.
    - **system pool** -- on GB10 NVML answers `NVMLError_NotSupported`,
      because a unified pool has no separate device memory to report (this
      is also why `nvidia-smi` prints `[N/A]`). There the system view *is*
      the accelerator view, so it is the right reading rather than a
      substitute for one.

    On the system path it tracks `total - available`, which excludes
    reclaimable page cache, and records a baseline so the figure
    attributable to this run can be separated from whatever else the
    machine was already holding.
    """

    INTERVAL_SECONDS = 0.25

    def __init__(self) -> None:
        self.peak_gb: Optional[float] = None
        self.baseline_gb: Optional[float] = None
        self.samples = 0
        self.source: Optional[str] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _read_nvml_gb(self) -> Optional[float]:
        try:
            import pynvml
        except ImportError:
            return None
        try:
            pynvml.nvmlInit()
        except Exception:
            return None
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**3)
        except Exception:
            # Includes NVMLError_NotSupported on unified memory.
            return None
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def _read_system_gb(self) -> Optional[float]:
        try:
            import psutil

            memory = psutil.virtual_memory()
            return (memory.total - memory.available) / (1024**3)
        except ImportError:
            pass
        try:
            fields = {}
            with open("/proc/meminfo", encoding="utf-8") as handle:
                for line in handle:
                    key, _, rest = line.partition(":")
                    fields[key] = int(rest.strip().split()[0])
            total = fields.get("MemTotal")
            available = fields.get("MemAvailable")
            if total and available is not None:
                return (total - available) * 1024 / (1024**3)
        except (OSError, ValueError, IndexError):
            pass
        return None

    def _read_gb(self) -> Optional[float]:
        if self.source == "accelerator":
            return self._read_nvml_gb()
        if self.source == "system_pool":
            return self._read_system_gb()
        value = self._read_nvml_gb()
        if value is not None:
            self.source = "accelerator"
            return value
        value = self._read_system_gb()
        if value is not None:
            self.source = "system_pool"
        return value

    def _run(self) -> None:
        while not self._stop.is_set():
            value = self._read_gb()
            if value is not None:
                self.samples += 1
                if self.baseline_gb is None:
                    self.baseline_gb = value
                if self.peak_gb is None or value > self.peak_gb:
                    self.peak_gb = value
            self._stop.wait(self.INTERVAL_SECONDS)

    def start(self) -> None:
        first = self._read_gb()
        if first is None:
            return
        self.baseline_gb = first
        self.peak_gb = first
        self.samples = 1
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def attributable_gb(self) -> Optional[float]:
        """Peak above the baseline, when a baseline was captured."""
        if self.peak_gb is None or self.baseline_gb is None:
            return None
        return max(0.0, self.peak_gb - self.baseline_gb)


def _percentile(values: List[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


class OpenAICompatRuntime(RuntimeProvider):
    """Drives any engine that speaks the OpenAI HTTP API.

    Measuring through one wire protocol is what makes a cross-engine
    comparison honest: time-to-first-token and inter-token latency are
    observed the same way whether the server is vLLM, TensorRT-LLM, SGLang
    or a NIM container, so a difference in the numbers is a difference in
    the engine rather than in the harness.

    Attaching to a server the user already started is the default. Starting
    one is opt-in, because ``vllm serve <repo>`` downloads weights, and
    nothing here may pull tens of gigabytes without being asked.
    """

    engine_id: str = ""

    def __init__(self, engine_registry: EngineRegistry = None) -> None:
        self.engines = engine_registry or EngineRegistry()
        self.candidate: Optional[CandidatePlan] = None
        self.spec: Optional[EngineSpec] = None
        self.base_url: Optional[str] = None
        self.process: Optional[subprocess.Popen] = None
        self.launch_command: Optional[List[str]] = None
        self.model_name: Optional[str] = None
        self._owns_server = False

    # ------------------------------------------------------------------
    # lifecycle

    def _engine_id_for(self, candidate: CandidatePlan) -> str:
        return self.engine_id or candidate.engine

    def _configured_base_url(self, engine_id: str) -> Optional[str]:
        for variable in (
            f"LOCALPILOT_{engine_id.upper()}_BASE_URL",
            "LOCALPILOT_ENGINE_BASE_URL",
        ):
            value = os.environ.get(variable)
            if value:
                return value.rstrip("/")
        return None

    def load_model(self, candidate: CandidatePlan) -> None:
        if candidate.simulated:
            raise ValueError(
                f"{type(self).__name__} refuses simulated candidates"
            )
        engine_id = self._engine_id_for(candidate)
        self.spec = self.engines.get(engine_id)
        if not self.spec.openai_base_path:
            raise RuntimeUnavailable(
                f"{engine_id} does not expose an OpenAI-compatible endpoint"
            )
        self.candidate = candidate
        self.model_name = candidate.source_id

        configured = self._configured_base_url(engine_id)
        if configured:
            self.base_url = configured
            self._owns_server = False
            return

        command = self.build_launch_command(candidate)
        self.launch_command = command
        if os.environ.get("LOCALPILOT_ALLOW_ENGINE_LAUNCH") != "1":
            raise RuntimeUnavailable(
                f"No running {engine_id} server was configured, and LocalPilot "
                "will not start one implicitly because that downloads model "
                "weights.\n"
                f"  Either export LOCALPILOT_{engine_id.upper()}_BASE_URL="
                "http://127.0.0.1:8000 for a server you already run,\n"
                "  or set LOCALPILOT_ALLOW_ENGINE_LAUNCH=1 to let LocalPilot "
                "run:\n"
                f"    {' '.join(command)}"
            )
        self.base_url = f"http://127.0.0.1:{self._port()}"
        self._owns_server = True

    def _port(self) -> int:
        return int(os.environ.get("LOCALPILOT_ENGINE_PORT", DEFAULT_PORT))

    def build_launch_command(self, candidate: CandidatePlan) -> List[str]:
        """Renders the engine's configured launch template.

        Returned even when launching is refused, so the message can show the
        user the exact command and let them decide.
        """
        spec = self.spec or self.engines.get(self._engine_id_for(candidate))
        launch = spec.launch or {}
        context = {
            "source_id": candidate.source_id,
            "model_path": candidate.model_path or candidate.source_id,
            "context_length": candidate.context_length,
            "max_num_seqs": candidate.runtime_config.get("max_num_seqs", 1),
            "gpu_memory_utilization": candidate.runtime_config.get(
                "gpu_memory_utilization", 0.9
            ),
            "kv_cache_dtype": candidate.kv_cache_dtype,
            "port": self._port(),
        }

        def render(value: Any) -> str:
            text = str(value)
            for key, replacement in context.items():
                text = text.replace("{" + key + "}", str(replacement))
            return text

        command = [render(part) for part in launch.get("command", [])]
        for flag, template in (launch.get("arguments") or {}).items():
            rendered = render(template)
            if "{" in rendered:
                continue
            command.extend([flag, rendered])

        flags = launch.get("flags") or {}
        if candidate.runtime_config.get("enable_prefix_caching") and flags.get(
            "prefix_caching"
        ):
            command.append(flags["prefix_caching"])

        speculative = candidate.runtime_config.get("speculative")
        if speculative and (launch.get("speculative") or {}):
            draft_context = {
                "draft_source_id": speculative.get("draft_source_id", ""),
                "num_speculative_tokens": speculative.get(
                    "num_speculative_tokens", 3
                ),
            }
            for flag, template in launch["speculative"].items():
                rendered = str(template)
                for key, replacement in draft_context.items():
                    rendered = rendered.replace("{" + key + "}", str(replacement))
                command.extend([flag, rendered])
        return command

    def start_model(self) -> None:
        if self.candidate is None or self.base_url is None:
            raise RuntimeError("No model loaded")
        if self._owns_server and self.process is None:
            self.process = subprocess.Popen(
                self.launch_command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        self._wait_for_health()
        self._adopt_served_model_name()

    def _adopt_served_model_name(self) -> None:
        """Ask the server which model it serves instead of assuming.

        A repository id is not a served name. An operator starting vLLM
        picks it with --served-model-name, and a container commonly serves
        a mounted path under a short alias, so sending the checkpoint's
        repo id gets a 404 from a server that is working perfectly.
        """
        base = (self.spec.openai_base_path or "/v1").rstrip("/")
        try:
            listing = self._get(f"{base}/models", timeout=10.0)
        except Exception:
            return
        served = [
            entry.get("id")
            for entry in (listing or {}).get("data", [])
            if entry.get("id")
        ]
        if not served:
            return
        if self.model_name in served:
            return
        self.model_name = served[0]

    def _wait_for_health(self) -> None:
        deadline = time.monotonic() + HEALTH_TIMEOUT_SECONDS
        last_error = None
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeUnavailable(
                    f"{self.spec.engine_id} exited with code "
                    f"{self.process.returncode} before becoming healthy"
                )
            try:
                self._get(self.spec.health_path or "/health")
                return
            except Exception as exc:
                last_error = exc
                time.sleep(2.0)
        raise RuntimeUnavailable(
            f"{self.spec.engine_id} did not become healthy within "
            f"{HEALTH_TIMEOUT_SECONDS:.0f}s: {last_error}"
        )

    def stop_model(self) -> None:
        if self.process is not None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=60)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except OSError:
                    pass
            finally:
                self.process = None
        self.candidate = None
        self.base_url = None
        self._owns_server = False

    def health_check(self) -> Dict[str, Any]:
        if self.base_url is None or self.spec is None:
            return {"healthy": False, "runtime": self.engine_id, "simulated": False}
        try:
            self._get(self.spec.health_path or "/health")
            healthy = True
            detail = None
        except Exception as exc:
            healthy = False
            detail = f"{type(exc).__name__}: {exc}"
        return {
            "healthy": healthy,
            "runtime": self.spec.engine_id,
            "simulated": False,
            "base_url": self.base_url,
            "managed_by_localpilot": self._owns_server,
            "device": self.candidate.device if self.candidate else None,
            "detail": detail,
        }

    # ------------------------------------------------------------------
    # HTTP

    def _get(self, path: str, timeout: float = 10.0) -> Any:
        request = urllib.request.Request(f"{self.base_url}{path}", method="GET")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
        try:
            return json.loads(body) if body else {}
        except json.JSONDecodeError:
            return body

    def _completions_path(self) -> str:
        base = (self.spec.openai_base_path or "/v1").rstrip("/")
        return f"{base}/chat/completions"

    def _payload(
        self, prompt: str, max_new_tokens: int, stream: bool
    ) -> Dict[str, Any]:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
            "stream": stream,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        if self.base_url is None:
            raise RuntimeError(f"{self.engine_id} runtime is not started")
        body = json.dumps(self._payload(prompt, max_new_tokens, False)).encode()
        request = urllib.request.Request(
            f"{self.base_url}{self._completions_path()}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            data = json.loads(response.read().decode("utf-8", "replace"))
        choices = data.get("choices") or []
        if not choices:
            return ""
        return str(choices[0].get("message", {}).get("content", ""))

    def _stream_once(self, prompt: str, max_new_tokens: int) -> StreamSample:
        body = json.dumps(self._payload(prompt, max_new_tokens, True)).encode()
        request = urllib.request.Request(
            f"{self.base_url}{self._completions_path()}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        first_token_at: Optional[float] = None
        delta_count = 0
        reported_tokens: Optional[int] = None
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                for raw in response:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        event = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    usage = event.get("usage")
                    if isinstance(usage, dict) and usage.get("completion_tokens"):
                        reported_tokens = int(usage["completion_tokens"])
                    for choice in event.get("choices") or []:
                        content = (choice.get("delta") or {}).get("content")
                        if content:
                            if first_token_at is None:
                                first_token_at = time.perf_counter()
                            delta_count += 1
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            return StreamSample(0.0, 0.0, 0, False, f"{type(exc).__name__}: {exc}")

        finished = time.perf_counter()
        if first_token_at is None:
            return StreamSample(
                0.0,
                (finished - started) * 1000,
                0,
                False,
                "stream produced no content",
            )
        # The server's own token count beats counting SSE frames, which can
        # batch several tokens into one delta.
        tokens = reported_tokens or delta_count
        return StreamSample(
            ttft_ms=(first_token_at - started) * 1000,
            total_ms=(finished - started) * 1000,
            output_tokens=tokens,
            ok=True,
        )

    def _stream_concurrent(
        self, prompts: List[str], max_new_tokens: int
    ) -> Tuple[List[StreamSample], float]:
        """Runs one request per prompt at the same time.

        The prompts must differ. Firing the identical request N times at a
        server with prefix caching on measures the cache, not the engine:
        every stream after the first skips prefill entirely, and the
        aggregate figure comes out far above what real concurrent users
        would see.
        """
        concurrency = len(prompts)
        samples: List[Optional[StreamSample]] = [None] * concurrency

        def worker(slot: int) -> None:
            samples[slot] = self._stream_once(prompts[slot], max_new_tokens)

        threads = [
            threading.Thread(target=worker, args=(slot,), daemon=True)
            for slot in range(concurrency)
        ]
        wall_start = time.perf_counter()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        wall_ms = (time.perf_counter() - wall_start) * 1000
        return [sample for sample in samples if sample is not None], wall_ms

    # ------------------------------------------------------------------
    # measurement

    def benchmark(
        self,
        prompts: List[Dict[str, Any]],
        warmup_runs: int,
        measured_runs: int,
        max_new_tokens: int,
    ) -> BenchmarkMetrics:
        if self.base_url is None or self.candidate is None:
            raise RuntimeError(f"{self.engine_id} runtime is not started")

        primary = prompts[0]["text"]
        concurrency = max(1, self.candidate.concurrency)
        batch_prompts = _distinct_prompts(prompts, concurrency)
        # Serial runs rotate too. Repeating one prompt lets warmup populate
        # the prefix cache and every measured run hit it, so the reported
        # TTFT describes a cache hit rather than a prefill.
        serial_prompts = _distinct_prompts(prompts, max(1, measured_runs))

        for _ in range(max(0, warmup_runs)):
            self._stream_once(primary, max_new_tokens)

        # Sample accelerator memory while generation is in flight. Reading it
        # once at the end reports whatever is resident after the work is
        # done, which is not the peak the configuration actually needed.
        sampler = MemorySampler()
        sampler.start()

        all_samples: List[StreamSample] = []
        wall_times: List[float] = []
        try:
            for index in range(max(1, measured_runs)):
                if concurrency == 1:
                    sample = self._stream_once(
                        serial_prompts[index % len(serial_prompts)],
                        max_new_tokens,
                    )
                    all_samples.append(sample)
                    wall_times.append(sample.total_ms)
                else:
                    batch, wall_ms = self._stream_concurrent(
                        batch_prompts, max_new_tokens
                    )
                    all_samples.extend(batch)
                    wall_times.append(wall_ms)
        finally:
            sampler.stop()

        successes = [sample for sample in all_samples if sample.ok]
        if not successes:
            errors = {sample.error for sample in all_samples if sample.error}
            raise RuntimeError(
                "every measured request failed: " + "; ".join(sorted(errors))
            )

        ttfts = [sample.ttft_ms for sample in successes]
        per_stream = [
            rate
            for rate in (sample.tokens_per_second for sample in successes)
            if rate
        ]
        tokens_total = sum(sample.output_tokens for sample in successes)

        throughput = mean(per_stream) if per_stream else 0.0

        # Aggregate throughput is every token the server produced divided by
        # the wall time it took, so concurrent streams count once each. This
        # is the number that improves with batching while per-stream speed
        # gets worse.
        aggregate = None
        total_wall_seconds = sum(wall_times) / 1000
        if total_wall_seconds > 0 and tokens_total:
            aggregate = round(tokens_total / total_wall_seconds, 2)

        quality_hits = 0
        for prompt in prompts:
            response = self.generate(prompt["text"], max_new_tokens).lower()
            if any(term.lower() in response for term in prompt["expected_terms"]):
                quality_hits += 1
        keyword_quality = quality_hits / max(1, len(prompts))

        engine_report = self._engine_memory_report()
        peak_memory_gb, cpu_percent, memory_source = self._resource_usage(
            sampler, engine_report
        )

        return BenchmarkMetrics(
            ttft_ms=round(mean(ttfts), 2),
            tpot_ms=round(1000 / throughput, 2) if throughput else 0.0,
            throughput_tokens_s=round(throughput, 2),
            total_latency_ms=round(mean([s.total_ms for s in successes]), 2),
            peak_memory_gb=round(peak_memory_gb, 2),
            cpu_usage_percent=cpu_percent,
            stability=round(len(successes) / max(1, len(all_samples)), 3),
            quality=round(keyword_quality, 3),
            runs=measured_runs,
            warmup_runs=warmup_runs,
            simulated=False,
            raw={
                "engine": self.spec.engine_id,
                "base_url": self.base_url,
                "managed_by_localpilot": self._owns_server,
                "requests": len(all_samples),
                "output_tokens_total": tokens_total,
                "launch_command": self.launch_command,
                "peak_memory_source": memory_source,
                "memory_baseline_gb": (
                    round(sampler.baseline_gb, 2)
                    if sampler.baseline_gb is not None else None
                ),
                "memory_attributable_gb": (
                    round(sampler.attributable_gb, 2)
                    if sampler.attributable_gb is not None else None
                ),
                "distinct_prompts": len(set(batch_prompts + serial_prompts)),
                "engine_memory": engine_report,
            },
            ttft_p95_ms=(
                round(_percentile(ttfts, 0.95), 2)
                if _percentile(ttfts, 0.95) is not None
                else None
            ),
            concurrency=concurrency,
            aggregate_throughput_tokens_s=aggregate,
            quality_keyword=round(keyword_quality, 3),
        )

    def _engine_memory_report(self) -> Optional[Dict[str, Any]]:
        """What the engine says it reserved, which is the real footprint.

        Sampling a pool cannot answer this in attach mode: the weights are
        resident before sampling starts, so the baseline already contains
        them and the observed delta is noise. The engine, on the other
        hand, knows exactly how many KV blocks it allocated.
        """
        if self.spec is None or self.candidate is None:
            return None
        from localpilot.engines.introspect import introspect

        config = introspect(
            self.spec.engine_id,
            self.spec.openai_base_path or "/v1",
            base_url=self.base_url,
        )
        if config is None or not config.raw_available:
            return None

        kv_bytes = self._kv_bytes_per_token()
        kv_gb = config.kv_allocated_gb(kv_bytes) if kv_bytes else None
        weights_gb = self._weights_gb()
        total = None
        if kv_gb is not None and weights_gb is not None:
            total = kv_gb + weights_gb
        return {
            "server": config.to_dict(),
            "kv_allocated_gb": None if kv_gb is None else round(kv_gb, 2),
            "weights_gb": weights_gb,
            "total_gb": None if total is None else round(total, 2),
        }

    def _kv_bytes_per_token(self) -> Optional[float]:
        from localpilot.models.registry import ModelRegistry

        try:
            return ModelRegistry().get(self.candidate.model_id).kv_bytes_per_token
        except Exception:
            return None

    def _weights_gb(self) -> Optional[float]:
        from localpilot.models.registry import ModelRegistry

        try:
            return ModelRegistry().get(self.candidate.model_id).weights_gb or None
        except Exception:
            return None

    def _resource_usage(
        self,
        sampler: "MemorySampler",
        engine_report: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Optional[float], str]:
        """The footprint, from the most trustworthy source available.

        Preference order, because each later option describes something
        progressively further from "what this configuration needs":

        1. **the engine's own accounting** -- allocated KV blocks plus the
           checkpoint's measured weight size.
        2. **a sampled pool peak** -- correct on a quiet machine, but it
           includes whatever else the pool was holding.
        3. **the planner's estimate** -- not an observation at all, and
           labelled so.
        """
        if engine_report and engine_report.get("total_gb") is not None:
            return (
                float(engine_report["total_gb"]),
                self._cpu_percent(),
                "engine_reported_weights_plus_allocated_kv",
            )
        if sampler.peak_gb is not None:
            return (
                sampler.peak_gb,
                self._cpu_percent(),
                f"{sampler.source}_peak_over_{sampler.samples}_samples",
            )
        return (
            float(self.candidate.expected_memory_gb),
            self._cpu_percent(),
            "planner_estimate_no_readable_source",
        )

    @staticmethod
    def _cpu_percent() -> Optional[float]:
        try:
            import psutil

            return float(psutil.cpu_percent(interval=0.1))
        except ImportError:
            return None


class VLLMRuntime(OpenAICompatRuntime):
    engine_id = "vllm"


class TRTLLMRuntime(OpenAICompatRuntime):
    engine_id = "trtllm"


class SGLangRuntime(OpenAICompatRuntime):
    engine_id = "sglang"


class NIMRuntime(OpenAICompatRuntime):
    engine_id = "nim"


class LlamaCppRuntime(OpenAICompatRuntime):
    engine_id = "llamacpp"
