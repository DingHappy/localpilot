from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional

from localpilot.benchmark.prompts import keyword_hit, load_benchmark_config, prompt_text
from localpilot.planner.policies import PolicyEngine
from localpilot.runtime.base import RuntimeProvider
from localpilot.schemas import BenchmarkMetrics, CandidatePlan
from localpilot.sizing import MemoryModel, decode_roofline_tokens_s


# Per-engine multipliers against the bandwidth ceiling. Priors for
# development only: which engine actually wins on a given machine is
# exactly the question a real run exists to answer.
ENGINE_EFFICIENCY = {
    "trtllm": 1.00,
    "vllm": 0.93,
    "sglang": 0.92,
    "nim": 0.90,
    "llamacpp": 0.70,
    "transformers": 0.35,
}

ENGINE_STARTUP_TTFT_MS = {
    "trtllm": 55.0,
    "vllm": 70.0,
    "sglang": 68.0,
    "nim": 75.0,
    "llamacpp": 120.0,
    "transformers": 260.0,
}


class MockRuntime(RuntimeProvider):
    """Deterministic development runtime. It never represents real hardware.

    The numbers come from a roofline model rather than fixed constants, so
    the orchestration, the search space and the scoring can be developed
    and demonstrated off-target while still moving in the directions the
    physics dictates: sparse beats dense on decode, batching raises
    aggregate throughput and hurts per-stream latency, speculative decoding
    helps a single stream and stops helping once compute is committed.

    Every value it produces is labelled simulated and is barred from being
    reported as a measurement.
    """

    def __init__(
        self,
        policies: PolicyEngine = None,
        memory_bandwidth_gbps: float = 273.0,
    ) -> None:
        self.candidate: Optional[CandidatePlan] = None
        self.started = False
        self.policies = policies or PolicyEngine()
        self.memory_model = MemoryModel(self.policies.memory_config)
        self.memory_bandwidth_gbps = memory_bandwidth_gbps
        self._answers: Optional[Dict[str, str]] = None

    def load_model(self, candidate: CandidatePlan) -> None:
        if not candidate.simulated:
            raise ValueError("MockRuntime only accepts simulated candidates")
        self.candidate = candidate

    def start_model(self) -> None:
        if self.candidate is None:
            raise RuntimeError("No model loaded")
        self.started = True

    def stop_model(self) -> None:
        self.started = False

    def health_check(self) -> Dict[str, Any]:
        return {
            "healthy": self.started and self.candidate is not None,
            "runtime": "mock",
            "simulated": True,
            "engine_modelled": self.candidate.engine if self.candidate else None,
        }

    def _canned_answers(self) -> Dict[str, str]:
        """Prompt text to canned reply, declared in the benchmark config.

        Keeping the fabricated answers in configuration rather than in code
        means the invented part of a simulated run is visible in review
        instead of buried in a chain of string matches.
        """
        if self._answers is None:
            config = load_benchmark_config()
            self._answers = {
                prompt["text"]: prompt["mock_answer"]
                for prompt in config.get("prompts", [])
                if prompt.get("mock_answer")
            }
        return self._answers

    def generate(self, prompt: Any, max_new_tokens: int = 64) -> str:
        if not self.started:
            raise RuntimeError("Mock runtime is not started")
        answer = self._canned_answers().get(prompt_text(prompt))
        if answer:
            return answer
        return (
            "Simulated local response from "
            f"{self.candidate.model_id} on {self.candidate.engine}."
        )

    def _rng(self) -> random.Random:
        seed_text = (
            f"{self.candidate.candidate_id}:{self.candidate.context_length}:"
            f"{self.candidate.concurrency}"
        ).encode("utf-8")
        return random.Random(int(hashlib.sha256(seed_text).hexdigest()[:16], 16))

    def _modelled_rates(self) -> Dict[str, float]:
        candidate = self.candidate
        engine = candidate.engine
        efficiency = ENGINE_EFFICIENCY.get(engine, 0.8)

        ceiling = decode_roofline_tokens_s(
            candidate.active_parameter_count_b or candidate.parameter_count_b,
            candidate.precision,
            self.memory_bandwidth_gbps,
            efficiency=self.policies.bandwidth_efficiency,
            memory_model=self.memory_model,
        ) or 40.0
        per_stream = ceiling * efficiency

        speculative = bool(candidate.runtime_config.get("speculative"))
        if speculative:
            # Drafting wins by cutting memory passes, and the win shrinks as
            # batching claims the compute it needs to verify.
            gain = self.policies.speculative_gain
            damping = 1.0 / (1.0 + 0.25 * max(0, candidate.concurrency - 1))
            per_stream *= 1 + (gain - 1) * damping

        concurrency = max(1, candidate.concurrency)
        if concurrency > 1:
            # One weight read serves the whole batch, so aggregate output
            # rises sublinearly while each stream slows down.
            exponent = self.policies.batch_amortization_exponent
            aggregate = per_stream * (concurrency**exponent)
            per_stream = aggregate / concurrency
        else:
            aggregate = per_stream

        context_penalty = 1 + (candidate.context_length / 1_000_000) * 0.35
        per_stream /= context_penalty
        aggregate /= context_penalty

        ttft = ENGINE_STARTUP_TTFT_MS.get(engine, 90.0)
        ttft += candidate.context_length / 1024 * 1.8
        ttft *= 1 + 0.12 * (concurrency - 1)
        if candidate.runtime_config.get("enable_prefix_caching"):
            ttft *= 0.9

        bytes_per_parameter = self.memory_model.bytes_for(candidate.precision)
        active_bytes = (
            (candidate.active_parameter_count_b or candidate.parameter_count_b)
            * 1e9
            * bytes_per_parameter
        )
        achieved_gbps = per_stream * concurrency * active_bytes / 1e9

        return {
            "per_stream": per_stream,
            "aggregate": aggregate,
            "ttft": ttft,
            "achieved_gbps": achieved_gbps,
        }

    def benchmark(
        self,
        prompts: List[Dict[str, Any]],
        warmup_runs: int,
        measured_runs: int,
        max_new_tokens: int,
    ) -> BenchmarkMetrics:
        if not self.started:
            raise RuntimeError("Mock runtime is not started")

        for _ in range(warmup_runs):
            self.generate(prompts[0]["text"], max_new_tokens)

        rates = self._modelled_rates()
        rng = self._rng()
        ttfts: List[float] = []
        throughputs: List[float] = []
        latencies: List[float] = []
        for _ in range(max(1, measured_runs)):
            jitter = rng.uniform(0.96, 1.04)
            ttft = rates["ttft"] * jitter
            per_stream = rates["per_stream"] / jitter
            ttfts.append(ttft)
            throughputs.append(per_stream)
            latencies.append(ttft + (max_new_tokens / max(per_stream, 1e-6)) * 1000)

        quality_hits = 0
        for prompt in prompts:
            response = self.generate(prompt, max_new_tokens)
            if keyword_hit(response, prompt):
                quality_hits += 1
        keyword_quality = quality_hits / max(1, len(prompts))
        quality = min(
            1.0, self.candidate.quality_score * 0.75 + keyword_quality * 0.25
        )

        mean_throughput = sum(throughputs) / len(throughputs)
        peak_memory = self.candidate.expected_memory_gb * rng.uniform(1.02, 1.08)

        return BenchmarkMetrics(
            ttft_ms=round(sum(ttfts) / len(ttfts), 2),
            tpot_ms=round(1000 / mean_throughput, 2),
            throughput_tokens_s=round(mean_throughput, 2),
            total_latency_ms=round(sum(latencies) / len(latencies), 2),
            peak_memory_gb=round(peak_memory, 2),
            cpu_usage_percent=None,
            stability=1.0,
            quality=round(quality, 3),
            runs=measured_runs,
            warmup_runs=warmup_runs,
            simulated=True,
            raw={
                "label": "SIMULATED_NOT_A_HARDWARE_BENCHMARK",
                "engine_modelled": self.candidate.engine,
                "model": "roofline: bandwidth / active bytes per token",
                "assumed_bandwidth_gbps": self.memory_bandwidth_gbps,
            },
            ttft_p95_ms=round(max(ttfts), 2),
            concurrency=self.candidate.concurrency,
            aggregate_throughput_tokens_s=round(rates["aggregate"], 2),
            quality_keyword=round(keyword_quality, 3),
            decode_bandwidth_gbps=round(rates["achieved_gbps"], 2),
            bandwidth_utilization=round(
                rates["achieved_gbps"] / self.memory_bandwidth_gbps, 3
            ),
        )
