from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from localpilot.runtime.base import RuntimeProvider, RuntimeUnavailable
from localpilot.schemas import BenchmarkMetrics, CandidatePlan


def _metric_mean(value: Any) -> float:
    if hasattr(value, "mean"):
        return float(value.mean)
    return float(value)


class OpenVINORuntime(RuntimeProvider):
    def __init__(self) -> None:
        self.candidate = None
        self.pipeline = None
        self.ov_genai = None

    def load_model(self, candidate: CandidatePlan) -> None:
        if candidate.simulated:
            raise ValueError("OpenVINORuntime refuses simulated candidates")
        if importlib.util.find_spec("openvino_genai") is None:
            raise RuntimeUnavailable(
                "openvino-genai is not installed; run doctor before real execution"
            )
        if not candidate.model_path:
            raise RuntimeUnavailable(
                "LOCALPILOT_MODEL_PATH must point to a local OpenVINO IR model"
            )
        model_path = Path(candidate.model_path).expanduser()
        if not model_path.is_dir():
            raise RuntimeUnavailable(f"Model path does not exist: {model_path}")

        import openvino_genai

        self.ov_genai = openvino_genai
        self.candidate = candidate
        self.pipeline = openvino_genai.LLMPipeline(
            str(model_path),
            candidate.device,
            **candidate.runtime_config,
        )

    def start_model(self) -> None:
        if self.pipeline is None:
            raise RuntimeError("No model loaded")

    def stop_model(self) -> None:
        self.pipeline = None
        self.candidate = None

    def health_check(self) -> Dict[str, Any]:
        return {
            "healthy": self.pipeline is not None,
            "runtime": "openvino",
            "simulated": False,
            "device": self.candidate.device if self.candidate else None,
        }

    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        if self.pipeline is None:
            raise RuntimeError("OpenVINO pipeline is not started")
        return str(
            self.pipeline.generate(prompt, max_new_tokens=max_new_tokens)
        )

    def benchmark(
        self,
        prompts: List[Dict[str, Any]],
        warmup_runs: int,
        measured_runs: int,
        max_new_tokens: int,
    ) -> BenchmarkMetrics:
        if self.pipeline is None or self.ov_genai is None:
            raise RuntimeError("OpenVINO pipeline is not started")

        config = self.ov_genai.GenerationConfig()
        config.max_new_tokens = max_new_tokens
        benchmark_prompt = prompts[0]["text"]
        for _ in range(warmup_runs):
            self.pipeline.generate([benchmark_prompt], config)

        ttfts = []
        tpots = []
        throughputs = []
        latencies = []
        success_count = 0
        for _ in range(measured_runs):
            started = time.perf_counter()
            result = self.pipeline.generate([benchmark_prompt], config)
            elapsed_ms = (time.perf_counter() - started) * 1000
            perf = result.perf_metrics
            ttfts.append(_metric_mean(perf.get_ttft()))
            tpots.append(_metric_mean(perf.get_tpot()))
            throughputs.append(_metric_mean(perf.get_throughput()))
            latencies.append(elapsed_ms)
            success_count += 1

        quality_hits = 0
        for prompt in prompts:
            response = self.generate(prompt["text"], max_new_tokens).lower()
            if any(term.lower() in response for term in prompt["expected_terms"]):
                quality_hits += 1

        peak_memory_gb = self.candidate.expected_memory_gb
        cpu_usage = None
        try:
            import psutil

            process = psutil.Process()
            peak_memory_gb = process.memory_info().rss / (1024**3)
            cpu_usage = process.cpu_percent(interval=0.05)
        except ImportError:
            pass

        return BenchmarkMetrics(
            ttft_ms=round(mean(ttfts), 2),
            tpot_ms=round(mean(tpots), 2),
            throughput_tokens_s=round(mean(throughputs), 2),
            total_latency_ms=round(mean(latencies), 2),
            peak_memory_gb=round(peak_memory_gb, 2),
            cpu_usage_percent=cpu_usage,
            stability=round(success_count / measured_runs, 3),
            quality=round(quality_hits / max(1, len(prompts)), 3),
            runs=measured_runs,
            warmup_runs=warmup_runs,
            simulated=False,
            raw={},
        )

