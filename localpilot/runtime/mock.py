from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List

from localpilot.runtime.base import RuntimeProvider
from localpilot.schemas import BenchmarkMetrics, CandidatePlan


class MockRuntime(RuntimeProvider):
    """Deterministic development runtime. It never represents real hardware."""

    def __init__(self) -> None:
        self.candidate = None
        self.started = False

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
        }

    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        if not self.started:
            raise RuntimeError("Mock runtime is not started")
        lowered = prompt.lower()
        if "open(path)" in lowered:
            return "Use a with context manager so the file is always closed."
        if "select * from users" in lowered:
            return "This is SQL injection. Use a parameterized query."
        if "shared counter" in lowered:
            return "This has a race condition. Protect it with a lock."
        return (
            "Simulated local response from "
            f"{self.candidate.model_id} on {self.candidate.device}."
        )

    def _rng(self) -> random.Random:
        seed_text = (
            f"{self.candidate.candidate_id}:{self.candidate.context_length}"
        ).encode("utf-8")
        seed = int(hashlib.sha256(seed_text).hexdigest()[:16], 16)
        return random.Random(seed)

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

        device_factor = {"GPU": 1.0, "NPU": 1.12, "CPU": 1.85}.get(
            self.candidate.device, 2.0
        )
        size_factor = max(1.0, self.candidate.expected_memory_gb / 2)
        rng = self._rng()
        ttfts = []
        throughputs = []
        latencies = []
        for _ in range(measured_runs):
            jitter = rng.uniform(0.96, 1.04)
            ttfts.append(260 * device_factor * size_factor * jitter)
            throughputs.append(48 / device_factor / size_factor / jitter)
            latencies.append(1500 * device_factor * size_factor * jitter)

        quality_hits = 0
        for prompt in prompts:
            response = self.generate(prompt["text"], max_new_tokens).lower()
            if any(term.lower() in response for term in prompt["expected_terms"]):
                quality_hits += 1
        task_quality = quality_hits / max(1, len(prompts))
        quality = min(
            1.0, self.candidate.quality_score * 0.75 + task_quality * 0.25
        )

        return BenchmarkMetrics(
            ttft_ms=round(sum(ttfts) / len(ttfts), 2),
            tpot_ms=round(
                1000 / (sum(throughputs) / len(throughputs)), 2
            ),
            throughput_tokens_s=round(sum(throughputs) / len(throughputs), 2),
            total_latency_ms=round(sum(latencies) / len(latencies), 2),
            peak_memory_gb=round(self.candidate.expected_memory_gb * 1.08, 2),
            cpu_usage_percent=None,
            stability=1.0,
            quality=round(quality, 3),
            runs=measured_runs,
            warmup_runs=warmup_runs,
            simulated=True,
            raw={"label": "SIMULATED_NOT_A_HARDWARE_BENCHMARK"},
        )

