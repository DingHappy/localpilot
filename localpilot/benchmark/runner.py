from __future__ import annotations

from localpilot.benchmark.prompts import load_benchmark_config, prompts_for_task
from localpilot.runtime.base import RuntimeProvider
from localpilot.schemas import BenchmarkMetrics, CandidatePlan


class BenchmarkRunner:
    def run(
        self,
        runtime: RuntimeProvider,
        candidate: CandidatePlan,
        task: str,
    ) -> BenchmarkMetrics:
        config = load_benchmark_config()
        prompts = prompts_for_task(task)
        if not prompts:
            raise RuntimeError(f"No benchmark prompts configured for task {task}")
        return runtime.benchmark(
            prompts=prompts,
            warmup_runs=int(config.get("warmup_runs", 1)),
            measured_runs=max(1, int(config.get("measured_runs", 3))),
            max_new_tokens=int(config.get("max_new_tokens", 64)),
        )

