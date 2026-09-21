from __future__ import annotations

from localpilot.benchmark.prompts import load_benchmark_config, prompts_for_task
from localpilot.runtime.base import RuntimeProvider
from localpilot.schemas import BenchmarkMetrics, CandidatePlan
from localpilot.utils import stable_hash


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
        result = runtime.benchmark(
            prompts=prompts,
            warmup_runs=int(config.get("warmup_runs", 1)),
            measured_runs=max(1, int(config.get("measured_runs", 3))),
            max_new_tokens=int(config.get("max_new_tokens", 64)),
        )
        result.raw = dict(result.raw or {})
        result.raw["benchmark_config_sha256"] = stable_hash(config)
        result.raw["benchmark_dataset_id"] = config.get("dataset_id")
        result.raw["max_new_tokens"] = int(config.get("max_new_tokens", 64))
        return result
