from __future__ import annotations

from typing import Callable, List

from localpilot.agents.base import Agent
from localpilot.executor.executor import Executor
from localpilot.planner.planner import Planner
from localpilot.schemas import (
    CandidatePlan,
    CandidateResult,
    HardwareProfile,
    ModelSpec,
)


class BenchAgent(Agent):
    """Executes candidates and measures them. Never decides a winner."""

    name = "bench"
    role = "Serves each candidate, measures it, and recovers from failures"

    def __init__(
        self,
        runtime_resolver: Callable,
        planner: Planner,
        quality_evaluator: Callable = None,
    ) -> None:
        super().__init__()
        self.runtime_resolver = runtime_resolver
        self.planner = planner
        self.quality_evaluator = quality_evaluator

    def measure(
        self,
        candidates: List[CandidatePlan],
        task: str,
        runtime_mode: str,
        hardware: HardwareProfile,
        models: List[ModelSpec],
    ) -> List[CandidateResult]:
        executor = Executor(
            self.runtime_resolver(runtime_mode),
            quality_evaluator=self.quality_evaluator,
        )
        results = executor.execute(candidates, task)

        for result in results:
            if result.status == "success" and result.benchmark:
                metrics = result.benchmark
                memory_source = (metrics.raw or {}).get(
                    "peak_memory_source", "unknown"
                )
                memory_word = (
                    "estimated memory"
                    if memory_source == "planner_estimate_no_readable_source"
                    else "peak memory"
                )
                detail = (
                    f"{result.candidate.candidate_id}: "
                    f"TTFT {metrics.ttft_ms:.0f} ms, "
                    f"{metrics.throughput_tokens_s:.1f} tok/s/stream, "
                    f"{metrics.peak_memory_gb:.1f} GB {memory_word}"
                )
                if metrics.aggregate_throughput_tokens_s:
                    detail += (
                        f", {metrics.aggregate_throughput_tokens_s:.1f} tok/s "
                        "aggregate"
                    )
                self.record(
                    "measure_candidate",
                    detail=detail,
                    data={
                        "candidate_id": result.candidate.candidate_id,
                        "ttft_ms": metrics.ttft_ms,
                        "throughput_tokens_s": metrics.throughput_tokens_s,
                        "aggregate_throughput_tokens_s": (
                            metrics.aggregate_throughput_tokens_s
                        ),
                        "peak_memory_gb": metrics.peak_memory_gb,
                        "peak_memory_source": memory_source,
                        "quality": metrics.quality,
                        "simulated": metrics.simulated,
                    },
                )
            else:
                self.record(
                    "candidate_failed",
                    status="failed",
                    detail=f"{result.candidate.candidate_id}: {result.error}",
                    data={
                        "candidate_id": result.candidate.candidate_id,
                        "error": result.error,
                    },
                )

        if not any(item.status == "success" for item in results):
            recovery = self.planner.recovery_plan(candidates, hardware, models)
            if not recovery:
                self.record(
                    "recovery_exhausted",
                    status="failed",
                    detail="No reversible smaller configuration was available",
                )
                return results

            self.record(
                "recovery_started",
                status="degraded",
                detail=(
                    f"All {len(candidates)} candidates failed; trying "
                    f"{len(recovery)} bounded fallbacks"
                ),
                data={
                    "attempts": [item.candidate_id for item in recovery],
                    "actions": [item.recovery_action for item in recovery],
                },
            )
            results.extend(executor.execute(recovery, task))

        return results
