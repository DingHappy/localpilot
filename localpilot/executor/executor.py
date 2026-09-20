from __future__ import annotations

from typing import Callable, List

from localpilot.benchmark.runner import BenchmarkRunner
from localpilot.runtime.base import RuntimeProvider
from localpilot.schemas import CandidatePlan, CandidateResult, StepStatus
from localpilot.utils import append_event


class Executor:
    def __init__(
        self,
        runtime_builder: Callable[[], RuntimeProvider],
        benchmark_runner: BenchmarkRunner = None,
    ) -> None:
        self.runtime_builder = runtime_builder
        self.benchmark_runner = benchmark_runner or BenchmarkRunner()

    def execute(
        self, candidates: List[CandidatePlan], task: str
    ) -> List[CandidateResult]:
        results = []
        for candidate in candidates:
            steps = []
            runtime = self.runtime_builder()
            try:
                steps.append(StepStatus("environment_check", "success"))
                runtime.load_model(candidate)
                steps.append(StepStatus("model_load", "success"))
                runtime.start_model()
                steps.append(StepStatus("model_start", "success"))
                health = runtime.health_check()
                if not health.get("healthy"):
                    raise RuntimeError("Runtime health check failed")
                steps.append(StepStatus("health_check", "success"))
                benchmark = self.benchmark_runner.run(runtime, candidate, task)
                steps.append(StepStatus("benchmark", "success"))
                result = CandidateResult(
                    candidate=candidate,
                    status="success",
                    steps=steps,
                    benchmark=benchmark,
                )
                append_event(
                    "candidate_success",
                    {
                        "candidate_id": candidate.candidate_id,
                        "runtime": candidate.runtime,
                        "simulated": candidate.simulated,
                    },
                )
            except Exception as exc:
                steps.append(
                    StepStatus(
                        "execution",
                        "failed",
                        f"{type(exc).__name__}: {exc}",
                    )
                )
                result = CandidateResult(
                    candidate=candidate,
                    status="failed",
                    steps=steps,
                    error=f"{type(exc).__name__}: {exc}",
                )
                append_event(
                    "candidate_failed",
                    {
                        "candidate_id": candidate.candidate_id,
                        "error_type": type(exc).__name__,
                    },
                )
            finally:
                try:
                    runtime.stop_model()
                except Exception:
                    pass
            results.append(result)
        return results

