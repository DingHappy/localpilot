from __future__ import annotations

from typing import Callable, List

from localpilot.benchmark.runner import BenchmarkRunner
from localpilot.runtime.base import RuntimeProvider
from localpilot.schemas import CandidatePlan, CandidateResult, StepStatus
from localpilot.utils import append_event


class Executor:
    """Owns one candidate's lifecycle: load, start, verify, measure, stop.

    ``quality_evaluator`` is invoked while the server is still running,
    because grading output requires the model that produced it to still be
    loaded. Its failure degrades the result rather than failing it: a
    candidate with measured speed and an ungraded answer is still worth
    reporting, clearly marked.
    """

    def __init__(
        self,
        runtime_builder: Callable[[], RuntimeProvider],
        benchmark_runner: BenchmarkRunner = None,
        quality_evaluator: Callable = None,
    ) -> None:
        self.runtime_builder = runtime_builder
        self.benchmark_runner = benchmark_runner or BenchmarkRunner()
        self.quality_evaluator = quality_evaluator

    def execute(
        self, candidates: List[CandidatePlan], task: str
    ) -> List[CandidateResult]:
        results = []
        for candidate in candidates:
            results.append(self._execute_one(candidate, task))
        return results

    def _execute_one(
        self, candidate: CandidatePlan, task: str
    ) -> CandidateResult:
        steps: List[StepStatus] = []
        runtime = self.runtime_builder()
        try:
            steps.append(StepStatus("environment_check", "success"))
            runtime.load_model(candidate)
            steps.append(StepStatus("model_load", "success"))
            runtime.start_model()
            steps.append(StepStatus("model_start", "success"))

            health = runtime.health_check()
            if not health.get("healthy"):
                raise RuntimeError(
                    f"Runtime health check failed: {health.get('detail')}"
                )
            steps.append(StepStatus("health_check", "success"))

            benchmark = self.benchmark_runner.run(runtime, candidate, task)
            steps.append(StepStatus("benchmark", "success"))

            if self.quality_evaluator is not None:
                self._apply_quality(runtime, candidate, task, benchmark, steps)

            append_event(
                "candidate_success",
                {
                    "candidate_id": candidate.candidate_id,
                    "engine": candidate.engine,
                    "runtime": candidate.runtime,
                    "simulated": candidate.simulated,
                },
            )
            return CandidateResult(
                candidate=candidate,
                status="success",
                steps=steps,
                benchmark=benchmark,
            )
        except Exception as exc:
            steps.append(
                StepStatus("execution", "failed", f"{type(exc).__name__}: {exc}")
            )
            append_event(
                "candidate_failed",
                {
                    "candidate_id": candidate.candidate_id,
                    "engine": candidate.engine,
                    "error_type": type(exc).__name__,
                },
            )
            return CandidateResult(
                candidate=candidate,
                status="failed",
                steps=steps,
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            try:
                runtime.stop_model()
            except Exception:
                pass

    def _apply_quality(
        self,
        runtime: RuntimeProvider,
        candidate: CandidatePlan,
        task: str,
        benchmark,
        steps: List[StepStatus],
    ) -> None:
        try:
            evaluation = self.quality_evaluator(runtime, candidate, task)
        except Exception as exc:
            steps.append(
                StepStatus(
                    "quality_grading",
                    "degraded",
                    f"{type(exc).__name__}: {exc}",
                )
            )
            return

        benchmark.quality_keyword = evaluation.get("keyword")
        benchmark.quality_judge = evaluation.get("judge")
        blended = evaluation.get("blended")
        if blended is not None:
            benchmark.quality = float(blended)
        benchmark.raw = dict(benchmark.raw or {})
        benchmark.raw["quality_detail"] = evaluation.get("detail")
        benchmark.raw["quality_samples"] = evaluation.get("samples", [])
        steps.append(
            StepStatus(
                "quality_grading",
                "success" if evaluation.get("judge") is not None else "degraded",
                str(evaluation.get("detail", "")),
            )
        )
