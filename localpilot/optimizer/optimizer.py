from __future__ import annotations

from typing import List

from localpilot.planner.policies import PolicyEngine
from localpilot.planner.scoring import score_results
from localpilot.schemas import CandidateResult


class NoEligibleCandidate(RuntimeError):
    """Expected rejection after measuring or filtering the bounded candidates."""


class Optimizer:
    def __init__(self, policies: PolicyEngine = None) -> None:
        self.policies = policies or PolicyEngine()

    def choose(
        self, results: List[CandidateResult], priority: str, acceptance=None
    ) -> CandidateResult:
        scored = score_results(results, priority, self.policies, acceptance)
        eligible = [item for item in scored if item.score is not None]
        if not eligible:
            errors = [item.error for item in results if item.error]
            errors.extend(f"{item.candidate.candidate_id}: {', '.join(item.gate_failures)}" for item in results if item.gate_failures)
            detail = "; ".join(errors) if errors else "quality or stability gate failed"
            raise NoEligibleCandidate(f"No candidate passed optimization gates: {detail}")
        if acceptance and acceptance.get("objective") in {"fastest_complete", "highest_quality"}:
            return max(eligible, key=lambda item: (item.score, -item.benchmark.total_latency_ms))
        return max(eligible, key=lambda item: item.score)
