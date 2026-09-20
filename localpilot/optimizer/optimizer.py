from __future__ import annotations

from typing import List

from localpilot.planner.policies import PolicyEngine
from localpilot.planner.scoring import score_results
from localpilot.schemas import CandidateResult


class Optimizer:
    def __init__(self, policies: PolicyEngine = None) -> None:
        self.policies = policies or PolicyEngine()

    def choose(
        self, results: List[CandidateResult], priority: str
    ) -> CandidateResult:
        scored = score_results(results, priority, self.policies)
        eligible = [item for item in scored if item.score is not None]
        if not eligible:
            errors = [item.error for item in results if item.error]
            detail = "; ".join(errors) if errors else "quality or stability gate failed"
            raise RuntimeError(f"No candidate passed optimization gates: {detail}")
        return max(eligible, key=lambda item: item.score)

