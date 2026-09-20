from __future__ import annotations

from typing import List

from localpilot.agents.base import Agent
from localpilot.planner.planner import Planner
from localpilot.schemas import CandidatePlan, HardwareProfile, Intent, ModelSpec
from localpilot.sizing import decode_roofline_tokens_s


class PlannerAgent(Agent):
    """Proposes what to try, and says why each rejection happened."""

    name = "planner"
    role = "Turns an intent plus a hardware profile into a ranked candidate set"

    def __init__(self, planner: Planner = None) -> None:
        super().__init__()
        self.planner = planner or Planner()

    def propose(
        self,
        intent: Intent,
        hardware: HardwareProfile,
        models: List[ModelSpec],
        runtime: str,
    ) -> List[CandidatePlan]:
        self.record(
            "read_hardware",
            detail=(
                f"{hardware.platform_id}: "
                f"{hardware.memory.get('total_gb')} GB "
                f"{'unified' if hardware.unified_memory else 'dedicated'}, "
                f"{hardware.memory_bandwidth_gbps or 'unknown'} GB/s"
            ),
            data={
                "platform_id": hardware.platform_id,
                "unified_memory": hardware.unified_memory,
                "engines_available": hardware.stack.get("engines_available", []),
                "simulated": hardware.simulated,
            },
        )

        candidates = self.planner.plan(intent, hardware, models, runtime)
        selection = self.planner.last_selection

        if selection and selection.rejected:
            for rejection in selection.rejected:
                self.record(
                    "reject_model",
                    status="rejected",
                    detail=f"{rejection.model_id}: {rejection.reason}",
                    data=rejection.to_dict(),
                )

        for candidate in candidates:
            ceiling = decode_roofline_tokens_s(
                candidate.active_parameter_count_b or candidate.parameter_count_b,
                candidate.precision,
                hardware.memory_bandwidth_gbps,
                efficiency=self.planner.policies.bandwidth_efficiency,
                memory_model=self.planner.memory,
            )
            self.record(
                "propose_candidate",
                detail=candidate.reason,
                data={
                    "candidate_id": candidate.candidate_id,
                    "engine": candidate.engine,
                    "precision": candidate.precision,
                    "context_length": candidate.context_length,
                    "concurrency": candidate.concurrency,
                    "speculative_decoding": bool(
                        candidate.runtime_config.get("speculative")
                    ),
                    "expected_memory_gb": candidate.expected_memory_gb,
                    "roofline_decode_tokens_s": (
                        round(ceiling, 1) if ceiling else None
                    ),
                },
            )

        self.record(
            "plan_complete",
            detail=(
                f"{len(candidates)} candidates across "
                f"{len({item.model_id for item in candidates})} models and "
                f"{len({item.engine for item in candidates})} engines"
            ),
        )
        return candidates
