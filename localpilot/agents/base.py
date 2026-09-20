from __future__ import annotations

from typing import Any, Dict, List

from localpilot.schemas import AgentStep


class Agent:
    """One role in the autopilot loop.

    The split is along authority, not along code size. The planner may
    propose but never measure; the bench agent may measure but never
    score quality; the judge may score quality but never rank. Keeping
    those apart is what stops a component from marking its own homework.
    """

    name = "agent"
    role = ""

    def __init__(self) -> None:
        self.trace: List[AgentStep] = []

    def record(
        self,
        action: str,
        status: str = "success",
        detail: str = "",
        data: Dict[str, Any] = None,
    ) -> AgentStep:
        step = AgentStep(
            agent=self.name,
            action=action,
            status=status,
            detail=detail,
            data=data or {},
        )
        self.trace.append(step)
        return step

    def drain(self) -> List[AgentStep]:
        steps = list(self.trace)
        self.trace.clear()
        return steps
