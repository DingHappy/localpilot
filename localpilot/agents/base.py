from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

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
        self.sink: Optional[Callable[[AgentStep], None]] = None

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
        # A real search starts and measures servers for minutes. Emitting
        # each step as it happens is what lets a caller watch the run
        # instead of waiting on a result that only exists at the end.
        if self.sink is not None:
            try:
                self.sink(step)
            except Exception:
                pass
        return step

    def drain(self) -> List[AgentStep]:
        steps = list(self.trace)
        self.trace.clear()
        return steps
