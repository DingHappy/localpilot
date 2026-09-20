from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from localpilot.utils import load_data_file, project_home


class PolicyEngine:
    def __init__(self, path: Path = None) -> None:
        self.path = path or (project_home() / "config" / "policies.yaml")
        self.data = load_data_file(self.path)

    def priority(self, name: str) -> Dict[str, Any]:
        priorities = self.data.get("priorities", {})
        return priorities.get(name, priorities["balanced"])

    @property
    def max_candidates(self) -> int:
        return int(self.data.get("max_candidates", 3))

    @property
    def safety_reserve_percent(self) -> float:
        return float(
            self.data.get("memory", {}).get("safety_reserve_percent", 20)
        )

    @property
    def recovery(self) -> Dict[str, Any]:
        return dict(
            self.data.get(
                "recovery",
                {
                    "max_attempts": 2,
                    "minimum_context_length": 2048,
                    "fallback_device": "CPU",
                },
            )
        )
