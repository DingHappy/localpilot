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

    def priority_names(self) -> list:
        return sorted(self.data.get("priorities", {}))

    @property
    def max_candidates(self) -> int:
        return int(self.data.get("max_candidates", 4))

    @property
    def max_candidates_per_model(self) -> int:
        return int(self.data.get("max_candidates_per_model", 2))

    @property
    def max_rebuilding_candidates(self) -> int:
        return int(self.data.get("max_rebuilding_candidates", 1))

    @property
    def memory_config(self) -> Dict[str, Any]:
        return dict(self.data.get("memory", {}))

    @property
    def safety_reserve_percent(self) -> float:
        return float(self.memory_config.get("safety_reserve_percent", 20))

    @property
    def search_space(self) -> Dict[str, Any]:
        return dict(self.data.get("search_space", {}))

    @property
    def roofline(self) -> Dict[str, Any]:
        return dict(self.data.get("roofline", {}))

    @property
    def bandwidth_efficiency(self) -> float:
        return float(self.roofline.get("bandwidth_efficiency", 0.75))

    @property
    def speculative_gain(self) -> float:
        return float(self.roofline.get("speculative_decoding_expected_gain", 1.8))

    @property
    def batch_amortization_exponent(self) -> float:
        return float(self.roofline.get("batch_amortization_exponent", 0.65))

    @property
    def judge(self) -> Dict[str, Any]:
        return dict(self.data.get("judge", {"enabled": False}))

    @property
    def quality_gate(self) -> float:
        return float(self.data.get("quality_gate", 0.55))

    @property
    def stability_gate(self) -> float:
        return float(self.data.get("stability_gate", 0.90))

    @property
    def recovery(self) -> Dict[str, Any]:
        return dict(
            self.data.get(
                "recovery",
                {
                    "max_attempts": 2,
                    "minimum_context_length": 4096,
                    "fallback_engine": "vllm",
                    "fallback_device": "CPU",
                },
            )
        )
