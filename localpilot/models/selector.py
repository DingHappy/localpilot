from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from localpilot.sizing import MemoryModel, decode_roofline_tokens_s
from localpilot.schemas import HardwareProfile, Intent, ModelSpec


@dataclass
class Rejection:
    model_id: str
    source_id: str
    gate: str
    reason: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "model_id": self.model_id,
            "source_id": self.source_id,
            "gate": self.gate,
            "reason": self.reason,
        }


@dataclass
class SelectionOutcome:
    selected: List[ModelSpec] = field(default_factory=list)
    rejected: List[Rejection] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected": [model.model_id for model in self.selected],
            "rejected": [item.to_dict() for item in self.rejected],
        }


class ModelSelector:
    """Gates the registry down to models this machine can actually serve.

    Rejections are kept rather than discarded. "The 550B checkpoint needs
    328 GB and you have 128" is more useful to a user than a registry that
    quietly omits it.
    """

    def __init__(self, memory_model: MemoryModel = None) -> None:
        self.memory = memory_model or MemoryModel()

    def select(
        self,
        intent: Intent,
        hardware: HardwareProfile,
        models: List[ModelSpec],
        available_engines: List[str] = None,
    ) -> SelectionOutcome:
        outcome = SelectionOutcome()
        available_engines = available_engines or []
        engine_set = set(available_engines)
        usable_devices = set(hardware.usable_devices)

        for model in models:
            if not model.serves(intent.task):
                continue

            required_modalities = set(intent.modalities) - {"text"}
            if required_modalities and not required_modalities.issubset(
                set(model.modalities)
            ):
                outcome.rejected.append(
                    Rejection(
                        model.model_id,
                        model.source_id,
                        "modality",
                        f"does not handle {sorted(required_modalities)}",
                    )
                )
                continue

            if not set(model.devices) & usable_devices:
                outcome.rejected.append(
                    Rejection(
                        model.model_id,
                        model.source_id,
                        "device",
                        f"needs one of {model.devices}, machine offers "
                        f"{sorted(usable_devices)}",
                    )
                )
                continue

            if engine_set and not set(model.engines) & engine_set:
                outcome.rejected.append(
                    Rejection(
                        model.model_id,
                        model.source_id,
                        "engine",
                        f"servable only by {model.engines}, none installed",
                    )
                )
                continue

            if model.context_length < intent.context_length:
                outcome.rejected.append(
                    Rejection(
                        model.model_id,
                        model.source_id,
                        "context",
                        f"supports {model.context_length} tokens, request "
                        f"needs {intent.context_length}",
                    )
                )
                continue

            estimate = self.memory.estimate(
                model,
                hardware,
                context_length=intent.context_length,
                concurrency=intent.concurrency,
            )
            if not estimate.fits:
                headroom = self.memory.max_context_for(
                    model, hardware, concurrency=intent.concurrency
                )
                detail = estimate.detail
                if headroom > 0:
                    detail += f"; would fit at {headroom} tokens"
                outcome.rejected.append(
                    Rejection(model.model_id, model.source_id, "memory", detail)
                )
                continue

            outcome.selected.append(model)

        outcome.selected.sort(
            key=lambda model: self._prior(model, intent, hardware), reverse=True
        )
        return outcome

    def _prior(
        self, model: ModelSpec, intent: Intent, hardware: HardwareProfile
    ) -> float:
        """Pre-measurement ordering only.

        Nothing here is a claim about performance; it decides which
        candidates are worth spending a real benchmark on, in what order.
        """
        speed = decode_roofline_tokens_s(
            model.active_parameter_count_b,
            model.precision,
            hardware.memory_bandwidth_gbps,
            memory_model=self.memory,
        )
        speed_term = min(1.0, (speed or 30.0) / 120.0)

        if intent.priority == "quality":
            return model.quality_score * 0.85 + speed_term * 0.15
        if intent.priority == "latency":
            return speed_term * 0.75 + model.quality_score * 0.25
        if intent.priority == "low_memory":
            weights = self.memory.weights_gb(model)
            return (1.0 / (1.0 + weights)) * 0.7 + model.quality_score * 0.3
        if intent.priority == "long_context":
            kv_term = 1.0 / (1.0 + model.kv_bytes_per_token / 8192)
            return kv_term * 0.6 + model.quality_score * 0.4
        return model.quality_score * 0.5 + speed_term * 0.5
