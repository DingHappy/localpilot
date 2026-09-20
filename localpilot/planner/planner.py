from __future__ import annotations

import os
from typing import List

from localpilot.models.selector import ModelSelector
from localpilot.planner.policies import PolicyEngine
from localpilot.schemas import CandidatePlan, HardwareProfile, Intent, ModelSpec


class Planner:
    def __init__(self, policies: PolicyEngine = None) -> None:
        self.policies = policies or PolicyEngine()
        self.selector = ModelSelector()

    def plan(
        self,
        intent: Intent,
        hardware: HardwareProfile,
        models: List[ModelSpec],
        runtime: str,
    ) -> List[CandidatePlan]:
        selected_models = self.selector.select(intent, hardware, models)
        if not selected_models:
            raise RuntimeError(f"No viable models for task {intent.task}")

        policy = self.policies.priority(intent.priority)
        preferred_devices = policy.get("prefer_device", ["GPU", "NPU", "CPU"])
        available = {item.split(".", 1)[0] for item in hardware.available_devices}
        usable_memory = hardware.memory.get("total_gb")
        if usable_memory is not None:
            usable_memory *= 1 - self.policies.safety_reserve_percent / 100

        candidates = []
        for device_rank, device in enumerate(preferred_devices):
            if device not in available:
                continue
            for model in selected_models:
                if device not in model.supported_devices:
                    continue
                expected_memory = round(
                    max(model.disk_size_gb * 1.35, model.parameter_count_b * 0.8),
                    2,
                )
                if usable_memory is not None and expected_memory > usable_memory:
                    continue
                variant = model.variants.get(device, {})
                precision = variant.get("precision", model.precision)
                confidence = max(
                    0.50,
                    0.92 - device_rank * 0.08 - (0.05 if not hardware.simulated else 0),
                )
                reason = (
                    f"{device} ranks #{device_rank + 1} for {intent.priority}; "
                    f"{model.model_id} fits the memory and context gates"
                )
                candidates.append(
                    CandidatePlan(
                        candidate_id=f"{model.model_id}-{device.lower()}-{precision.lower()}",
                        model_id=model.model_id,
                        source_id=model.source_id,
                        device=device,
                        precision=precision,
                        context_length=min(intent.context_length, model.context_length),
                        runtime=runtime,
                        expected_memory_gb=expected_memory,
                        quality_score=model.quality_score,
                        reason=reason,
                        confidence=round(confidence, 2),
                        simulated=hardware.simulated,
                        model_path=os.environ.get("LOCALPILOT_MODEL_PATH") or None,
                    )
                )

        if not candidates:
            raise RuntimeError("No executable model and device combinations were found")

        if intent.priority == "quality":
            candidates.sort(
                key=lambda item: (item.quality_score, item.confidence), reverse=True
            )
        return candidates[: self.policies.max_candidates]

    def recovery_plan(
        self,
        failed_candidates: List[CandidatePlan],
        hardware: HardwareProfile,
        models: List[ModelSpec],
    ) -> List[CandidatePlan]:
        if not failed_candidates:
            return []

        recovery_policy = self.policies.recovery
        maximum = max(0, int(recovery_policy.get("max_attempts", 2)))
        minimum_context = max(
            512, int(recovery_policy.get("minimum_context_length", 2048))
        )
        fallback_device = str(
            recovery_policy.get("fallback_device", "CPU")
        ).upper()
        primary = failed_candidates[0]
        model = next(
            (item for item in models if item.model_id == primary.model_id),
            None,
        )
        if model is None:
            return []

        available = {
            item.split(".", 1)[0] for item in hardware.available_devices
        }
        reduced_context = max(minimum_context, primary.context_length // 2)
        candidates = []
        seen = {item.candidate_id for item in failed_candidates}

        if reduced_context < primary.context_length:
            candidate_id = f"{primary.candidate_id}-context-{reduced_context}"
            if candidate_id not in seen:
                candidates.append(
                    CandidatePlan(
                        candidate_id=candidate_id,
                        model_id=primary.model_id,
                        source_id=primary.source_id,
                        device=primary.device,
                        precision=primary.precision,
                        context_length=reduced_context,
                        runtime=primary.runtime,
                        expected_memory_gb=round(
                            primary.expected_memory_gb * 0.85, 2
                        ),
                        quality_score=primary.quality_score,
                        reason=(
                            f"Recovery after {primary.candidate_id}: reduce "
                            f"context to {reduced_context}"
                        ),
                        confidence=max(0.40, primary.confidence - 0.10),
                        simulated=primary.simulated,
                        model_path=primary.model_path,
                        runtime_config=dict(primary.runtime_config),
                        fallback_of=primary.candidate_id,
                        recovery_action="reduce_context",
                    )
                )

        if (
            fallback_device in available
            and fallback_device in model.supported_devices
            and fallback_device != primary.device
        ):
            variant = model.variants.get(fallback_device, {})
            precision = variant.get("precision", model.precision)
            candidate_id = (
                f"{model.model_id}-{fallback_device.lower()}-"
                f"{precision.lower()}-recovery"
            )
            if candidate_id not in seen:
                candidates.append(
                    CandidatePlan(
                        candidate_id=candidate_id,
                        model_id=model.model_id,
                        source_id=model.source_id,
                        device=fallback_device,
                        precision=precision,
                        context_length=reduced_context,
                        runtime=primary.runtime,
                        expected_memory_gb=round(
                            primary.expected_memory_gb * 0.90, 2
                        ),
                        quality_score=model.quality_score,
                        reason=(
                            f"Recovery after {primary.candidate_id}: switch to "
                            f"{fallback_device} with context {reduced_context}"
                        ),
                        confidence=max(0.35, primary.confidence - 0.18),
                        simulated=primary.simulated,
                        model_path=primary.model_path,
                        fallback_of=primary.candidate_id,
                        recovery_action="fallback_device_and_context",
                    )
                )

        return candidates[:maximum]
