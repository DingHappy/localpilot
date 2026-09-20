from __future__ import annotations

import uuid
from typing import Optional

from localpilot.executor.executor import Executor
from localpilot.executor.process import runtime_factory
from localpilot.hardware.profiler import HardwareProfiler
from localpilot.intent.parser import parse_intent
from localpilot.models.registry import ModelRegistry
from localpilot.optimizer.optimizer import Optimizer
from localpilot.planner.planner import Planner
from localpilot.profiles.store import ProfileStore
from localpilot.schemas import (
    AutopilotResult,
    CandidateResult,
    SavedProfile,
    StepStatus,
)
from localpilot.utils import append_event, stable_hash, utc_now


class Orchestrator:
    def __init__(
        self,
        profiler: HardwareProfiler = None,
        registry: ModelRegistry = None,
        planner: Planner = None,
        optimizer: Optimizer = None,
        store: ProfileStore = None,
        runtime_resolver=runtime_factory,
    ) -> None:
        self.profiler = profiler or HardwareProfiler()
        self.registry = registry or ModelRegistry()
        self.planner = planner or Planner()
        self.optimizer = optimizer or Optimizer(self.planner.policies)
        self.store = store or ProfileStore()
        self.runtime_resolver = runtime_resolver

    def resolve_mode(self, requested: str) -> str:
        if requested in {"mock", "openvino"}:
            return requested
        actual = self.profiler.profile(simulate=False)
        return "openvino" if actual.real_execution_ready else "mock"

    def autopilot(
        self,
        text: str,
        mode: str = "auto",
        reuse_profile: bool = True,
    ) -> AutopilotResult:
        run_id = str(uuid.uuid4())
        intent = parse_intent(text)
        resolved_mode = self.resolve_mode(mode)
        simulated = resolved_mode == "mock"
        hardware = self.profiler.profile(simulate=simulated)
        warnings = []
        if simulated:
            warnings.append(
                "SIMULATED: no real Intel/OpenVINO benchmark was performed."
            )

        if reuse_profile:
            cached = self.store.find_match(
                hardware.fingerprint,
                intent.task,
                intent.priority,
                simulated,
            )
            if cached and self._verify_cached(cached):
                cached.last_verified_at = utc_now()
                self.store.save(cached)
                candidate_result = CandidateResult(
                    candidate=cached.candidate,
                    status="success",
                    steps=[
                        StepStatus("profile_lookup", "success", "PROFILE HIT"),
                        StepStatus("health_check", "success"),
                    ],
                    benchmark=cached.benchmark,
                    score=cached.score,
                )
                result = AutopilotResult(
                    run_id=run_id,
                    intent=intent,
                    hardware=hardware,
                    candidates=[candidate_result],
                    best_profile=cached,
                    profile_reused=True,
                    status="READY",
                    warnings=warnings,
                )
                self.store.save_run(run_id, result.to_dict())
                append_event(
                    "profile_reused",
                    {
                        "run_id": run_id,
                        "profile_key": cached.profile_key,
                        "simulated": simulated,
                    },
                )
                return result

        candidates = self.planner.plan(
            intent=intent,
            hardware=hardware,
            models=self.registry.for_task(intent.task),
            runtime=resolved_mode,
        )
        executor = Executor(self.runtime_resolver(resolved_mode))
        results = executor.execute(candidates, intent.task)
        if not any(item.status == "success" for item in results):
            recovery_candidates = self.planner.recovery_plan(
                failed_candidates=candidates,
                hardware=hardware,
                models=self.registry.for_task(intent.task),
            )
            if recovery_candidates:
                warnings.append(
                    "Primary candidates failed; LocalPilot executed bounded recovery."
                )
                append_event(
                    "recovery_started",
                    {
                        "run_id": run_id,
                        "attempts": len(recovery_candidates),
                        "simulated": simulated,
                    },
                )
                results.extend(
                    executor.execute(recovery_candidates, intent.task)
                )
        best = self.optimizer.choose(results, intent.priority)
        now = utc_now()
        profile_key = stable_hash(
            {
                "hardware": hardware.fingerprint,
                "task": intent.task,
                "priority": intent.priority,
                "candidate": best.candidate.candidate_id,
                "runtime": resolved_mode,
            }
        )[:20]
        profile = SavedProfile(
            profile_key=profile_key,
            hardware_fingerprint=hardware.fingerprint,
            task=intent.task,
            priority=intent.priority,
            candidate=best.candidate,
            benchmark=best.benchmark,
            score=float(best.score),
            runtime_versions={
                "openvino": hardware.openvino.get("version"),
                "runtime": resolved_mode,
            },
            simulated=simulated,
            created_at=now,
            last_verified_at=now,
        )
        self.store.save(profile)
        result = AutopilotResult(
            run_id=run_id,
            intent=intent,
            hardware=hardware,
            candidates=results,
            best_profile=profile,
            profile_reused=False,
            status="READY",
            warnings=warnings,
        )
        self.store.save_run(run_id, result.to_dict())
        append_event(
            "autopilot_completed",
            {
                "run_id": run_id,
                "profile_key": profile_key,
                "candidate_id": best.candidate.candidate_id,
                "simulated": simulated,
            },
        )
        return result

    def _verify_cached(self, profile: SavedProfile) -> bool:
        runtime = self.runtime_resolver(profile.candidate.runtime)()
        try:
            runtime.load_model(profile.candidate)
            runtime.start_model()
            return bool(runtime.health_check().get("healthy"))
        except Exception:
            return False
        finally:
            try:
                runtime.stop_model()
            except Exception:
                pass
