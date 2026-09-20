from __future__ import annotations

import uuid
from typing import List, Optional

from localpilot.agents.bench_agent import BenchAgent
from localpilot.agents.judge_agent import JudgeAgent
from localpilot.agents.planner_agent import PlannerAgent
from localpilot.executor.process import runtime_factory
from localpilot.hardware.profiler import HardwareProfiler
from localpilot.intent.parser import parse_intent
from localpilot.models.registry import ModelRegistry
from localpilot.optimizer.optimizer import Optimizer
from localpilot.planner.planner import Planner
from localpilot.profiles.store import ProfileStore
from localpilot.schemas import (
    AgentStep,
    AutopilotResult,
    CandidateResult,
    SavedProfile,
    StepStatus,
)
from localpilot.utils import append_event, stable_hash, utc_now


class Orchestrator:
    """Runs the loop: understand, profile, plan, execute, judge, remember.

    Three agents with separated authority do the work. This class only
    sequences them, applies the optimizer's ranking, and persists the
    result so the next identical request can skip straight to a verified
    configuration.
    """

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

    # ------------------------------------------------------------------

    def resolve_mode(self, requested: str) -> str:
        """Picks the runtime that will actually execute.

        A named engine is honoured as given so a failure surfaces instead of
        being silently downgraded to a simulation.
        """
        if requested == "mock":
            return "mock"
        if requested != "auto":
            return requested

        hardware = self.profiler.profile(simulate=False)
        if not hardware.real_execution_ready:
            return "mock"
        installed = [
            engine_id
            for engine_id in hardware.stack.get("engines_available", [])
            if self.planner.engines.servable(engine_id)
        ]
        return installed[0] if installed else "mock"

    def autopilot(
        self,
        text: str,
        mode: str = "auto",
        reuse_profile: bool = True,
    ) -> AutopilotResult:
        run_id = str(uuid.uuid4())
        started_at = utc_now()
        intent = parse_intent(text)
        resolved_mode = self.resolve_mode(mode)
        simulated = resolved_mode == "mock"
        hardware = self.profiler.profile(simulate=simulated)

        warnings: List[str] = []
        if simulated:
            warnings.append(
                "SIMULATED: no benchmark ran on real hardware. These numbers "
                "are modelled and must not be reported as measurements."
            )

        trace: List[AgentStep] = []
        models = self.registry.for_task(intent.task)

        if reuse_profile:
            reused = self._try_reuse(
                run_id, intent, hardware, simulated, warnings, started_at, trace
            )
            if reused is not None:
                return reused

        judge = JudgeAgent(self.planner.policies)
        planner_agent = PlannerAgent(self.planner)
        bench_agent = BenchAgent(
            runtime_resolver=self.runtime_resolver,
            planner=self.planner,
            quality_evaluator=judge.evaluate,
        )

        candidates = planner_agent.propose(intent, hardware, models, resolved_mode)
        trace.extend(planner_agent.drain())

        results = bench_agent.measure(
            candidates, intent.task, resolved_mode, hardware, models
        )
        trace.extend(bench_agent.drain())
        trace.extend(judge.drain())

        best = self.optimizer.choose(results, intent.priority)
        trace.append(
            AgentStep(
                agent="optimizer",
                action="rank_candidates",
                status="success",
                detail=(
                    f"{best.candidate.candidate_id} wins with "
                    f"{best.score:.2f}/100 under priority '{intent.priority}'"
                ),
                data={
                    "winner": best.candidate.candidate_id,
                    "score": best.score,
                    "components": best.score_components,
                    "ranking": [
                        {
                            "candidate_id": item.candidate.candidate_id,
                            "score": item.score,
                            "gate_failures": item.gate_failures,
                        }
                        for item in results
                    ],
                },
            )
        )

        profile = self._save_profile(
            intent, hardware, best, resolved_mode, simulated
        )
        trace.append(
            AgentStep(
                agent="memory",
                action="save_profile",
                status="success",
                detail=(
                    f"Profile {profile.profile_key} stored for "
                    f"{intent.task}/{intent.priority} on this fingerprint"
                ),
                data={"profile_key": profile.profile_key},
            )
        )

        result = AutopilotResult(
            run_id=run_id,
            intent=intent,
            hardware=hardware,
            candidates=results,
            best_profile=profile,
            profile_reused=False,
            status="READY",
            warnings=warnings,
            agent_trace=trace,
            started_at=started_at,
            finished_at=utc_now(),
        )
        self.store.save_run(run_id, result.to_dict())
        append_event(
            "autopilot_completed",
            {
                "run_id": run_id,
                "profile_key": profile.profile_key,
                "candidate_id": best.candidate.candidate_id,
                "engine": best.candidate.engine,
                "simulated": simulated,
            },
        )
        return result

    # ------------------------------------------------------------------

    def _try_reuse(
        self,
        run_id: str,
        intent,
        hardware,
        simulated: bool,
        warnings: List[str],
        started_at: str,
        trace: List[AgentStep],
    ) -> Optional[AutopilotResult]:
        cached = self.store.find_match(
            hardware.fingerprint, intent.task, intent.priority, simulated
        )
        if cached is None:
            return None
        if not self._verify_cached(cached):
            trace.append(
                AgentStep(
                    agent="memory",
                    action="profile_rejected",
                    status="degraded",
                    detail=(
                        f"Profile {cached.profile_key} no longer starts; "
                        "replanning from scratch"
                    ),
                )
            )
            return None

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
        trace.append(
            AgentStep(
                agent="memory",
                action="profile_reused",
                status="success",
                detail=(
                    f"Reused {cached.candidate.candidate_id} verified at "
                    f"{cached.created_at}; skipped the search entirely"
                ),
                data={"profile_key": cached.profile_key},
            )
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
            agent_trace=trace,
            started_at=started_at,
            finished_at=utc_now(),
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

    def _save_profile(
        self, intent, hardware, best, resolved_mode: str, simulated: bool
    ) -> SavedProfile:
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
                "runtime": resolved_mode,
                "engine": best.candidate.engine,
                "platform": hardware.platform_id,
                "driver": hardware.accelerator.get("driver_version"),
                "cuda": hardware.stack.get("cuda", {}).get("nvcc_version"),
            },
            simulated=simulated,
            created_at=now,
            last_verified_at=now,
            platform_id=hardware.platform_id,
        )
        self.store.save(profile)
        return profile

    def _verify_cached(self, profile: SavedProfile) -> bool:
        try:
            builder = self.runtime_resolver(profile.candidate.runtime)
        except ValueError:
            return False
        runtime = builder()
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
