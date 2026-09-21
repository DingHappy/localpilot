from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from localpilot.profiles.store import ProfileStore
from localpilot.schemas import SavedProfile
from localpilot.utils import atomic_write_json, utc_now


DECISIONS = {"KEEP", "REBENCH", "RECONFIGURE", "SWITCH"}


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _number(value: Any, name: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if result < 0:
        raise ValueError(f"{name} cannot be negative")
    return result


@dataclass(frozen=True)
class RuntimeObservation:
    """One aggregated production observation, never a raw request payload."""

    timestamp: str = field(default_factory=utc_now)
    source_sequence: Optional[str] = None
    request_p95_ms: Optional[float] = None
    ttft_p95_ms: Optional[float] = None
    throughput_tokens_s: Optional[float] = None
    error_rate: Optional[float] = None
    quality: Optional[float] = None
    available_memory_gb: Optional[float] = None
    queue_depth: Optional[float] = None
    active_requests: Optional[float] = None
    healthy: Optional[bool] = None

    def __post_init__(self) -> None:
        _parse_time(self.timestamp)
        for name in (
            "ttft_p95_ms",
            "request_p95_ms",
            "throughput_tokens_s",
            "error_rate",
            "quality",
            "available_memory_gb",
            "queue_depth",
            "active_requests",
        ):
            _number(getattr(self, name), name)
        if self.error_rate is not None and self.error_rate > 1:
            raise ValueError("error_rate must be between 0 and 1")
        if self.quality is not None and self.quality > 1:
            raise ValueError("quality must be between 0 and 1")
        if self.healthy is not None and not isinstance(self.healthy, bool):
            raise ValueError("healthy must be true or false")

    @classmethod
    def from_mapping(cls, payload: Dict[str, Any]) -> "RuntimeObservation":
        if not isinstance(payload, dict):
            raise ValueError("metrics input must be a JSON object")
        timestamp = str(payload.get("timestamp") or utc_now())
        _parse_time(timestamp)
        healthy = payload.get("healthy")
        if healthy is not None and not isinstance(healthy, bool):
            raise ValueError("healthy must be true or false")
        error_rate = _number(payload.get("error_rate"), "error_rate")
        quality = _number(payload.get("quality"), "quality")
        if error_rate is not None and error_rate > 1:
            raise ValueError("error_rate must be between 0 and 1")
        if quality is not None and quality > 1:
            raise ValueError("quality must be between 0 and 1")
        return cls(
            timestamp=timestamp,
            source_sequence=(
                str(payload["source_sequence"])
                if payload.get("source_sequence") is not None
                else None
            ),
            request_p95_ms=_number(
                payload.get("request_p95_ms"), "request_p95_ms"
            ),
            ttft_p95_ms=_number(payload.get("ttft_p95_ms"), "ttft_p95_ms"),
            throughput_tokens_s=_number(
                payload.get("throughput_tokens_s"), "throughput_tokens_s"
            ),
            error_rate=error_rate,
            quality=quality,
            available_memory_gb=_number(
                payload.get("available_memory_gb"), "available_memory_gb"
            ),
            queue_depth=_number(payload.get("queue_depth"), "queue_depth"),
            active_requests=_number(
                payload.get("active_requests"), "active_requests"
            ),
            healthy=healthy,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ServiceObjectives:
    max_request_p95_ms: Optional[float] = None
    max_ttft_p95_ms: Optional[float] = None
    min_throughput_tokens_s: Optional[float] = None
    max_error_rate: float = 0.02
    min_quality: Optional[float] = None
    min_available_memory_gb: Optional[float] = None
    max_queue_depth: Optional[float] = None

    def __post_init__(self) -> None:
        values = {
            "max_ttft_p95_ms": self.max_ttft_p95_ms,
            "max_request_p95_ms": self.max_request_p95_ms,
            "min_throughput_tokens_s": self.min_throughput_tokens_s,
            "max_error_rate": self.max_error_rate,
            "min_quality": self.min_quality,
            "min_available_memory_gb": self.min_available_memory_gb,
            "max_queue_depth": self.max_queue_depth,
        }
        for name, value in values.items():
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.max_error_rate > 1:
            raise ValueError("max_error_rate must be between 0 and 1")
        if self.min_quality is not None and self.min_quality > 1:
            raise ValueError("min_quality must be between 0 and 1")

    @classmethod
    def from_profile(
        cls,
        profile: SavedProfile,
        *,
        max_request_p95_ms: Optional[float] = None,
        max_ttft_p95_ms: Optional[float] = None,
        min_throughput_tokens_s: Optional[float] = None,
        max_error_rate: float = 0.02,
        min_quality: Optional[float] = None,
        min_available_memory_gb: Optional[float] = None,
        max_queue_depth: Optional[float] = None,
    ) -> "ServiceObjectives":
        benchmark = profile.benchmark
        baseline_ttft = benchmark.ttft_p95_ms or benchmark.ttft_ms
        baseline_throughput = (
            benchmark.aggregate_throughput_tokens_s
            if profile.candidate.concurrency > 1
            and benchmark.aggregate_throughput_tokens_s is not None
            else benchmark.throughput_tokens_s
        )
        return cls(
            max_request_p95_ms=(
                float(max_request_p95_ms)
                if max_request_p95_ms is not None
                else round(float(benchmark.total_latency_ms) * 1.5, 3)
            ),
            max_ttft_p95_ms=(
                float(max_ttft_p95_ms)
                if max_ttft_p95_ms is not None
                else round(float(baseline_ttft) * 1.5, 3)
            ),
            min_throughput_tokens_s=(
                float(min_throughput_tokens_s)
                if min_throughput_tokens_s is not None
                else round(float(baseline_throughput) * 0.7, 3)
            ),
            max_error_rate=float(max_error_rate),
            min_quality=(
                float(min_quality)
                if min_quality is not None
                else max(0.0, round(float(benchmark.quality) - 0.05, 3))
            ),
            min_available_memory_gb=(
                float(min_available_memory_gb)
                if min_available_memory_gb is not None
                else None
            ),
            max_queue_depth=(
                float(max_queue_depth)
                if max_queue_depth is not None
                else float(max(2, profile.candidate.concurrency * 2))
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReconcilePolicy:
    consecutive_breaches: int = 3
    cooldown_seconds: int = 900

    def __post_init__(self) -> None:
        if self.consecutive_breaches < 1:
            raise ValueError("consecutive_breaches must be at least 1")
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds cannot be negative")


@dataclass
class ReconcileResult:
    plan_id: str
    created_at: str
    profile_key: Optional[str]
    decision: str
    actionable: bool
    deferred_by_cooldown: bool
    reasons: List[str]
    current_breaches: List[str]
    sustained_breaches: List[str]
    consecutive_required: int
    thresholds: Dict[str, Any]
    observation: Dict[str, Any]
    environment_changed: bool = False
    requirements_changed: bool = False
    safe_execution: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.decision not in DECISIONS:
            raise ValueError(f"unknown reconcile decision: {self.decision}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Reconciler:
    """Turns aggregated drift signals into a reviewable, non-destructive plan."""

    def __init__(self, store: ProfileStore, now=utc_now) -> None:
        self.store = store
        self.now = now
        self.control_root = store.root / "control"
        self.observation_path = self.control_root / "observations.jsonl"
        self.plan_root = self.control_root / "plans"

    def reconcile(
        self,
        observation: RuntimeObservation,
        objectives: ServiceObjectives,
        policy: ReconcilePolicy = ReconcilePolicy(),
        *,
        hardware_fingerprint: Optional[str] = None,
        desired_requirements: Optional[Dict[str, Any]] = None,
    ) -> ReconcileResult:
        current = self.store.current()
        profile_key = current.get("profile_key")
        profile = self.store.load(profile_key) if profile_key else None
        created_at = self.now()

        if profile is None or current.get("status") != "READY":
            result = self._result(
                created_at=created_at,
                profile=profile,
                observation=observation,
                objectives=objectives,
                policy=policy,
                decision="SWITCH",
                actionable=True,
                reasons=["No READY profile is available; run autopilot before serving."],
                current_breaches=[],
                sustained_breaches=[],
            )
            self._save_plan(result)
            return result

        self._append_observation(
            {"profile_key": profile.profile_key, "observation": observation.to_dict()}
        )
        history = self._history(profile.profile_key)
        current_breaches = self._breaches(observation, objectives)
        sustained = self._sustained_breaches(
            history, objectives, policy.consecutive_breaches
        )
        environment_changed = bool(
            hardware_fingerprint
            and hardware_fingerprint != profile.hardware_fingerprint
        )
        requirements_changed = bool(
            desired_requirements is not None
            and desired_requirements != profile.requirements
        )

        if observation.healthy is False:
            decision = "SWITCH"
            reasons = ["The active service reported unhealthy."]
        elif requirements_changed:
            decision = self._requirement_decision(
                profile.requirements, desired_requirements or {}
            )
            reasons = ["The requested workload no longer matches this profile."]
        elif environment_changed:
            decision = "REBENCH"
            reasons = ["The execution environment fingerprint changed."]
        elif "quality" in sustained:
            decision = "SWITCH"
            reasons = ["Quality stayed below its acceptance threshold."]
        elif "error_rate" in sustained:
            decision = "REBENCH"
            reasons = ["The error rate stayed above its acceptance threshold."]
        elif sustained:
            decision = "RECONFIGURE"
            reasons = [
                "Runtime metrics repeatedly exceeded their configured thresholds: "
                + ", ".join(sustained)
                + "."
            ]
        elif current_breaches:
            decision = "KEEP"
            reasons = [
                "A breach was observed but has not persisted for "
                f"{policy.consecutive_breaches} consecutive observations."
            ]
        else:
            decision = "KEEP"
            reasons = ["The active profile remains within its observed thresholds."]

        actionable = decision != "KEEP"
        deferred = False
        if actionable and self._cooldown_active(
            profile.profile_key, created_at, policy.cooldown_seconds
        ):
            actionable = False
            deferred = True
            reasons.append("A recent adjustment plan is still inside the cooldown window.")

        result = self._result(
            created_at=created_at,
            profile=profile,
            observation=observation,
            objectives=objectives,
            policy=policy,
            decision=decision,
            actionable=actionable,
            reasons=reasons,
            current_breaches=current_breaches,
            sustained_breaches=sustained,
            environment_changed=environment_changed,
            requirements_changed=requirements_changed,
            deferred=deferred,
        )
        self._save_plan(result)
        return result

    @staticmethod
    def _breaches(
        observation: RuntimeObservation, objectives: ServiceObjectives
    ) -> List[str]:
        failures: List[str] = []
        checks = (
            ("request_p95_ms", observation.request_p95_ms, objectives.max_request_p95_ms, lambda value, limit: value > limit),
            ("ttft_p95_ms", observation.ttft_p95_ms, objectives.max_ttft_p95_ms, lambda value, limit: value > limit),
            ("throughput_tokens_s", observation.throughput_tokens_s, objectives.min_throughput_tokens_s, lambda value, limit: value < limit),
            ("error_rate", observation.error_rate, objectives.max_error_rate, lambda value, limit: value > limit),
            ("quality", observation.quality, objectives.min_quality, lambda value, limit: value < limit),
            ("available_memory_gb", observation.available_memory_gb, objectives.min_available_memory_gb, lambda value, limit: value < limit),
            ("queue_depth", observation.queue_depth, objectives.max_queue_depth, lambda value, limit: value > limit),
        )
        for name, value, limit, failed in checks:
            if value is not None and limit is not None and failed(value, limit):
                failures.append(name)
        return failures

    def _sustained_breaches(
        self,
        history: List[RuntimeObservation],
        objectives: ServiceObjectives,
        count: int,
    ) -> List[str]:
        if len(history) < count:
            return []
        windows = [set(self._breaches(item, objectives)) for item in history[-count:]]
        return sorted(set.intersection(*windows)) if windows else []

    @staticmethod
    def _requirement_decision(
        previous: Dict[str, Any], desired: Dict[str, Any]
    ) -> str:
        capability_axes = {"task", "capabilities", "modalities", "privacy"}
        if any(previous.get(key) != desired.get(key) for key in capability_axes):
            return "SWITCH"
        return "RECONFIGURE"

    def _result(
        self,
        *,
        created_at: str,
        profile: Optional[SavedProfile],
        observation: RuntimeObservation,
        objectives: ServiceObjectives,
        policy: ReconcilePolicy,
        decision: str,
        actionable: bool,
        reasons: List[str],
        current_breaches: List[str],
        sustained_breaches: List[str],
        environment_changed: bool = False,
        requirements_changed: bool = False,
        deferred: bool = False,
    ) -> ReconcileResult:
        high_risk = decision == "SWITCH"
        changes_runtime = decision in {"RECONFIGURE", "SWITCH"}
        return ReconcileResult(
            plan_id=str(uuid.uuid4()),
            created_at=created_at,
            profile_key=profile.profile_key if profile else None,
            decision=decision,
            actionable=actionable,
            deferred_by_cooldown=deferred,
            reasons=reasons,
            current_breaches=current_breaches,
            sustained_breaches=sustained_breaches,
            consecutive_required=policy.consecutive_breaches,
            thresholds=objectives.to_dict(),
            observation=observation.to_dict(),
            environment_changed=environment_changed,
            requirements_changed=requirements_changed,
            safe_execution={
                "automatic_apply": False,
                "requires_validation": decision != "KEEP",
                "requires_drain": changes_runtime,
                "requires_canary": high_risk,
                "rollback_profile_key": profile.profile_key if profile else None,
                "note": (
                    "This plan records a decision only; reconcile never mutates "
                    "the active serving configuration."
                ),
            },
        )

    def _append_observation(self, record: Dict[str, Any]) -> bool:
        self.observation_path.parent.mkdir(parents=True, exist_ok=True)
        sequence = (record.get("observation") or {}).get("source_sequence")
        if sequence is not None and self.observation_path.exists():
            for line in reversed(
                self.observation_path.read_text(encoding="utf-8").splitlines()
            ):
                try:
                    previous = json.loads(line)
                except (ValueError, TypeError, json.JSONDecodeError):
                    continue
                if previous.get("profile_key") != record.get("profile_key"):
                    continue
                previous_sequence = (previous.get("observation") or {}).get(
                    "source_sequence"
                )
                if previous_sequence == sequence:
                    return False
                break
        with self.observation_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return True

    def _history(self, profile_key: str) -> List[RuntimeObservation]:
        if not self.observation_path.exists():
            return []
        observations: List[RuntimeObservation] = []
        for line in self.observation_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                if record.get("profile_key") != profile_key:
                    continue
                observations.append(
                    RuntimeObservation.from_mapping(record.get("observation") or {})
                )
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
        return observations

    def _cooldown_active(
        self, profile_key: str, created_at: str, cooldown_seconds: int
    ) -> bool:
        if cooldown_seconds == 0 or not self.plan_root.exists():
            return False
        current_time = _parse_time(created_at)
        for path in sorted(
            self.plan_root.glob("plan-*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        ):
            try:
                plan = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            if plan.get("profile_key") != profile_key:
                continue
            if plan.get("decision") == "KEEP" or not plan.get("actionable"):
                continue
            elapsed = (current_time - _parse_time(plan["created_at"])).total_seconds()
            return 0 <= elapsed < cooldown_seconds
        return False

    def _save_plan(self, result: ReconcileResult) -> Path:
        path = self.plan_root / f"plan-{result.plan_id}.json"
        atomic_write_json(path, result.to_dict())
        return path
