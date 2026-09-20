from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


class Serializable:
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Intent(Serializable):
    task: str = "chat"
    privacy: str = "local_only"
    priority: str = "balanced"
    quality: str = "high"
    context_length: int = 8192
    concurrency: int = 1
    preferred_language: List[str] = field(default_factory=lambda: ["en"])
    capabilities: List[str] = field(default_factory=list)
    raw_text: str = field(default="", repr=False)


@dataclass
class DeviceInfo(Serializable):
    id: str
    kind: str
    name: str
    available: bool
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HardwareProfile(Serializable):
    os: Dict[str, Any]
    cpu: Dict[str, Any]
    gpu: Dict[str, Any]
    npu: Dict[str, Any]
    memory: Dict[str, Any]
    disk: Dict[str, Any]
    openvino: Dict[str, Any]
    available_devices: List[str]
    devices: List[DeviceInfo]
    fingerprint: str
    simulated: bool = False
    real_execution_ready: bool = False
    notes: List[str] = field(default_factory=list)


@dataclass
class ModelSpec(Serializable):
    model_id: str
    source_id: str
    task: str
    parameter_count_b: float
    disk_size_gb: float
    precision: str
    supported_devices: List[str]
    minimum_ram_gb: float
    recommended_ram_gb: float
    context_length: int
    quality_score: float
    openvino_compatible: bool
    validation: str
    variants: Dict[str, Dict[str, Any]]


@dataclass
class CandidatePlan(Serializable):
    candidate_id: str
    model_id: str
    source_id: str
    device: str
    precision: str
    context_length: int
    runtime: str
    expected_memory_gb: float
    quality_score: float
    reason: str
    confidence: float
    simulated: bool
    model_path: Optional[str] = None
    runtime_config: Dict[str, Any] = field(default_factory=dict)
    fallback_of: Optional[str] = None
    recovery_action: Optional[str] = None


@dataclass
class BenchmarkMetrics(Serializable):
    ttft_ms: float
    tpot_ms: float
    throughput_tokens_s: float
    total_latency_ms: float
    peak_memory_gb: float
    cpu_usage_percent: Optional[float]
    stability: float
    quality: float
    runs: int
    warmup_runs: int
    simulated: bool
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepStatus(Serializable):
    name: str
    status: str
    message: str = ""


@dataclass
class CandidateResult(Serializable):
    candidate: CandidatePlan
    status: str
    steps: List[StepStatus]
    benchmark: Optional[BenchmarkMetrics] = None
    score: Optional[float] = None
    error: Optional[str] = None


@dataclass
class SavedProfile(Serializable):
    profile_key: str
    hardware_fingerprint: str
    task: str
    priority: str
    candidate: CandidatePlan
    benchmark: BenchmarkMetrics
    score: float
    runtime_versions: Dict[str, Any]
    simulated: bool
    created_at: str
    last_verified_at: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SavedProfile":
        return cls(
            profile_key=data["profile_key"],
            hardware_fingerprint=data["hardware_fingerprint"],
            task=data["task"],
            priority=data["priority"],
            candidate=CandidatePlan(**data["candidate"]),
            benchmark=BenchmarkMetrics(**data["benchmark"]),
            score=float(data["score"]),
            runtime_versions=dict(data.get("runtime_versions", {})),
            simulated=bool(data["simulated"]),
            created_at=data["created_at"],
            last_verified_at=data["last_verified_at"],
        )


@dataclass
class AutopilotResult(Serializable):
    run_id: str
    intent: Intent
    hardware: HardwareProfile
    candidates: List[CandidateResult]
    best_profile: SavedProfile
    profile_reused: bool
    status: str
    warnings: List[str] = field(default_factory=list)
