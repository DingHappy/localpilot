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
    modalities: List[str] = field(default_factory=lambda: ["text"])
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
    accelerator: Dict[str, Any]
    memory: Dict[str, Any]
    disk: Dict[str, Any]
    stack: Dict[str, Any]
    engines: Dict[str, Any]
    available_devices: List[str]
    devices: List[DeviceInfo]
    fingerprint: str
    platform_id: str = "unknown"
    unified_memory: bool = False
    memory_bandwidth_gbps: Optional[float] = None
    simulated: bool = False
    real_execution_ready: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def usable_devices(self) -> List[str]:
        return [device.split(".", 1)[0] for device in self.available_devices]


@dataclass
class ModelSpec(Serializable):
    """A servable model artifact.

    ``parameter_count_b`` gates capacity because every expert has to be
    resident. ``active_parameter_count_b`` gates decode speed because only
    the routed experts are read per token. On a unified-memory machine the
    two diverge sharply, which is the whole reason a mixture-of-experts
    model behaves differently here than on a discrete-GPU box.
    """

    model_id: str
    source_id: str
    tasks: List[str]
    vendor: str
    parameter_count_b: float
    active_parameter_count_b: float
    precision: str
    weights_gb: float
    context_length: int
    quality_score: float
    engines: List[str]
    devices: List[str]
    kv_bytes_per_token: float
    modalities: List[str] = field(default_factory=lambda: ["text"])
    capabilities: List[str] = field(default_factory=list)
    state_mb_per_sequence: float = 0.0
    draft_source_id: Optional[str] = None
    draft_weights_gb: float = 0.0
    draft_parameter_count_b: float = 0.0
    validation: str = "unverified_on_target"
    notes: str = ""

    @property
    def is_mixture_of_experts(self) -> bool:
        return self.active_parameter_count_b < self.parameter_count_b * 0.9

    @property
    def supports_speculative_decoding(self) -> bool:
        return bool(self.draft_source_id)

    def serves(self, task: str) -> bool:
        return task in self.tasks


@dataclass
class MemoryEstimate(Serializable):
    weights_gb: float
    kv_cache_gb: float
    activation_overhead_gb: float
    total_gb: float
    budget_gb: float
    fits: bool
    detail: str = ""


@dataclass
class CandidatePlan(Serializable):
    candidate_id: str
    model_id: str
    source_id: str
    engine: str
    device: str
    precision: str
    context_length: int
    runtime: str
    expected_memory_gb: float
    quality_score: float
    reason: str
    confidence: float
    simulated: bool
    concurrency: int = 1
    kv_cache_dtype: str = "auto"
    active_parameter_count_b: float = 0.0
    parameter_count_b: float = 0.0
    memory_estimate: Optional[MemoryEstimate] = None
    model_path: Optional[str] = None
    runtime_config: Dict[str, Any] = field(default_factory=dict)
    knobs: Dict[str, Any] = field(default_factory=dict)
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
    ttft_p95_ms: Optional[float] = None
    concurrency: int = 1
    aggregate_throughput_tokens_s: Optional[float] = None
    quality_keyword: Optional[float] = None
    quality_judge: Optional[float] = None
    decode_bandwidth_gbps: Optional[float] = None
    bandwidth_utilization: Optional[float] = None


@dataclass
class StepStatus(Serializable):
    name: str
    status: str
    message: str = ""


@dataclass
class AgentStep(Serializable):
    agent: str
    action: str
    status: str
    detail: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateResult(Serializable):
    candidate: CandidatePlan
    status: str
    steps: List[StepStatus]
    benchmark: Optional[BenchmarkMetrics] = None
    score: Optional[float] = None
    score_components: Dict[str, float] = field(default_factory=dict)
    gate_failures: List[str] = field(default_factory=list)
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
    platform_id: str = "unknown"
    requirements: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SavedProfile":
        candidate = dict(data["candidate"])
        estimate = candidate.get("memory_estimate")
        if isinstance(estimate, dict):
            candidate["memory_estimate"] = MemoryEstimate(**estimate)
        return cls(
            profile_key=data["profile_key"],
            hardware_fingerprint=data["hardware_fingerprint"],
            task=data["task"],
            priority=data["priority"],
            candidate=CandidatePlan(**candidate),
            benchmark=BenchmarkMetrics(**data["benchmark"]),
            score=float(data["score"]),
            runtime_versions=dict(data.get("runtime_versions", {})),
            simulated=bool(data["simulated"]),
            created_at=data["created_at"],
            last_verified_at=data["last_verified_at"],
            platform_id=data.get("platform_id", "unknown"),
            requirements=dict(data.get("requirements", {})),
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
    agent_trace: List[AgentStep] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""
