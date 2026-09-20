"""Shared fixtures for the LocalPilot test suite."""

from __future__ import annotations

from localpilot.schemas import (
    BenchmarkMetrics,
    CandidatePlan,
    CandidateResult,
    DeviceInfo,
    HardwareProfile,
    ModelSpec,
)


def spark_profile(memory_gb: float = 128.0, simulated: bool = True) -> HardwareProfile:
    """A DGX Spark-shaped profile: one accelerator, one unified pool."""
    return HardwareProfile(
        os={"name": "Linux", "release": "6.11", "version": "", "machine": "aarch64"},
        cpu={"model": "Grace", "architecture": "aarch64", "logical_cores": 20,
             "workload": {}},
        accelerator={
            "detected": True,
            "source": "test",
            "device_count": 1,
            "names": ["NVIDIA GB10"],
            "memory_total_gb": memory_gb,
            "driver_version": "580.00",
        },
        memory={"total_gb": memory_gb, "available_gb": memory_gb - 8},
        disk={"total_gb": 4000.0, "free_gb": 3000.0},
        stack={"platform": "dgx_spark", "engines_available": ["vllm", "trtllm"],
               "cuda": {"nvcc_version": "13.0"}},
        engines={
            "vllm": {"available": True},
            "trtllm": {"available": True},
        },
        available_devices=["CUDA", "CPU"],
        devices=[
            DeviceInfo("CUDA", "gpu", "NVIDIA GB10", True,
                       {"memory_total_gb": memory_gb}),
            DeviceInfo("CPU", "cpu", "Grace", True),
        ],
        fingerprint="test-spark-fingerprint",
        platform_id="dgx_spark",
        unified_memory=True,
        memory_bandwidth_gbps=273.0,
        simulated=simulated,
        real_execution_ready=not simulated,
    )


def discrete_profile(vram_gb: float = 24.0) -> HardwareProfile:
    """A discrete GPU: host RAM is plentiful, the board's VRAM is the ceiling."""
    profile = spark_profile(memory_gb=256.0)
    profile.accelerator = dict(profile.accelerator)
    profile.accelerator["names"] = ["NVIDIA RTX 6000"]
    profile.accelerator["memory_total_gb"] = vram_gb
    profile.platform_id = "cuda_discrete"
    profile.unified_memory = False
    profile.memory_bandwidth_gbps = 960.0
    return profile


def model(
    model_id: str = "test-moe",
    parameter_count_b: float = 30.0,
    active_parameter_count_b: float = 3.0,
    precision: str = "NVFP4",
    weights_gb: float = 20.0,
    kv_bytes_per_token: float = 7168,
    tasks=("coding", "chat"),
    engines=("vllm", "trtllm"),
    draft_source_id: str = None,
    context_length: int = 131072,
    quality_score: float = 0.85,
) -> ModelSpec:
    return ModelSpec(
        model_id=model_id,
        source_id=f"test/{model_id}",
        tasks=list(tasks),
        vendor="test",
        parameter_count_b=parameter_count_b,
        active_parameter_count_b=active_parameter_count_b,
        precision=precision,
        weights_gb=weights_gb,
        context_length=context_length,
        quality_score=quality_score,
        engines=list(engines),
        devices=["CUDA"],
        kv_bytes_per_token=kv_bytes_per_token,
        draft_source_id=draft_source_id,
        draft_weights_gb=1.3 if draft_source_id else 0.0,
    )


def candidate(
    candidate_id: str = "test-candidate",
    engine: str = "vllm",
    precision: str = "NVFP4",
    runtime: str = "mock",
    context_length: int = 8192,
    concurrency: int = 1,
    simulated: bool = True,
    speculative: bool = False,
    parameter_count_b: float = 30.0,
    active_parameter_count_b: float = 3.0,
    expected_memory_gb: float = 22.0,
) -> CandidatePlan:
    runtime_config = {
        "max_num_seqs": concurrency,
        "gpu_memory_utilization": 0.9,
        "kv_cache_dtype": "auto",
        "enable_prefix_caching": False,
    }
    if speculative:
        runtime_config["speculative"] = {
            "draft_source_id": "test/draft",
            "num_speculative_tokens": 3,
        }
    return CandidatePlan(
        candidate_id=candidate_id,
        model_id="test-moe",
        source_id="test/test-moe",
        engine=engine,
        device="CUDA",
        precision=precision,
        context_length=context_length,
        runtime=runtime,
        expected_memory_gb=expected_memory_gb,
        quality_score=0.85,
        reason="test",
        confidence=0.9,
        simulated=simulated,
        concurrency=concurrency,
        active_parameter_count_b=active_parameter_count_b,
        parameter_count_b=parameter_count_b,
        runtime_config=runtime_config,
    )


def metrics(
    ttft_ms: float = 100.0,
    throughput_tokens_s: float = 100.0,
    peak_memory_gb: float = 20.0,
    quality: float = 0.85,
    stability: float = 1.0,
    concurrency: int = 1,
    aggregate: float = None,
) -> BenchmarkMetrics:
    return BenchmarkMetrics(
        ttft_ms=ttft_ms,
        tpot_ms=1000 / throughput_tokens_s,
        throughput_tokens_s=throughput_tokens_s,
        total_latency_ms=ttft_ms * 3,
        peak_memory_gb=peak_memory_gb,
        cpu_usage_percent=None,
        stability=stability,
        quality=quality,
        runs=3,
        warmup_runs=1,
        simulated=True,
        concurrency=concurrency,
        aggregate_throughput_tokens_s=aggregate,
    )


def result(candidate_plan: CandidatePlan, benchmark: BenchmarkMetrics) -> CandidateResult:
    return CandidateResult(candidate_plan, "success", [], benchmark)
