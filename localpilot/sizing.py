from __future__ import annotations

from typing import Any, Dict, Optional

from localpilot.schemas import HardwareProfile, MemoryEstimate, ModelSpec


GIB = 1024**3

# Bytes actually resident per parameter, including the block scales that
# sub-byte formats carry. NVFP4 packs two values per byte and adds one FP8
# scale per 16-element block, so the honest figure is 0.5 + 1/16.
DEFAULT_BYTES_PER_PARAMETER = {
    "NVFP4": 0.5625,
    "FP4": 0.5625,
    "MXFP4": 0.5625,
    "INT4": 0.5625,
    "AWQ_INT4": 0.5625,
    "GGUF_Q4_K_S": 0.55,
    "GGUF_Q4_K_M": 0.60,
    "FP8": 1.0,
    "INT8": 1.0,
    "BF16": 2.0,
    "FP16": 2.0,
    "FP32": 4.0,
}

# Embeddings, norms, routers and the vision tower usually stay in higher
# precision even in a quantized checkpoint.
UNQUANTIZED_TAIL_FRACTION = 0.08

KV_DTYPE_SCALE = {"auto": 1.0, "fp16": 1.0, "bf16": 1.0, "fp8": 0.5, "int8": 0.5}


class MemoryModel:
    """Estimates resident memory for a (model, precision, context, batch).

    The estimate only has to be good enough to gate candidates before they
    run. Every surviving candidate is measured for real, and the measured
    peak is what gets written into the profile.
    """

    def __init__(self, config: Dict[str, Any] = None) -> None:
        config = config or {}
        table = dict(DEFAULT_BYTES_PER_PARAMETER)
        table.update(config.get("bytes_per_parameter", {}))
        self.bytes_per_parameter = {
            key.upper(): float(value) for key, value in table.items()
        }
        self.safety_reserve_percent = float(
            config.get("safety_reserve_percent", 20)
        )
        self.activation_floor_gb = float(config.get("activation_floor_gb", 1.5))
        self.activation_fraction = float(config.get("activation_fraction", 0.06))
        self.unquantized_tail_fraction = float(
            config.get("unquantized_tail_fraction", UNQUANTIZED_TAIL_FRACTION)
        )

    def bytes_for(self, precision: str) -> float:
        key = (precision or "BF16").upper()
        if key in self.bytes_per_parameter:
            return self.bytes_per_parameter[key]
        for candidate, value in self.bytes_per_parameter.items():
            if key.startswith(candidate) or candidate in key:
                return value
        return self.bytes_per_parameter["BF16"]

    def weights_gb(
        self,
        model: ModelSpec,
        precision: str = None,
        speculative_decoding: bool = False,
    ) -> float:
        """Resident weight bytes.

        A published checkpoint's real file size beats any formula, because
        quantized repos keep an unpredictable tail of layers -- embeddings,
        routers, vision towers -- at higher precision. The derivation below
        is only a fallback for models with no measured size.
        """
        if model.weights_gb and (precision is None or precision == model.precision):
            total = model.weights_gb
        else:
            per_parameter = self.bytes_for(precision or model.precision)
            parameters = model.parameter_count_b * 1e9
            quantized = (
                parameters * (1 - self.unquantized_tail_fraction) * per_parameter
            )
            tail = parameters * self.unquantized_tail_fraction * 2.0
            total = (quantized + tail) / GIB
        if speculative_decoding:
            total += model.draft_weights_gb
        return total

    def kv_cache_gb(
        self,
        model: ModelSpec,
        context_length: int,
        concurrency: int = 1,
        kv_cache_dtype: str = "auto",
    ) -> float:
        scale = KV_DTYPE_SCALE.get((kv_cache_dtype or "auto").lower(), 1.0)
        total_bytes = (
            model.kv_bytes_per_token
            * max(1, context_length)
            * max(1, concurrency)
            * scale
        )
        # Hybrid Mamba blocks carry a recurrent state instead of a KV cache.
        # It is constant in context length but charged per sequence.
        total_bytes += model.state_mb_per_sequence * (1024**2) * max(1, concurrency)
        return total_bytes / GIB

    def budget_gb(self, hardware: HardwareProfile) -> Optional[float]:
        pool = self._memory_pool_gb(hardware)
        if pool is None:
            return None
        return pool * (1 - self.safety_reserve_percent / 100)

    def _memory_pool_gb(self, hardware: HardwareProfile) -> Optional[float]:
        """The pool a model can actually occupy.

        With unified memory the accelerator and the OS draw on one pool, so
        the ceiling is system memory. With a discrete GPU the ceiling is
        whatever that board carries, regardless of host RAM.
        """
        if not hardware.unified_memory:
            dedicated = hardware.accelerator.get("memory_total_gb")
            if dedicated:
                return float(dedicated)
        total = hardware.memory.get("total_gb")
        return float(total) if total else None

    def estimate(
        self,
        model: ModelSpec,
        hardware: HardwareProfile,
        context_length: int,
        concurrency: int = 1,
        precision: str = None,
        kv_cache_dtype: str = "auto",
        speculative_decoding: bool = False,
    ) -> MemoryEstimate:
        weights = self.weights_gb(model, precision, speculative_decoding)
        kv_cache = self.kv_cache_gb(
            model, context_length, concurrency, kv_cache_dtype
        )
        activation = max(self.activation_floor_gb, weights * self.activation_fraction)
        total = weights + kv_cache + activation
        budget = self.budget_gb(hardware)

        if budget is None:
            return MemoryEstimate(
                weights_gb=round(weights, 2),
                kv_cache_gb=round(kv_cache, 2),
                activation_overhead_gb=round(activation, 2),
                total_gb=round(total, 2),
                budget_gb=0.0,
                fits=True,
                detail="memory pool unknown; gate skipped",
            )

        fits = total <= budget
        if fits:
            detail = f"{total:.1f} GB of a {budget:.1f} GB budget"
        else:
            detail = (
                f"needs {total:.1f} GB but only {budget:.1f} GB is safely "
                f"available ({weights:.1f} GB weights + {kv_cache:.1f} GB KV "
                f"at {context_length} tokens x{concurrency})"
            )
        return MemoryEstimate(
            weights_gb=round(weights, 2),
            kv_cache_gb=round(kv_cache, 2),
            activation_overhead_gb=round(activation, 2),
            total_gb=round(total, 2),
            budget_gb=round(budget, 2),
            fits=fits,
            detail=detail,
        )

    def max_context_for(
        self,
        model: ModelSpec,
        hardware: HardwareProfile,
        concurrency: int = 1,
        precision: str = None,
        kv_cache_dtype: str = "auto",
        speculative_decoding: bool = False,
    ) -> int:
        """Largest context that still fits, rounded down to 1K tokens."""
        budget = self.budget_gb(hardware)
        if budget is None:
            return model.context_length
        weights = self.weights_gb(model, precision, speculative_decoding)
        activation = max(self.activation_floor_gb, weights * self.activation_fraction)
        state_gb = (
            model.state_mb_per_sequence * (1024**2) * max(1, concurrency) / GIB
        )
        remaining_gb = budget - weights - activation - state_gb
        if remaining_gb <= 0:
            return 0
        scale = KV_DTYPE_SCALE.get((kv_cache_dtype or "auto").lower(), 1.0)
        per_token = model.kv_bytes_per_token * max(1, concurrency) * scale
        if per_token <= 0:
            return model.context_length
        tokens = int(remaining_gb * GIB / per_token)
        tokens = (tokens // 1024) * 1024
        return max(0, min(tokens, model.context_length))


def decode_roofline_tokens_s(
    active_parameter_count_b: float,
    precision: str,
    memory_bandwidth_gbps: Optional[float],
    efficiency: float = 0.75,
    memory_model: MemoryModel = None,
) -> Optional[float]:
    """Upper bound on single-stream decode speed.

    Generating one token reads every active weight exactly once, so the
    ceiling is bandwidth divided by active bytes. This is what makes a
    sparse model fast on a bandwidth-starved machine: capacity is charged
    on total parameters, speed only on the active ones.
    """
    if not memory_bandwidth_gbps or active_parameter_count_b <= 0:
        return None
    model = memory_model or MemoryModel()
    bytes_per_parameter = model.bytes_for(precision)
    bytes_per_token = active_parameter_count_b * 1e9 * bytes_per_parameter
    if bytes_per_token <= 0:
        return None
    bytes_per_second = memory_bandwidth_gbps * 1e9 * efficiency
    return bytes_per_second / bytes_per_token
