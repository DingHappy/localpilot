from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from localpilot.engines.registry import configured_base_url


@dataclass
class ServerConfig:
    """What an attached server is actually running.

    In attach mode a candidate's knobs are declarations, not facts: the
    server was started by someone else with one configuration, and every
    request is answered by that configuration whatever the plan says. So
    the plan has to be reconciled against this, or a "comparison" between
    two knob settings is two measurements of the same setting.
    """

    base_url: str
    served_models: List[str] = field(default_factory=list)
    block_size: Optional[int] = None
    num_gpu_blocks: Optional[int] = None
    cache_dtype: Optional[str] = None
    gpu_memory_utilization: Optional[float] = None
    max_model_len: Optional[int] = None
    prefix_cache_queries: Optional[float] = None
    prefix_cache_hits: Optional[float] = None
    raw_available: bool = False

    @property
    def kv_capacity_tokens(self) -> Optional[int]:
        if self.block_size and self.num_gpu_blocks:
            return self.block_size * self.num_gpu_blocks
        return None

    def kv_allocated_gb(self, kv_bytes_per_token: float) -> Optional[float]:
        """Bytes the engine really reserved, not what the planner guessed."""
        tokens = self.kv_capacity_tokens
        if not tokens or kv_bytes_per_token <= 0:
            return None
        return tokens * kv_bytes_per_token / (1024**3)

    @property
    def prefix_cache_hit_rate(self) -> Optional[float]:
        if not self.prefix_cache_queries:
            return None
        return (self.prefix_cache_hits or 0.0) / self.prefix_cache_queries

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_url": self.base_url,
            "served_models": list(self.served_models),
            "block_size": self.block_size,
            "num_gpu_blocks": self.num_gpu_blocks,
            "kv_capacity_tokens": self.kv_capacity_tokens,
            "cache_dtype": self.cache_dtype,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "prefix_cache_hit_rate": (
                round(self.prefix_cache_hit_rate, 4)
                if self.prefix_cache_hit_rate is not None
                else None
            ),
            "introspected": self.raw_available,
        }


def _fetch(url: str, timeout: float = 6.0) -> Optional[str]:
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, method="GET"), timeout=timeout
        ) as response:
            return response.read().decode("utf-8", "replace")
    except Exception:
        return None


_LABEL = re.compile(r'(\w+)="([^"]*)"')


def _parse_prometheus(text: str) -> Dict[str, Any]:
    """Pulls the few values that describe the running configuration.

    Deliberately not a general Prometheus parser: only the labels of
    `cache_config_info` and a couple of counters are needed, and a tolerant
    reader keeps working when the exposition changes shape.
    """
    found: Dict[str, Any] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if "cache_config_info" in line:
            for key, value in _LABEL.findall(line):
                found.setdefault(key, value)
        elif "prefix_cache_queries_total" in line and "external" not in line:
            found["prefix_cache_queries"] = _tail_float(line)
        elif "prefix_cache_hits_total" in line and "external" not in line:
            found["prefix_cache_hits"] = _tail_float(line)
    return found


def _tail_float(line: str) -> Optional[float]:
    try:
        return float(line.rsplit(" ", 1)[1])
    except (IndexError, ValueError):
        return None


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def introspect(
    engine_id: str, base_path: str = "/v1", base_url: str = None
) -> Optional[ServerConfig]:
    """Asks a running server what it is, rather than assuming.

    Returns None when no server is configured, so callers can tell "not
    attached" from "attached but silent".
    """
    url = base_url or configured_base_url(engine_id)
    if not url:
        return None
    config = ServerConfig(base_url=url)

    listing = _fetch(f"{url}{(base_path or '/v1').rstrip('/')}/models")
    if listing:
        try:
            payload = json.loads(listing)
        except ValueError:
            payload = {}
        for entry in (payload or {}).get("data", []):
            if entry.get("id"):
                config.served_models.append(entry["id"])
            if config.max_model_len is None:
                config.max_model_len = _as_int(entry.get("max_model_len"))

    metrics = _fetch(f"{url}/metrics")
    if metrics:
        found = _parse_prometheus(metrics)
        config.raw_available = bool(found)
        config.block_size = _as_int(found.get("block_size"))
        config.num_gpu_blocks = _as_int(found.get("num_gpu_blocks"))
        raw_dtype = found.get("cache_dtype")
        config.cache_dtype = raw_dtype if raw_dtype else None
        config.gpu_memory_utilization = _as_float(
            found.get("gpu_memory_utilization")
        )
        config.prefix_cache_queries = found.get("prefix_cache_queries")
        config.prefix_cache_hits = found.get("prefix_cache_hits")
    return config


def reconcile(candidate, config: ServerConfig) -> List[str]:
    """Lists the ways a candidate's declared knobs differ from reality.

    An attached server cannot be reconfigured per candidate, so any
    difference here means the candidate would be measured as something
    other than what it claims to be.
    """
    if config is None or not config.raw_available:
        return []

    mismatches: List[str] = []

    declared_kv = (candidate.kv_cache_dtype or "auto").lower()
    actual_kv = (config.cache_dtype or "auto").lower()
    if declared_kv != actual_kv:
        mismatches.append(
            f"KV cache dtype declared {declared_kv}, server runs {actual_kv}"
        )

    declared_util = (candidate.runtime_config or {}).get(
        "gpu_memory_utilization"
    )
    if (
        declared_util is not None
        and config.gpu_memory_utilization is not None
        and abs(float(declared_util) - config.gpu_memory_utilization) > 0.01
    ):
        mismatches.append(
            f"gpu_memory_utilization declared {declared_util}, server runs "
            f"{config.gpu_memory_utilization}"
        )

    if (
        config.max_model_len
        and candidate.context_length
        and candidate.context_length > config.max_model_len
    ):
        mismatches.append(
            f"context {candidate.context_length} exceeds the server's "
            f"max_model_len {config.max_model_len}"
        )

    if (candidate.runtime_config or {}).get("speculative"):
        mismatches.append(
            "speculative decoding declared, but an attached server cannot be "
            "given a draft model per candidate"
        )
    return mismatches
