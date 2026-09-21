from __future__ import annotations

import math
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional

from localpilot.utils import utc_now


def _percentile(values: list[float], quantile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[rank]


@dataclass(frozen=True)
class RequestToken:
    started_at: float


class RequestTelemetry:
    """Thread-safe aggregate request telemetry with no prompt or output data."""

    def __init__(self, max_samples: int = 200) -> None:
        if max_samples < 1:
            raise ValueError("max_samples must be at least 1")
        self.max_samples = max_samples
        self._lock = threading.Lock()
        self._source_id = uuid.uuid4().hex
        self._samples: Deque[tuple[float, bool]] = deque(maxlen=max_samples)
        self._active_requests = 0
        self._total_requests = 0
        self._total_errors = 0

    def begin(self) -> RequestToken:
        with self._lock:
            self._active_requests += 1
        return RequestToken(started_at=time.perf_counter())

    def complete(self, token: RequestToken, *, ok: bool) -> None:
        latency_ms = max(0.0, (time.perf_counter() - token.started_at) * 1000)
        with self._lock:
            self._active_requests = max(0, self._active_requests - 1)
            self._total_requests += 1
            if not ok:
                self._total_errors += 1
            self._samples.append((latency_ms, ok))

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            samples = list(self._samples)
            active = self._active_requests
            total = self._total_requests
            errors = self._total_errors
        latencies = [latency for latency, _ok in samples]
        window_errors = sum(1 for _latency, ok in samples if not ok)
        window_requests = len(samples)
        return {
            "timestamp": utc_now(),
            "source_sequence": f"{self._source_id}:{total}",
            "window_requests": window_requests,
            "window_size": self.max_samples,
            "request_p95_ms": (
                round(_percentile(latencies, 0.95), 3) if latencies else None
            ),
            "error_rate": (
                round(window_errors / window_requests, 6)
                if window_requests
                else None
            ),
            "active_requests": active,
            "total_requests": total,
            "total_errors": errors,
        }
