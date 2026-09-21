from __future__ import annotations

import threading
import time
from typing import Any, Dict


class TrafficGate:
    """Stops new work and lets in-flight requests finish before a change."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._accepting = True
        self._active = 0

    def acquire(self) -> bool:
        with self._condition:
            if not self._accepting:
                return False
            self._active += 1
            return True

    def release(self) -> None:
        with self._condition:
            self._active = max(0, self._active - 1)
            if self._active == 0:
                self._condition.notify_all()

    def drain(self, timeout: float) -> Dict[str, Any]:
        if timeout < 0:
            raise ValueError("drain timeout cannot be negative")
        deadline = time.monotonic() + timeout
        with self._condition:
            self._accepting = False
            while self._active:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    # A failed drain is a failed change. Resume immediately so
                    # an operator cannot accidentally leave the service dark.
                    self._accepting = True
                    self._condition.notify_all()
                    return {
                        "status": "TIMEOUT",
                        "drained": False,
                        "accepting": True,
                        "active_requests": self._active,
                    }
                self._condition.wait(timeout=remaining)
            return {
                "status": "DRAINED",
                "drained": True,
                "accepting": False,
                "active_requests": 0,
            }

    def resume(self) -> Dict[str, Any]:
        with self._condition:
            self._accepting = True
            self._condition.notify_all()
            return self.status()

    def status(self) -> Dict[str, Any]:
        with self._condition:
            return {
                "accepting": self._accepting,
                "active_requests": self._active,
                "state": "SERVING" if self._accepting else "DRAINING",
            }
