from __future__ import annotations

import threading
import traceback
import uuid
from typing import Any, Dict, List, Optional

from localpilot.orchestrator import Orchestrator, AutopilotRejected
from localpilot.utils import utc_now


class AutopilotJob:
    """One autopilot run, tracked so a caller can poll it.

    A real search starts servers and measures them, which takes minutes.
    Holding an HTTP request open for that is fragile, so the run happens on
    a worker thread and the client polls for the trace as it fills.
    """

    def __init__(self, goal: str, mode: str, reuse_profile: bool) -> None:
        self.job_id = str(uuid.uuid4())
        self.goal = goal
        self.mode = mode
        self.reuse_profile = reuse_profile
        self.status = "queued"
        self.created_at = utc_now()
        self.finished_at: Optional[str] = None
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        # Filled step by step while the run is in flight, so a poller has
        # something to show long before the result exists.
        self.trace: List[Dict[str, Any]] = []
        self._trace_lock = threading.Lock()

    def append_step(self, step) -> None:
        with self._trace_lock:
            self.trace.append(step.to_dict())

    def snapshot_trace(self) -> List[Dict[str, Any]]:
        with self._trace_lock:
            return list(self.trace)

    def to_dict(self, include_result: bool = True) -> Dict[str, Any]:
        payload = {
            "job_id": self.job_id,
            "goal": self.goal,
            "mode": self.mode,
            "status": self.status,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }
        if include_result:
            payload["trace"] = self.snapshot_trace()
            payload["result"] = self.result
        return payload


class JobRegistry:
    def __init__(self, orchestrator_factory=Orchestrator, max_history: int = 50):
        self.orchestrator_factory = orchestrator_factory
        self.max_history = max_history
        self._jobs: Dict[str, AutopilotJob] = {}
        self._order: List[str] = []
        self._lock = threading.Lock()

    def submit(
        self, goal: str, mode: str = "auto", reuse_profile: bool = True
    ) -> AutopilotJob:
        job = AutopilotJob(goal, mode, reuse_profile)
        with self._lock:
            self._jobs[job.job_id] = job
            self._order.append(job.job_id)
            while len(self._order) > self.max_history:
                self._jobs.pop(self._order.pop(0), None)

        thread = threading.Thread(target=self._run, args=(job,), daemon=True)
        thread.start()
        return job

    def _run(self, job: AutopilotJob) -> None:
        job.status = "running"
        try:
            orchestrator = self.orchestrator_factory()
            result = orchestrator.autopilot(
                job.goal,
                mode=job.mode,
                reuse_profile=job.reuse_profile,
                progress=job.append_step,
            )
            job.result = result.to_dict()
            job.status = "succeeded"
        except AutopilotRejected as exc:
            job.error = str(exc)
            job.result = exc.run
            job.status = "failed"
        except Exception as exc:
            job.error = f"{type(exc).__name__}: {exc}"
            job.result = {"traceback": traceback.format_exc(limit=8)}
            job.status = "failed"
        finally:
            job.finished_at = utc_now()

    def get(self, job_id: str) -> Optional[AutopilotJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def recent(self, limit: int = 20) -> List[AutopilotJob]:
        with self._lock:
            ids = list(reversed(self._order))[:limit]
            return [self._jobs[job_id] for job_id in ids if job_id in self._jobs]
