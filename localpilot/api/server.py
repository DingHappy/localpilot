from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict

from localpilot.api.jobs import JobRegistry
from localpilot.benchmark.prompts import load_benchmark_config
from localpilot.control.reconcile import (
    ReconcilePolicy,
    Reconciler,
    RuntimeObservation,
    ServiceObjectives,
)
from localpilot.control.telemetry import RequestTelemetry
from localpilot.control.traffic import TrafficGate
from localpilot.engines.registry import EngineRegistry
from localpilot.executor.process import runtime_factory, runtime_names
from localpilot.hardware.profiler import HardwareProfiler
from localpilot.intent.parser import parse_intent
from localpilot.models.registry import ModelRegistry
from localpilot.orchestrator import profile_requirements
from localpilot.planner.policies import PolicyEngine
from localpilot.profiles.store import ProfileStore
from localpilot.schemas import CandidatePlan, MemoryEstimate
from localpilot.sizing import MemoryModel, decode_roofline_tokens_s

WEB_ROOT = Path(__file__).resolve().parent.parent / "web"


def _candidate_from_state(state: Dict[str, Any]) -> CandidatePlan:
    payload = dict(state)
    estimate = payload.get("memory_estimate")
    if isinstance(estimate, dict):
        payload["memory_estimate"] = MemoryEstimate(**estimate)
    return CandidatePlan(**payload)


class RuntimeSession:
    """Keeps one serving runtime alive across chat requests."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.runtime = None
        self.candidate_id = None

    def ensure(self, candidate: CandidatePlan):
        with self.lock:
            if (
                self.runtime is not None
                and self.candidate_id == candidate.candidate_id
            ):
                return self.runtime
            self._stop_locked()
            runtime = runtime_factory(candidate.runtime)()
            runtime.load_model(candidate)
            runtime.start_model()
            health = runtime.health_check()
            if not health.get("healthy"):
                runtime.stop_model()
                raise RuntimeError(
                    f"Runtime failed its health check: {health.get('detail')}"
                )
            self.runtime = runtime
            self.candidate_id = candidate.candidate_id
            return runtime

    def _stop_locked(self) -> None:
        if self.runtime is not None:
            try:
                self.runtime.stop_model()
            finally:
                self.runtime = None
                self.candidate_id = None

    def stop(self) -> None:
        with self.lock:
            self._stop_locked()


def create_app():
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI is not installed. Install the API extra with "
            "pip install -e .[api]"
        ) from exc

    app = FastAPI(
        title="LocalPilot",
        version="0.2.0",
        description="Local AI compute autopilot for NVIDIA DGX Spark",
    )
    store = ProfileStore()
    session = RuntimeSession()
    jobs = JobRegistry()
    engines = EngineRegistry()
    registry = ModelRegistry()
    policies = PolicyEngine()
    memory_model = MemoryModel(policies.memory_config)
    telemetry = RequestTelemetry()
    traffic = TrafficGate()

    # ------------------------------------------------------------------
    # status

    @app.get("/health")
    def health():
        current = store.current()
        return {
            "status": current.get("status", "STOPPED"),
            "ready": current.get("status") == "READY",
            "simulated": current.get("simulated"),
            "runtimes": runtime_names(),
            "traffic": traffic.status(),
        }

    @app.get("/v1/hardware")
    def hardware(simulate: bool = False):
        profile = HardwareProfiler().profile(simulate=simulate)
        return profile.to_dict()

    @app.get("/v1/engines")
    def engine_list():
        probed = engines.probe_all()
        return {
            "object": "list",
            "data": [
                {**spec.to_dict(), "probe": probed.get(spec.engine_id, {})}
                for spec in engines.all()
            ],
        }

    @app.get("/v1/registry")
    def model_registry(task: str = None, simulate: bool = False):
        """The model registry with this machine's sizing applied.

        Returning the memory estimate and the bandwidth ceiling alongside
        each entry is what makes the catalogue actionable: the same
        checkpoint is a good or an impossible choice depending on the pool
        it has to fit in.
        """
        profile = HardwareProfiler().profile(simulate=simulate)
        models = registry.for_task(task) if task else registry.all()
        payload = []
        for model in models:
            estimate = memory_model.estimate(
                model, profile, context_length=8192, concurrency=1
            )
            ceiling = decode_roofline_tokens_s(
                model.active_parameter_count_b,
                model.precision,
                profile.memory_bandwidth_gbps,
                efficiency=policies.bandwidth_efficiency,
                memory_model=memory_model,
            )
            payload.append(
                {
                    **model.to_dict(),
                    "sizing": estimate.to_dict(),
                    "roofline_decode_tokens_s": (
                        round(ceiling, 1) if ceiling else None
                    ),
                    "max_context_at_budget": memory_model.max_context_for(
                        model, profile
                    ),
                    "mixture_of_experts": model.is_mixture_of_experts,
                }
            )
        return {"object": "list", "data": payload, "simulated": profile.simulated}

    @app.get("/v1/policies")
    def policy_view():
        return {
            "priorities": policies.priority_names(),
            "search_space": policies.search_space,
            "roofline": policies.roofline,
            "gates": {
                "quality": policies.quality_gate,
                "stability": policies.stability_gate,
            },
            "max_candidates": policies.max_candidates,
        }

    @app.post("/v1/intent")
    def intent_preview(payload: dict):
        goal = str(payload.get("goal", "")).strip()
        if not goal:
            raise HTTPException(status_code=400, detail="goal is required")
        return parse_intent(goal).to_dict()

    # ------------------------------------------------------------------
    # autopilot

    @app.post("/v1/autopilot")
    def start_autopilot(payload: dict):
        goal = str(payload.get("goal", "")).strip()
        if not goal:
            raise HTTPException(status_code=400, detail="goal is required")
        mode = str(payload.get("mode", "auto"))
        reuse = bool(payload.get("reuse_profile", True))
        job = jobs.submit(goal, mode=mode, reuse_profile=reuse)

        if payload.get("wait"):
            deadline = time.monotonic() + float(payload.get("timeout", 120))
            while time.monotonic() < deadline and job.status in {
                "queued",
                "running",
            }:
                time.sleep(0.2)
        return job.to_dict()

    @app.get("/v1/autopilot/{job_id}")
    def autopilot_status(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        return job.to_dict()

    @app.get("/v1/autopilot")
    def autopilot_history(limit: int = 20):
        return {
            "object": "list",
            "data": [job.to_dict(include_result=False) for job in jobs.recent(limit)],
        }

    # ------------------------------------------------------------------
    # profiles and runs

    @app.get("/v1/profiles")
    def profiles():
        return {"object": "list", "data": store.list_profiles()}

    @app.get("/v1/profiles/current")
    def current_profile():
        return store.current()

    @app.get("/v1/runs")
    def runs(limit: int = 20):
        return {"object": "list", "data": store.list_runs(limit)}

    @app.get("/v1/runs/{run_id}")
    def run_detail(run_id: str):
        data = store.load_run(run_id)
        if data is None:
            raise HTTPException(status_code=404, detail="unknown run")
        return data

    @app.get("/v1/benchmarks")
    def benchmarks():
        config = load_benchmark_config()
        return {
            "warmup_runs": config.get("warmup_runs"),
            "measured_runs": config.get("measured_runs"),
            "max_new_tokens": config.get("max_new_tokens"),
            "prompts": [
                {
                    "id": prompt.get("id"),
                    "task": prompt.get("task"),
                    "expected_terms": prompt.get("expected_terms"),
                }
                for prompt in config.get("prompts", [])
            ],
        }

    # ------------------------------------------------------------------
    # adaptive control

    @app.get("/v1/telemetry")
    def telemetry_snapshot():
        return telemetry.snapshot()

    @app.post("/v1/reconcile")
    def reconcile_runtime(payload: dict):
        current = store.current()
        profile_key = current.get("profile_key")
        saved = store.load(profile_key) if profile_key else None
        thresholds = payload.get("thresholds") or {}
        if not isinstance(thresholds, dict):
            raise HTTPException(status_code=400, detail="thresholds must be an object")
        supplied_metrics = payload.get("metrics") or {}
        if not isinstance(supplied_metrics, dict):
            raise HTTPException(status_code=400, detail="metrics must be an object")
        observation = RuntimeObservation.from_mapping(
            {**telemetry.snapshot(), **supplied_metrics}
        )
        try:
            if saved is None:
                objectives = ServiceObjectives(
                    max_request_p95_ms=thresholds.get("max_request_p95_ms"),
                    max_ttft_p95_ms=thresholds.get("max_ttft_p95_ms"),
                    min_throughput_tokens_s=thresholds.get(
                        "min_throughput_tokens_s"
                    ),
                    max_error_rate=float(thresholds.get("max_error_rate", 0.02)),
                    min_quality=thresholds.get("min_quality"),
                    min_available_memory_gb=thresholds.get(
                        "min_available_memory_gb"
                    ),
                    max_queue_depth=thresholds.get("max_queue_depth"),
                )
                hardware_fingerprint = None
            else:
                objectives = ServiceObjectives.from_profile(
                    saved,
                    max_request_p95_ms=thresholds.get("max_request_p95_ms"),
                    max_ttft_p95_ms=thresholds.get("max_ttft_p95_ms"),
                    min_throughput_tokens_s=thresholds.get(
                        "min_throughput_tokens_s"
                    ),
                    max_error_rate=float(thresholds.get("max_error_rate", 0.02)),
                    min_quality=thresholds.get("min_quality"),
                    min_available_memory_gb=thresholds.get(
                        "min_available_memory_gb"
                    ),
                    max_queue_depth=thresholds.get("max_queue_depth"),
                )
                hardware_fingerprint = HardwareProfiler().profile(
                    simulate=saved.simulated
                ).fingerprint
            desired = (
                profile_requirements(parse_intent(str(payload["goal"])))
                if payload.get("goal")
                else None
            )
            result = Reconciler(store).reconcile(
                observation,
                objectives,
                ReconcilePolicy(
                    consecutive_breaches=int(payload.get("window", 3)),
                    cooldown_seconds=int(payload.get("cooldown_seconds", 900)),
                ),
                hardware_fingerprint=hardware_fingerprint,
                desired_requirements=desired,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return result.to_dict()

    # ------------------------------------------------------------------
    # inference

    @app.get("/v1/models")
    def models():
        current = store.current()
        candidate = current.get("candidate")
        data = []
        if candidate:
            data.append(
                {
                    "id": "localpilot-best",
                    "object": "model",
                    "owned_by": "localpilot",
                    "source_model": candidate["model_id"],
                    "engine": candidate.get("engine"),
                    "simulated": current.get("simulated"),
                }
            )
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    def chat_completions(payload: dict):
        if payload.get("stream"):
            raise HTTPException(
                status_code=400,
                detail="Streaming is not implemented in this MVP",
            )
        current = store.current()
        if current.get("status") != "READY" or not current.get("candidate"):
            raise HTTPException(
                status_code=503,
                detail="Run localpilot autopilot before calling the API",
            )
        messages = payload.get("messages") or []
        if not messages:
            raise HTTPException(status_code=400, detail="messages is required")
        prompt = "\n".join(
            str(message.get("content", ""))
            for message in messages
            if message.get("role") in {"system", "user"}
        )
        candidate = _candidate_from_state(current["candidate"])
        if not traffic.acquire():
            raise HTTPException(
                status_code=503,
                detail="LocalPilot is draining in-flight requests for a safe change",
            )
        request_token = telemetry.begin()
        ok = False
        try:
            runtime = session.ensure(candidate)
            output = runtime.generate(
                prompt, max_new_tokens=int(payload.get("max_tokens", 256))
            )
            ok = True
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"{type(exc).__name__}: {exc}"
            ) from exc
        finally:
            telemetry.complete(request_token, ok=ok)
            traffic.release()
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.get("model", "localpilot-best"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": output},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "localpilot": {
                "simulated": candidate.simulated,
                "engine": candidate.engine,
                "device": candidate.device,
                "precision": candidate.precision,
            },
        }

    @app.post("/internal/stop")
    def stop():
        session.stop()
        store.stop()
        return {"status": "STOPPED"}

    @app.post("/internal/drain")
    def drain(payload: dict = None):
        body = payload or {}
        try:
            timeout = float(body.get("timeout", 30))
            return traffic.drain(timeout)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/internal/resume")
    def resume():
        return traffic.resume()

    @app.post("/internal/activate")
    def activate(payload: dict):
        state = traffic.status()
        if state["accepting"] or state["active_requests"]:
            raise HTTPException(
                status_code=409,
                detail="Drain traffic successfully before activating a profile",
            )
        profile_key = str(payload.get("profile_key", "")).strip()
        if not profile_key:
            raise HTTPException(status_code=400, detail="profile_key is required")
        staged = store.load(profile_key)
        if staged is None:
            raise HTTPException(status_code=404, detail="unknown profile")

        previous_state = store.current()
        previous_key = previous_state.get("profile_key")
        previous_candidate = previous_state.get("candidate")
        candidate = staged.candidate
        started = time.perf_counter()
        try:
            runtime = session.ensure(candidate)
            health = runtime.health_check()
            if not health.get("healthy"):
                raise RuntimeError(
                    f"staged runtime failed health check: {health.get('detail')}"
                )
            output = runtime.generate("Reply with LOCALPILOT_READY", max_new_tokens=16)
            if not str(output).strip():
                raise RuntimeError("staged runtime returned an empty business probe")
            store.activate(profile_key)
        except Exception as exc:
            store.set_current(previous_state)
            if previous_candidate:
                try:
                    session.ensure(_candidate_from_state(previous_candidate))
                except Exception as rollback_exc:
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            f"Activation failed ({type(exc).__name__}: {exc}); "
                            "the previous profile state was restored but its runtime "
                            f"could not be restarted ({type(rollback_exc).__name__}: "
                            f"{rollback_exc})"
                        ),
                    ) from exc
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Activation failed and previous profile was restored: "
                    f"{type(exc).__name__}: {exc}"
                ),
            ) from exc
        return {
            "status": "ACTIVATED",
            "profile_key": profile_key,
            "previous_profile_key": previous_key,
            "candidate_id": candidate.candidate_id,
            "probe_status": "passed",
            "activation_ms": round((time.perf_counter() - started) * 1000, 3),
            "traffic": traffic.status(),
            "next_step": "resume traffic after reviewing this result",
        }

    # ------------------------------------------------------------------
    # dashboard

    if WEB_ROOT.is_dir():
        app.mount(
            "/static", StaticFiles(directory=str(WEB_ROOT)), name="static"
        )

        @app.get("/")
        def dashboard():
            return FileResponse(str(WEB_ROOT / "index.html"))

    return app


try:
    app = create_app()
except RuntimeError:
    app = None
