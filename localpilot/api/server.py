from __future__ import annotations

import threading
import time
import uuid

from localpilot.executor.process import runtime_factory
from localpilot.profiles.store import ProfileStore
from localpilot.schemas import CandidatePlan


class RuntimeSession:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.runtime = None
        self.candidate_id = None

    def ensure(self, candidate: CandidatePlan):
        with self.lock:
            if self.runtime is not None and self.candidate_id == candidate.candidate_id:
                return self.runtime
            self.stop()
            runtime = runtime_factory(candidate.runtime)()
            runtime.load_model(candidate)
            runtime.start_model()
            if not runtime.health_check().get("healthy"):
                runtime.stop_model()
                raise RuntimeError("Runtime failed its health check")
            self.runtime = runtime
            self.candidate_id = candidate.candidate_id
            return runtime

    def stop(self) -> None:
        if self.runtime is not None:
            try:
                self.runtime.stop_model()
            finally:
                self.runtime = None
                self.candidate_id = None


def create_app():
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI is not installed. Install the API extra with pip install -e .[api]"
        ) from exc

    app = FastAPI(title="LocalPilot", version="0.1.0")
    store = ProfileStore()
    session = RuntimeSession()

    @app.get("/health")
    def health():
        current = store.current()
        return {
            "status": current.get("status", "STOPPED"),
            "ready": current.get("status") == "READY",
            "simulated": current.get("simulated"),
        }

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
        candidate = CandidatePlan(**current["candidate"])
        try:
            runtime = session.ensure(candidate)
            output = runtime.generate(
                prompt,
                max_new_tokens=int(payload.get("max_tokens", 256)),
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
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
                "device": candidate.device,
                "precision": candidate.precision,
            },
        }

    @app.post("/internal/stop")
    def stop():
        session.stop()
        store.stop()
        return {"status": "STOPPED"}

    return app


try:
    app = create_app()
except RuntimeError:
    app = None

