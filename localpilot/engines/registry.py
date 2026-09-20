from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from localpilot.utils import load_data_file, project_home


@dataclass
class EngineSpec:
    engine_id: str
    display_name: str
    kind: str
    probe: Dict[str, Any] = field(default_factory=dict)
    openai_base_path: Optional[str] = None
    health_path: Optional[str] = None
    launch: Dict[str, Any] = field(default_factory=dict)
    knobs: List[str] = field(default_factory=list)
    supports: Dict[str, Any] = field(default_factory=dict)
    rebuild_per_configuration: bool = False
    notes: str = ""

    def supports_feature(self, name: str) -> bool:
        return bool(self.supports.get(name))

    def has_knob(self, name: str) -> bool:
        return name in self.knobs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "display_name": self.display_name,
            "kind": self.kind,
            "knobs": list(self.knobs),
            "supports": dict(self.supports),
            "rebuild_per_configuration": self.rebuild_per_configuration,
            "notes": self.notes,
        }


def configured_base_url(engine_id: str) -> Optional[str]:
    """A server the user already runs, per engine or as a global default."""
    for variable in (
        f"LOCALPILOT_{engine_id.upper()}_BASE_URL",
        "LOCALPILOT_ENGINE_BASE_URL",
    ):
        value = os.environ.get(variable)
        if value:
            return value.rstrip("/")
    return None


def served_models(engine_id: str, base_path: str = "/v1") -> List[str]:
    """Which models a configured server actually serves, if any.

    Attaching to a server the user already started means the plan can only
    honestly measure what that server has loaded. Asking it is the only way
    to know.
    """
    base_url = configured_base_url(engine_id)
    if not base_url:
        return []
    request = urllib.request.Request(
        f"{base_url}{(base_path or '/v1').rstrip('/')}/models", method="GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=5.0) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
    except Exception:
        return []
    return [
        entry.get("id")
        for entry in (payload or {}).get("data", [])
        if entry.get("id")
    ]


def _endpoint_answers(base_url: str, health_path: str, timeout: float = 1.5) -> bool:
    request = urllib.request.Request(
        f"{base_url}{health_path or '/health'}", method="GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 500
    except urllib.error.HTTPError as exc:
        # A 4xx still proves something is listening and speaking HTTP.
        return exc.code < 500
    except Exception:
        return False


def _local_images(binary_path: Optional[str], patterns: List[str]) -> List[str]:
    if not binary_path or not patterns:
        return []
    try:
        result = subprocess.run(
            [binary_path, "image", "ls", "--format", "{{.Repository}}:{{.Tag}}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    found = []
    for line in result.stdout.splitlines():
        name = line.strip()
        if name and any(pattern in name for pattern in patterns):
            found.append(name)
    return found


def _container_runtime() -> Optional[str]:
    for candidate in ("docker", "podman"):
        path = shutil.which(candidate)
        if path:
            return path
    return None


def probe_engine(spec: EngineSpec) -> Dict[str, Any]:
    """Decides whether this engine can actually serve on this machine.

    "Is it installed on the host" is the wrong question. On DGX Spark the
    supported way to run vLLM is a container, so a host-only probe reports
    every engine missing on a machine that is ready to serve. Three signals
    are checked, strongest first:

    1. **reachable** -- a configured base URL answers its health path. This
       is proof rather than inference: something is serving right now.
    2. **container** -- a matching image is present locally, so it can be
       started without a pull.
    3. **host** -- the module or binary is installed directly.

    Nothing here imports the engine or starts anything: importing vllm or
    tensorrt_llm initializes CUDA and costs seconds, and this runs before
    every plan.
    """
    module = spec.probe.get("python_module")
    binary = spec.probe.get("binary")

    found_module = None
    if module:
        try:
            found_module = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found_module = False
    binary_path = shutil.which(binary) if binary else None

    report: Dict[str, Any] = {
        "engine_id": spec.engine_id,
        "kind": spec.kind,
        "python_module": found_module,
        "binary_path": binary_path,
        "available": False,
        "source": None,
    }

    base_url = configured_base_url(spec.engine_id)
    if base_url:
        report["base_url"] = base_url
        if _endpoint_answers(base_url, spec.health_path):
            report["available"] = True
            report["source"] = "reachable"
            report["detail"] = f"answering at {base_url}"
            return report
        report["detail"] = (
            f"{base_url} is configured but not answering "
            f"{spec.health_path or '/health'}"
        )

    patterns = spec.probe.get("docker_image_contains") or []
    runtime = _container_runtime()
    if patterns:
        images = _local_images(runtime, patterns)
        report["local_images"] = images
        if images:
            report["available"] = True
            report["source"] = "container"
            report.setdefault("detail", f"image present: {images[0]}")
            return report
        if runtime and not report.get("detail"):
            report["detail"] = (
                f"{Path(runtime).name} is installed but no image matching "
                f"{patterns} is present locally"
            )

    # The host fallback must not apply to a container engine: its `binary`
    # is the container runtime, not the engine. Counting `docker` on PATH as
    # "NIM is available" puts a candidate in the plan that cannot start
    # without a multi-gigabyte pull.
    if spec.kind != "container" and (found_module or binary_path):
        report["available"] = True
        report["source"] = "host"
        report.setdefault("detail", binary_path or f"python module {module}")
    return report


class EngineRegistry:
    def __init__(self, path: Path = None) -> None:
        self.path = path or (project_home() / "config" / "engines.yaml")
        self._cache: Optional[List[EngineSpec]] = None

    def all(self) -> List[EngineSpec]:
        if self._cache is None:
            data = load_data_file(self.path)
            self._cache = [
                EngineSpec(**item) for item in data.get("engines", [])
            ]
        return list(self._cache)

    def get(self, engine_id: str) -> EngineSpec:
        for spec in self.all():
            if spec.engine_id == engine_id:
                return spec
        raise KeyError(f"Unknown engine: {engine_id}")

    def ids(self) -> List[str]:
        return [spec.engine_id for spec in self.all()]

    def probe_all(self) -> Dict[str, Dict[str, Any]]:
        return {spec.engine_id: probe_engine(spec) for spec in self.all()}

    def available_ids(self) -> List[str]:
        return [
            engine_id
            for engine_id, report in self.probe_all().items()
            if report["available"]
        ]

    def servable(self, engine_id: str) -> bool:
        """Whether the engine can be benchmarked over HTTP."""
        try:
            spec = self.get(engine_id)
        except KeyError:
            return False
        return bool(spec.openai_base_path) and spec.kind in {"server", "container"}
