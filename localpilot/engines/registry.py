from __future__ import annotations

import importlib.util
import subprocess
import shutil
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


def probe_engine(spec: EngineSpec) -> Dict[str, Any]:
    """Look for an engine without importing or running it.

    Importing vllm or tensorrt_llm initializes CUDA and costs seconds, so a
    check that has to run before every plan uses spec lookup and PATH only.
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

    available = bool(found_module) or bool(binary_path)
    image_prefix = spec.probe.get("docker_image_prefix")
    images = None
    if spec.kind == "container" and image_prefix:
        # A container runtime on PATH says nothing about whether the image
        # is here. Reporting the engine as available on that basis would
        # put a candidate into the plan that cannot start without a pull.
        images = _local_images(binary_path, image_prefix)
        available = bool(binary_path) and bool(images)

    report = {
        "engine_id": spec.engine_id,
        "available": available,
        "python_module": found_module,
        "binary_path": binary_path,
        "kind": spec.kind,
    }
    if images is not None:
        report["local_images"] = images
        if binary_path and not images:
            report["detail"] = (
                f"{binary} is installed but no {image_prefix}* image is "
                "present locally"
            )
    return report


def _local_images(binary_path: Optional[str], prefix: str) -> List[str]:
    if not binary_path:
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
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip().startswith(prefix)
    ]


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
