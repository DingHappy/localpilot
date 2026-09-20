from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from localpilot.utils import load_data_file, project_home


def load_benchmark_config(path: Path = None) -> Dict[str, Any]:
    target = path or (project_home() / "config" / "benchmark.yaml")
    return load_data_file(target)


def prompts_for_task(task: str, path: Path = None) -> List[Dict[str, Any]]:
    config = load_benchmark_config(path)
    return [
        item for item in config.get("prompts", []) if item.get("task") == task
    ]

