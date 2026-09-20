from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from localpilot.utils import load_data_file, project_home


FALLBACK_TASK = "chat"


def load_benchmark_config(path: Path = None) -> Dict[str, Any]:
    target = path or (project_home() / "config" / "benchmark.yaml")
    return load_data_file(target)


def prompts_for_task(task: str, path: Path = None) -> List[Dict[str, Any]]:
    """Prompts for a task, falling back to the generic chat set.

    A task with no prompts of its own should still be measurable, otherwise
    adding a model for a new task silently breaks the benchmark path.
    """
    config = load_benchmark_config(path)
    everything = config.get("prompts", [])
    matching = [item for item in everything if item.get("task") == task]
    if matching:
        return matching
    return [item for item in everything if item.get("task") == FALLBACK_TASK]


def tasks_with_prompts(path: Path = None) -> List[str]:
    config = load_benchmark_config(path)
    return sorted(
        {item["task"] for item in config.get("prompts", []) if item.get("task")}
    )
