from __future__ import annotations

from pathlib import Path
import os
from typing import Any, Dict, List

from localpilot.utils import config_file, load_data_file
from localpilot.benchmark.acceptance import validate_acceptance


FALLBACK_TASK = "chat"


def load_benchmark_config(path: Path = None) -> Dict[str, Any]:
    override = os.environ.get("LOCALPILOT_BENCHMARK_CONFIG")
    target = path or (Path(override).expanduser() if override else config_file("benchmark.yaml"))
    config = load_data_file(target)
    validate_acceptance(config.get("acceptance", {}))
    return config


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


def prompt_text(prompt: Any) -> str:
    if isinstance(prompt, dict):
        return str(prompt.get("text", ""))
    return str(prompt)


def prompt_has_image(prompt: Any) -> bool:
    return isinstance(prompt, dict) and bool(prompt.get("image_url"))


def keyword_hit(answer: str, prompt: Dict[str, Any]) -> bool:
    terms = [str(term).lower() for term in prompt.get("expected_terms", [])]
    lowered = answer.lower()
    if not terms:
        return False
    hits = [term in lowered for term in terms]
    return all(hits) if prompt.get("expected_terms_match") == "all" else any(hits)
