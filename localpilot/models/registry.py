from __future__ import annotations

from pathlib import Path
from typing import List

from localpilot.schemas import ModelSpec
from localpilot.utils import config_file, load_data_file


class ModelRegistry:
    def __init__(self, path: Path = None) -> None:
        self.path = path or config_file("models.yaml")

    def all(self) -> List[ModelSpec]:
        data = load_data_file(self.path)
        return [ModelSpec(**item) for item in data.get("models", [])]

    def for_task(self, task: str) -> List[ModelSpec]:
        return [model for model in self.all() if model.serves(task)]

    def tasks(self) -> List[str]:
        names = {task for model in self.all() for task in model.tasks}
        return sorted(names)

    def get(self, model_id: str) -> ModelSpec:
        for model in self.all():
            if model.model_id == model_id:
                return model
        raise KeyError(f"Unknown model: {model_id}")
