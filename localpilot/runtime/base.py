from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from localpilot.schemas import BenchmarkMetrics, CandidatePlan


class RuntimeUnavailable(RuntimeError):
    pass


class RuntimeProvider(ABC):
    @abstractmethod
    def load_model(self, candidate: CandidatePlan) -> None:
        raise NotImplementedError

    @abstractmethod
    def start_model(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop_model(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def generate(self, prompt: Any, max_new_tokens: int = 64) -> str:
        raise NotImplementedError

    @abstractmethod
    def benchmark(
        self,
        prompts: List[Dict[str, Any]],
        warmup_runs: int,
        measured_runs: int,
        max_new_tokens: int,
    ) -> BenchmarkMetrics:
        raise NotImplementedError
