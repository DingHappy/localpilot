from __future__ import annotations

from abc import ABC, abstractmethod

from localpilot.schemas import HardwareProfile


class ComputeProvider(ABC):
    @abstractmethod
    def profile(self, simulate: bool = False) -> HardwareProfile:
        raise NotImplementedError

    @abstractmethod
    def supported_devices(self) -> list:
        raise NotImplementedError

    @abstractmethod
    def capabilities(self) -> dict:
        raise NotImplementedError

