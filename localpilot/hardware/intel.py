from __future__ import annotations

from localpilot.hardware.base import ComputeProvider
from localpilot.hardware.profiler import HardwareProfiler
from localpilot.schemas import HardwareProfile


class IntelProvider(ComputeProvider):
    def __init__(self) -> None:
        self._profiler = HardwareProfiler()
        self._last_profile = None

    def profile(self, simulate: bool = False) -> HardwareProfile:
        self._last_profile = self._profiler.profile(simulate=simulate)
        return self._last_profile

    def supported_devices(self) -> list:
        profile = self._last_profile or self.profile()
        if "intel" not in profile.cpu.get("model", "").lower():
            return []
        return list(profile.available_devices)

    def capabilities(self) -> dict:
        devices = self.supported_devices()
        return {
            "provider": "intel",
            "openvino": bool(devices),
            "devices": devices,
            "generative_ai": bool(devices),
        }

