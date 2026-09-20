from __future__ import annotations

from typing import Any, Dict, List

from localpilot.hardware.base import ComputeProvider
from localpilot.hardware.profiler import HardwareProfiler
from localpilot.schemas import HardwareProfile


class NvidiaProvider(ComputeProvider):
    """CUDA compute provider, with DGX Spark treated as its own platform.

    Unified memory is the distinction that matters downstream: it changes
    the capacity ceiling from board VRAM to system memory, and it makes a
    model's memory footprint compete with the operating system.
    """

    def __init__(self, profiler: HardwareProfiler = None) -> None:
        self._profiler = profiler or HardwareProfiler()
        self._last_profile: HardwareProfile = None

    def profile(self, simulate: bool = False) -> HardwareProfile:
        self._last_profile = self._profiler.profile(simulate=simulate)
        return self._last_profile

    def _profile(self) -> HardwareProfile:
        return self._last_profile or self.profile()

    def supported_devices(self) -> List[str]:
        profile = self._profile()
        if not profile.accelerator.get("detected"):
            return [device for device in profile.usable_devices if device == "CPU"]
        return list(profile.available_devices)

    def capabilities(self) -> Dict[str, Any]:
        profile = self._profile()
        return {
            "provider": "nvidia",
            "platform": profile.platform_id,
            "cuda": bool(profile.accelerator.get("detected")),
            "devices": self.supported_devices(),
            "unified_memory": profile.unified_memory,
            "memory_bandwidth_gbps": profile.memory_bandwidth_gbps,
            "engines": profile.stack.get("engines_available", []),
            "generative_ai": profile.real_execution_ready,
        }
