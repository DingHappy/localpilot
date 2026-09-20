from __future__ import annotations

from typing import Callable

from localpilot.runtime.base import RuntimeProvider
from localpilot.runtime.mock import MockRuntime
from localpilot.runtime.openvino import OpenVINORuntime


def runtime_factory(name: str) -> Callable[[], RuntimeProvider]:
    if name == "mock":
        return MockRuntime
    if name == "openvino":
        return OpenVINORuntime
    raise ValueError(f"Unsupported runtime: {name}")

