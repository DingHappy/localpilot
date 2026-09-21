from __future__ import annotations

from typing import Callable, Dict

from localpilot.runtime.base import RuntimeProvider
from localpilot.runtime.mock import MockRuntime
from localpilot.runtime.openai_compat import (
    LlamaCppRuntime,
    NIMRuntime,
    OllamaRuntime,
    SGLangRuntime,
    TRTLLMRuntime,
    VLLMRuntime,
)

RUNTIMES: Dict[str, Callable[[], RuntimeProvider]] = {
    "mock": MockRuntime,
    "vllm": VLLMRuntime,
    "trtllm": TRTLLMRuntime,
    "sglang": SGLangRuntime,
    "nim": NIMRuntime,
    "llamacpp": LlamaCppRuntime,
    "ollama": OllamaRuntime,
}


def runtime_factory(name: str) -> Callable[[], RuntimeProvider]:
    try:
        return RUNTIMES[name]
    except KeyError:
        raise ValueError(
            f"Unsupported runtime: {name}. Known runtimes: "
            f"{', '.join(sorted(RUNTIMES))}"
        ) from None


def runtime_names() -> list:
    return sorted(RUNTIMES)
