from __future__ import annotations

from typing import Dict, List

from localpilot.planner.policies import PolicyEngine
from localpilot.schemas import CandidateResult


def _higher_is_better(values: List[float], value: float) -> float:
    low, high = min(values), max(values)
    if high == low:
        return 1.0
    return (value - low) / (high - low)


def _lower_is_better(values: List[float], value: float) -> float:
    return 1.0 - _higher_is_better(values, value)


def score_results(
    results: List[CandidateResult],
    priority: str,
    policies: PolicyEngine,
) -> List[CandidateResult]:
    successful = [item for item in results if item.benchmark is not None]
    if not successful:
        return results

    quality_gate = float(policies.data.get("quality_gate", 0.55))
    stability_gate = float(policies.data.get("stability_gate", 0.90))
    eligible = [
        item
        for item in successful
        if item.benchmark.quality >= quality_gate
        and item.benchmark.stability >= stability_gate
    ]
    if not eligible:
        return results

    weights: Dict[str, float] = policies.priority(priority)["weights"]
    ttfts = [item.benchmark.ttft_ms for item in eligible]
    throughputs = [item.benchmark.throughput_tokens_s for item in eligible]
    memories = [item.benchmark.peak_memory_gb for item in eligible]
    qualities = [item.benchmark.quality for item in eligible]
    stabilities = [item.benchmark.stability for item in eligible]

    for item in eligible:
        metrics = item.benchmark
        components = {
            "quality": _higher_is_better(qualities, metrics.quality),
            "latency": _lower_is_better(ttfts, metrics.ttft_ms),
            "throughput": _higher_is_better(
                throughputs, metrics.throughput_tokens_s
            ),
            "memory": _lower_is_better(memories, metrics.peak_memory_gb),
            "stability": _higher_is_better(stabilities, metrics.stability),
        }
        item.score = round(
            sum(weights[name] * components[name] for name in weights) * 100,
            2,
        )
    return results

