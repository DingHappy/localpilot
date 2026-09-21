from __future__ import annotations

from typing import Dict, List
import math
from localpilot.benchmark.acceptance import validate_acceptance

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
    acceptance=None,
) -> List[CandidateResult]:
    """Scores measured candidates against each other.

    Every component is normalized across the candidates that actually ran,
    so a score is a ranking within this run on this machine. It is not a
    portable benchmark number and is never presented as one.
    """
    acceptance = validate_acceptance({} if acceptance is None else acceptance)
    for item in results:
        item.score = None
        item.score_components = {}
        item.gate_failures = []
    measured = [item for item in results if item.benchmark is not None and item.status == "success"]
    if not measured:
        return results

    quality_gate = acceptance.get("min_quality", policies.quality_gate)
    stability_gate = policies.stability_gate

    eligible = []
    for item in measured:
        failures = []
        if not math.isfinite(item.benchmark.quality) or item.benchmark.quality < quality_gate:
            failures.append(
                f"quality {item.benchmark.quality:.2f} below gate {quality_gate:.2f}"
            )
        if not math.isfinite(item.benchmark.stability) or item.benchmark.stability < stability_gate:
            failures.append(
                f"stability {item.benchmark.stability:.2f} below gate "
                f"{stability_gate:.2f}"
            )
        needs_total = acceptance.get("objective") in {"fastest_complete", "highest_quality"} or "max_total_latency_ms" in acceptance
        total = item.benchmark.total_latency_ms
        if needs_total and (not isinstance(total, (int, float)) or not math.isfinite(total) or total <= 0):
            failures.append("complete response latency is missing or invalid")
        elif "max_total_latency_ms" in acceptance and total > acceptance["max_total_latency_ms"]:
            failures.append(f"complete response {total:.2f} ms exceeds gate {acceptance['max_total_latency_ms']:.2f} ms")
        item.gate_failures = failures
        if not failures:
            eligible.append(item)

    if not eligible:
        return results

    objective = acceptance.get("objective", "weighted")
    if objective in {"fastest_complete", "highest_quality"}:
        fastest = min(item.benchmark.total_latency_ms for item in eligible)
        for item in eligible:
            value = fastest / item.benchmark.total_latency_ms if objective == "fastest_complete" else item.benchmark.quality
            item.score_components = {objective: value}
            item.score = 100 * value
        return results

    weights: Dict[str, float] = policies.priority(priority)["weights"]
    prefer_aggregate = priority == "throughput"

    def throughput_of(item: CandidateResult) -> float:
        metrics = item.benchmark
        if prefer_aggregate and metrics.aggregate_throughput_tokens_s:
            return metrics.aggregate_throughput_tokens_s
        return metrics.throughput_tokens_s

    ttfts = [item.benchmark.ttft_ms for item in eligible]
    throughputs = [throughput_of(item) for item in eligible]
    memories = [item.benchmark.peak_memory_gb for item in eligible]
    qualities = [item.benchmark.quality for item in eligible]
    stabilities = [item.benchmark.stability for item in eligible]

    for item in eligible:
        metrics = item.benchmark
        components = {
            "quality": _higher_is_better(qualities, metrics.quality),
            "latency": _lower_is_better(ttfts, metrics.ttft_ms),
            "throughput": _higher_is_better(throughputs, throughput_of(item)),
            "memory": _lower_is_better(memories, metrics.peak_memory_gb),
            "stability": _higher_is_better(stabilities, metrics.stability),
        }
        item.score_components = {
            name: round(value, 4) for name, value in components.items()
        }
        item.score = round(
            sum(weights[name] * components[name] for name in weights) * 100, 2
        )
    return results
