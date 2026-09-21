"""Small, explicit acceptance policies for an existing benchmark dataset."""
import math


def validate_acceptance(value):
    if not isinstance(value, dict):
        raise ValueError("acceptance must be an object")
    allowed = {"objective", "min_quality", "max_total_latency_ms"}
    if set(value) - allowed:
        raise ValueError("unknown acceptance fields: " + ", ".join(sorted(set(value) - allowed)))
    objective = value.get("objective", "weighted")
    if objective not in {"weighted", "fastest_complete", "highest_quality"}:
        raise ValueError("unsupported acceptance objective")
    for key in ("min_quality", "max_total_latency_ms"):
        if key not in value:
            continue
        number = value[key]
        if type(number) not in (int, float) or not math.isfinite(number):
            raise ValueError(f"{key} must be a finite number")
        if key == "min_quality" and not 0 <= number <= 1:
            raise ValueError("min_quality must be between 0 and 1")
        if key == "max_total_latency_ms" and number <= 0:
            raise ValueError("max_total_latency_ms must be positive")
    if objective == "fastest_complete" and "min_quality" not in value:
        raise ValueError("fastest_complete requires min_quality")
    if objective == "highest_quality" and "max_total_latency_ms" not in value:
        raise ValueError("highest_quality requires max_total_latency_ms")
    return dict(value)
