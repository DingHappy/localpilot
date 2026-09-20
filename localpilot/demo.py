from __future__ import annotations

from typing import Any, Dict

from localpilot.orchestrator import Orchestrator


DEFAULT_DEMO_GOAL = (
    "帮我部署一个完全本地运行的代码审查 AI。"
    "我的代码不能离开这台电脑。"
    "质量要够用，但响应速度优先。"
)

# The decisions a DGX Spark user actually faces before the first token.
# Each one has a defensible wrong answer, which is the point.
MANUAL_DECISIONS = [
    "Pick an engine: vLLM, TensorRT-LLM, SGLang, a NIM container or llama.cpp",
    "Pick a checkpoint, and decide whether NVFP4, FP8 or BF16 of it",
    "Work out whether the weights plus KV cache fit 128 GB shared with the OS",
    "Choose a context length, knowing KV cost scales with it",
    "Choose batch width, trading per-stream latency for aggregate throughput",
    "Decide whether to run speculative decoding, and find a draft checkpoint",
    "Decide whether to quantize the KV cache",
    "Build comparable benchmarks so the numbers mean the same thing",
    "Compare, choose, and write the winning configuration down somewhere",
]


def run_demo(
    mode: str = "mock",
    goal: str = DEFAULT_DEMO_GOAL,
    orchestrator: Orchestrator = None,
) -> Dict[str, Any]:
    engine = orchestrator or Orchestrator()
    result = engine.autopilot(goal, mode=mode, reuse_profile=False)
    payload = result.to_dict()

    return {
        "demo_a": {
            "name": "Manual configuration",
            "executed": False,
            "user_decisions": MANUAL_DECISIONS,
        },
        "demo_b": {
            "name": "LocalPilot autopilot",
            "goal": goal,
            "result": payload,
        },
        "findings": _findings(payload),
        "disclaimer": (
            "Simulated mode demonstrates orchestration and the search space. "
            "Its numbers are modelled from bandwidth and active parameters "
            "per token; they are not a hardware benchmark."
            if result.hardware.simulated
            else None
        ),
    }


def _dimensions(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "precision": candidate["precision"],
        "engine": candidate["engine"],
        "speculative decoding": bool(
            (candidate.get("runtime_config") or {}).get("speculative")
        ),
        "active parameters": candidate.get("active_parameter_count_b"),
        "total parameters": candidate.get("parameter_count_b"),
        "batch width": candidate.get("concurrency", 1),
        "KV precision": candidate.get("kv_cache_dtype", "auto"),
        "context": candidate.get("context_length"),
    }


def _describe_difference(left: Dict[str, Any], right: Dict[str, Any]) -> str:
    """Names every dimension two candidates differ on.

    A speed ratio between two configurations is only interpretable if the
    differences are stated. Comparing an NVFP4 speculative run on one
    engine against a BF16 run on another and calling the gap a precision
    effect attributes the whole difference to one of three causes, which
    the run does not support.
    """
    left_dimensions, right_dimensions = _dimensions(left), _dimensions(right)
    parts = []
    for key, value in left_dimensions.items():
        other = right_dimensions[key]
        if value == other:
            continue
        if isinstance(value, bool):
            parts.append(f"{key} {'on' if value else 'off'} vs "
                         f"{'on' if other else 'off'}")
        elif isinstance(value, (int, float)) and isinstance(other, (int, float)):
            parts.append(f"{key} {value:g} vs {other:g}")
        else:
            parts.append(f"{key} {value} vs {other}")
    return ", ".join(parts)


def _findings(payload: Dict[str, Any]) -> list:
    """Contrasts this run produced, each stated with its confounders.

    A winner alone is not evidence. What makes the result persuasive is the
    pair it beat, by how much, and on which axes the two differed -- so
    every comparison here carries the full difference list rather than
    crediting the gap to whichever cause is most flattering.
    """
    measured = [
        item for item in payload.get("candidates", []) if item.get("benchmark")
    ]
    findings = []

    def rate(item):
        return item["benchmark"]["throughput_tokens_s"]

    pairs = []
    for index, left in enumerate(measured):
        for right in measured[index + 1:]:
            faster, slower = (left, right) if rate(left) >= rate(right) else (right, left)
            difference = _describe_difference(faster["candidate"], slower["candidate"])
            axes = difference.count(",") + 1 if difference else 0
            if not axes:
                continue
            ratio = rate(faster) / max(rate(slower), 1e-6)
            pairs.append((axes, -ratio, faster, slower, difference))

    # Prefer the cleanest comparisons: fewest confounders, then largest gap.
    pairs.sort(key=lambda entry: (entry[0], entry[1]))
    for axes, negative_ratio, faster, slower, difference in pairs[:2]:
        memory_delta = (
            slower["benchmark"]["peak_memory_gb"]
            - faster["benchmark"]["peak_memory_gb"]
        )
        sentence = (
            f"{rate(faster):.0f} vs {rate(slower):.0f} tok/s "
            f"({-negative_ratio:.1f}x) with {difference}"
        )
        if memory_delta > 1:
            sentence += f", and {memory_delta:.0f} GB less memory"
        findings.append(sentence)

    sparse = [
        item
        for item in measured
        if item["candidate"].get("active_parameter_count_b")
        and item["candidate"]["active_parameter_count_b"]
        < item["candidate"]["parameter_count_b"]
    ]
    if len(sparse) >= 2:
        heaviest = max(sparse, key=lambda item: item["candidate"]["parameter_count_b"])
        lightest = min(sparse, key=lambda item: item["candidate"]["parameter_count_b"])
        if heaviest is not lightest:
            findings.append(
                f"Capacity and speed came apart: "
                f"{heaviest['candidate']['parameter_count_b']:g}B total needed "
                f"{heaviest['benchmark']['peak_memory_gb']:.0f} GB against "
                f"{lightest['benchmark']['peak_memory_gb']:.0f} GB, while decode "
                f"tracked active parameters "
                f"({heaviest['candidate']['active_parameter_count_b']:g}B vs "
                f"{lightest['candidate']['active_parameter_count_b']:g}B)"
            )

    rejections = [
        step
        for step in payload.get("agent_trace", [])
        if step.get("action") == "reject_model"
    ]
    for step in rejections[:1]:
        findings.append(
            f"{step['data'].get('model_id')} was rejected before anything ran: "
            f"{step['data'].get('reason')}"
        )

    return findings


def print_demo(demo: Dict[str, Any]) -> None:
    manual = demo["demo_a"]
    result = demo["demo_b"]["result"]
    hardware = result["hardware"]

    print("=" * 72)
    print("DEMO A - Manual configuration")
    print("=" * 72)
    print()
    print(f"{len(manual['user_decisions'])} decisions before the first token,")
    print("each with a plausible wrong answer:")
    print()
    for index, decision in enumerate(manual["user_decisions"], start=1):
        print(f"  {index}. {decision}")
    print()

    print("=" * 72)
    print("DEMO B - LocalPilot autopilot")
    print("=" * 72)
    print()
    print(f"Goal: {demo['demo_b']['goal']}")
    print()
    intent = result["intent"]
    print(f"Understood as: {intent['task']} / {intent['priority']} / "
          f"{intent['context_length']} tokens / x{intent['concurrency']} / "
          f"{intent['privacy']}")
    label = "SIMULATED " if hardware["simulated"] else ""
    print(f"Hardware:      {label}{hardware['platform_id']}, "
          f"{hardware['memory'].get('total_gb')} GB "
          f"{'unified' if hardware['unified_memory'] else 'host'}, "
          f"{hardware.get('memory_bandwidth_gbps') or '?'} GB/s")
    print()

    print("Candidates measured against each other")
    print()
    header = (
        f"  {'configuration':46s} {'ttft':>8s} {'tok/s':>8s} "
        f"{'agg':>8s} {'mem':>7s} {'score':>6s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    winner_id = result["best_profile"]["candidate"]["candidate_id"]
    for item in result["candidates"]:
        candidate = item["candidate"]
        benchmark = item.get("benchmark")
        mark = "*" if candidate["candidate_id"] == winner_id else " "
        if not benchmark:
            print(f" {mark}{candidate['candidate_id'][:46]:46s} "
                  f"{'failed: ' + str(item.get('error'))[:40]}")
            continue
        score = item.get("score")
        aggregate = benchmark.get("aggregate_throughput_tokens_s")
        print(
            f" {mark}{candidate['candidate_id'][:46]:46s} "
            f"{benchmark['ttft_ms']:>7.0f}m "
            f"{benchmark['throughput_tokens_s']:>8.1f} "
            f"{(f'{aggregate:.0f}' if aggregate else '-'):>8s} "
            f"{benchmark['peak_memory_gb']:>6.1f}G "
            f"{(f'{score:.1f}' if score is not None else 'gated'):>6s}"
        )
    print()

    if demo.get("findings"):
        print("What the run showed")
        print()
        for finding in demo["findings"]:
            print(f"  - {finding}")
        print()

    best = result["best_profile"]
    candidate = best["candidate"]
    print("Chosen configuration")
    print()
    print(f"  Model       {candidate['model_id']}")
    print(f"  Source      {candidate['source_id']}")
    print(f"  Engine      {candidate['engine']}")
    print(f"  Precision   {candidate['precision']}")
    print(f"  Context     {candidate['context_length']}")
    speculative = (candidate.get("runtime_config") or {}).get("speculative")
    if speculative:
        print(f"  Speculative {speculative.get('draft_source_id')}")
    print(f"  Score       {best['score']:.2f}/100")
    print(f"  Profile     {best['profile_key']} (reused on the next identical ask)")
    print()

    if demo.get("disclaimer"):
        print(f"WARNING: {demo['disclaimer']}")
