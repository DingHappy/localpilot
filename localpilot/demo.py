from __future__ import annotations

from typing import Any, Dict

from localpilot.orchestrator import Orchestrator


DEFAULT_DEMO_GOAL = (
    "帮我部署一个完全本地运行的代码审查 AI。"
    "我的代码不能离开这台电脑。"
    "质量要够用，但响应速度优先。"
)


def run_demo(
    mode: str = "mock",
    goal: str = DEFAULT_DEMO_GOAL,
    orchestrator: Orchestrator = None,
) -> Dict[str, Any]:
    engine = orchestrator or Orchestrator()
    result = engine.autopilot(
        goal,
        mode=mode,
        reuse_profile=False,
    )
    return {
        "demo_a": {
            "name": "Manual configuration",
            "executed": False,
            "user_decisions": [
                "Inspect CPU, GPU, and NPU",
                "Choose a model artifact",
                "Check device and quantization compatibility",
                "Choose device and precision",
                "Set context and runtime parameters",
                "Run comparable benchmarks",
                "Compare results and remember the winner",
            ],
        },
        "demo_b": {
            "name": "LocalPilot autopilot",
            "goal": goal,
            "result": result.to_dict(),
        },
        "disclaimer": (
            "Mock mode demonstrates orchestration only. It is not a hardware benchmark."
            if result.hardware.simulated
            else None
        ),
    }


def print_demo(demo: Dict[str, Any]) -> None:
    manual = demo["demo_a"]
    result = demo["demo_b"]["result"]
    print("DEMO A — Manual configuration")
    print()
    for index, decision in enumerate(manual["user_decisions"], start=1):
        print(f"{index}. {decision}")
    print()
    print("DEMO B — LocalPilot autopilot")
    print()
    print(f"Goal: {demo['demo_b']['goal']}")
    print(
        "Hardware: "
        + (
            "SIMULATED Intel CPU + GPU + NPU"
            if result["hardware"]["simulated"]
            else ", ".join(result["hardware"]["available_devices"])
        )
    )
    print()
    print("Candidates")
    for item in result["candidates"]:
        candidate = item["candidate"]
        benchmark = item.get("benchmark")
        score = item.get("score")
        fallback = (
            f" fallback={candidate['recovery_action']}"
            if candidate.get("recovery_action")
            else ""
        )
        if benchmark:
            score_text = (
                f"{score:.2f}" if score is not None else "not_eligible"
            )
            summary = (
                f"TTFT={benchmark['ttft_ms']:.2f}ms "
                f"TPS={benchmark['throughput_tokens_s']:.2f} "
                f"score={score_text}"
            )
        else:
            summary = f"failed={item.get('error')}"
        print(
            f"- {candidate['model_id']} / {candidate['device']} / "
            f"{candidate['precision']}: {summary}{fallback}"
        )
    best = result["best_profile"]
    print()
    print("Best configuration")
    print(f"Model:     {best['candidate']['model_id']}")
    print(f"Device:    {best['candidate']['device']}")
    print(f"Precision: {best['candidate']['precision']}")
    print(f"Context:   {best['candidate']['context_length']}")
    print(f"Score:     {best['score']:.2f}/100")
    print("Status:    READY")
    if demo.get("disclaimer"):
        print()
        print(f"WARNING: {demo['disclaimer']}")
