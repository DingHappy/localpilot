from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict

from localpilot.benchmark.runner import BenchmarkRunner
from localpilot.demo import DEFAULT_DEMO_GOAL, print_demo, run_demo
from localpilot.executor.process import runtime_factory
from localpilot.hardware.profiler import HardwareProfiler
from localpilot.intent.parser import parse_intent
from localpilot.models.registry import ModelRegistry
from localpilot.orchestrator import Orchestrator
from localpilot.planner.planner import Planner
from localpilot.profiles.store import ProfileStore
from localpilot.schemas import CandidatePlan


def _json_default(value: Any):
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _print_json(value: Any) -> None:
    print(
        json.dumps(
            value,
            default=_json_default,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def _print_status(state: Dict[str, Any]) -> None:
    if state.get("status") != "READY":
        print("LocalPilot\n\nStatus         STOPPED")
        return
    candidate = state["candidate"]
    benchmark = state["benchmark"]
    prefix = "SIMULATED " if state.get("simulated") else ""
    print("LocalPilot")
    print()
    print(f"Model          {candidate['model_id']}")
    print(f"Runtime        {candidate['runtime']}")
    print(f"Device         {prefix}{candidate['device']}")
    print(f"Precision      {candidate['precision']}")
    print(f"Context        {candidate['context_length']}")
    print()
    print(f"TTFT           {benchmark['ttft_ms']:.2f} ms")
    print(f"Throughput     {benchmark['throughput_tokens_s']:.2f} tok/s")
    print(f"Memory         {benchmark['peak_memory_gb']:.2f} GB")
    print()
    print("Optimization   Completed")
    print(f"Score          {state['score']:.2f}/100")
    print(f"Status         {state['status']}")
    if state.get("simulated"):
        print()
        print("WARNING        Development simulation, not a hardware benchmark")


def _task_prompt(task: str, priority: str) -> str:
    phrases = {
        "coding": "Deploy a completely local code review AI.",
        "chat": "Deploy a completely local chat AI.",
        "embedding": "Deploy a completely local embedding model.",
    }
    suffixes = {
        "latency": " Response speed is the priority.",
        "quality": " Quality is the priority.",
        "low_memory": " Low memory usage is the priority.",
        "balanced": " Balance speed and quality.",
    }
    return phrases.get(task, phrases["chat"]) + suffixes.get(
        priority, suffixes["balanced"]
    )


def command_doctor(args) -> int:
    profile = HardwareProfiler().profile(simulate=False)
    if args.json:
        _print_json(profile)
    else:
        print("LocalPilot doctor")
        print(f"OS: {profile.os['name']} {profile.os['release']}")
        print(f"CPU: {profile.cpu['model']}")
        print(f"OpenVINO installed: {profile.openvino['installed']}")
        devices = ", ".join(profile.available_devices) or "none"
        print(f"OpenVINO devices: {devices}")
        print(f"Real Intel execution ready: {profile.real_execution_ready}")
        for note in profile.notes:
            print(f"- {note}")
    return 0


def command_profile(args) -> int:
    profile = HardwareProfiler().profile(simulate=args.simulate)
    _print_json(profile)
    return 0


def command_recommend(args) -> int:
    orchestrator = Orchestrator()
    resolved = orchestrator.resolve_mode(args.mode)
    profile = HardwareProfiler().profile(simulate=resolved == "mock")
    intent = parse_intent(_task_prompt(args.task, args.priority))
    candidates = Planner().plan(
        intent,
        profile,
        ModelRegistry().for_task(args.task),
        resolved,
    )
    _print_json(
        {
            "intent": intent,
            "hardware": profile,
            "candidates": candidates,
            "warning": "SIMULATED candidates" if profile.simulated else None,
        }
    )
    return 0


def command_autopilot(args) -> int:
    result = Orchestrator().autopilot(
        args.goal,
        mode=args.mode,
        reuse_profile=not args.no_reuse,
    )
    if args.json:
        _print_json(result)
    else:
        if result.profile_reused:
            print("PROFILE HIT: reused a verified matching configuration")
        _print_status(ProfileStore().current())
    return 0


def command_task_autopilot(args) -> int:
    goal = _task_prompt(args.task, args.priority)
    result = Orchestrator().autopilot(
        goal,
        mode=args.mode,
        reuse_profile=not getattr(args, "no_reuse", False),
    )
    if args.json:
        _print_json(result)
    else:
        _print_status(ProfileStore().current())
    return 0


def command_benchmark(args) -> int:
    current = ProfileStore().current()
    if current.get("status") != "READY" or not current.get("candidate"):
        print("No active profile. Run localpilot autopilot first.")
        return 2
    candidate = CandidatePlan(**current["candidate"])
    runtime = runtime_factory(candidate.runtime)()
    try:
        runtime.load_model(candidate)
        runtime.start_model()
        metrics = BenchmarkRunner().run(
            runtime,
            candidate,
            "coding" if "coder" in candidate.model_id else "chat",
        )
    finally:
        runtime.stop_model()
    _print_json(metrics)
    return 0


def command_status(args) -> int:
    state = ProfileStore().current()
    if args.json:
        _print_json(state)
    else:
        _print_status(state)
    return 0


def command_stop(args) -> int:
    ProfileStore().stop()
    print("LocalPilot state is STOPPED. Saved optimization profiles were retained.")
    return 0


def command_serve(args) -> int:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed. Install the API extra: pip install -e .[api]")
        return 2
    from localpilot.api.server import create_app

    host = os.environ.get("LOCALPILOT_HOST", args.host)
    port = int(os.environ.get("LOCALPILOT_PORT", args.port))
    uvicorn.run(create_app(), host=host, port=port)
    return 0


def command_demo(args) -> int:
    demo = run_demo(mode=args.mode, goal=args.goal)
    if args.json:
        _print_json(demo)
    else:
        print_demo(demo)
    return 0



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="localpilot",
        description="AI compute autopilot for local inference",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor")
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(func=command_doctor)

    profile = subparsers.add_parser("profile")
    profile.add_argument("--simulate", action="store_true")
    profile.set_defaults(func=command_profile)

    recommend = subparsers.add_parser("recommend")
    recommend.add_argument("--task", default="coding", choices=["coding", "chat"])
    recommend.add_argument(
        "--priority",
        default="balanced",
        choices=["latency", "quality", "balanced", "low_memory"],
    )
    recommend.add_argument(
        "--mode", default="auto", choices=["auto", "mock", "openvino"]
    )
    recommend.set_defaults(func=command_recommend)

    for name in ("deploy", "optimize"):
        task_command = subparsers.add_parser(name)
        task_command.add_argument(
            "--task", default="coding", choices=["coding", "chat"]
        )
        task_command.add_argument(
            "--priority",
            default="balanced",
            choices=["latency", "quality", "balanced", "low_memory"],
        )
        task_command.add_argument(
            "--mode", default="auto", choices=["auto", "mock", "openvino"]
        )
        task_command.add_argument("--json", action="store_true")
        task_command.set_defaults(func=command_task_autopilot)

    autopilot = subparsers.add_parser("autopilot")
    autopilot.add_argument("goal")
    autopilot.add_argument(
        "--mode", default="auto", choices=["auto", "mock", "openvino"]
    )
    autopilot.add_argument("--no-reuse", action="store_true")
    autopilot.add_argument("--json", action="store_true")
    autopilot.set_defaults(func=command_autopilot)

    benchmark = subparsers.add_parser("benchmark")
    benchmark.set_defaults(func=command_benchmark)

    status = subparsers.add_parser("status")
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=command_status)

    stop = subparsers.add_parser("stop")
    stop.set_defaults(func=command_stop)

    serve = subparsers.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=command_serve)

    demo = subparsers.add_parser("demo")
    demo.add_argument("--goal", default=DEFAULT_DEMO_GOAL)
    demo.add_argument(
        "--mode", default="mock", choices=["mock", "openvino"]
    )
    demo.add_argument("--json", action="store_true")
    demo.set_defaults(func=command_demo)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        if getattr(args, "json", False):
            _print_json(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        else:
            print(f"LocalPilot failed: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
