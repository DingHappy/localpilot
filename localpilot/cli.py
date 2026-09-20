from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict

from localpilot.benchmark.runner import BenchmarkRunner
from localpilot.demo import DEFAULT_DEMO_GOAL, print_demo, run_demo
from localpilot.engines.registry import EngineRegistry
from localpilot.executor.process import runtime_factory, runtime_names
from localpilot.hardware.profiler import HardwareProfiler
from localpilot.intent.parser import parse_intent
from localpilot.models.registry import ModelRegistry
from localpilot.orchestrator import Orchestrator
from localpilot.planner.planner import Planner
from localpilot.planner.policies import PolicyEngine
from localpilot.profiles.store import ProfileStore
from localpilot.reporting import export_run, latest_run_id, results_root
from localpilot.schemas import CandidatePlan, MemoryEstimate
from localpilot.sizing import MemoryModel, decode_roofline_tokens_s


MODES = ["auto", "mock"] + [name for name in runtime_names() if name != "mock"]
PRIORITIES = PolicyEngine().priority_names()
TASKS = ModelRegistry().tasks()


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


def _candidate_from_state(state: Dict[str, Any]) -> CandidatePlan:
    payload = dict(state)
    estimate = payload.get("memory_estimate")
    if isinstance(estimate, dict):
        payload["memory_estimate"] = MemoryEstimate(**estimate)
    return CandidatePlan(**payload)


def _print_status(state: Dict[str, Any]) -> None:
    if state.get("status") != "READY":
        print("LocalPilot\n\nStatus         STOPPED")
        return
    candidate = state["candidate"]
    benchmark = state["benchmark"]
    prefix = "SIMULATED " if state.get("simulated") else ""
    speculative = (candidate.get("runtime_config") or {}).get("speculative")

    print("LocalPilot")
    print()
    print(f"Model          {candidate['model_id']}")
    print(f"Source         {candidate['source_id']}")
    print(f"Engine         {candidate.get('engine')}")
    print(f"Device         {prefix}{candidate['device']}")
    print(f"Precision      {candidate['precision']}")
    print(f"Context        {candidate['context_length']}")
    print(f"Concurrency    {candidate.get('concurrency', 1)}")
    print(f"KV cache       {candidate.get('kv_cache_dtype', 'auto')}")
    if speculative:
        print(f"Speculative    on ({speculative.get('draft_source_id')})")
    print()
    print(f"TTFT           {benchmark['ttft_ms']:.2f} ms")
    print(f"Decode         {benchmark['throughput_tokens_s']:.2f} tok/s per stream")
    if benchmark.get("aggregate_throughput_tokens_s"):
        print(
            f"Aggregate      "
            f"{benchmark['aggregate_throughput_tokens_s']:.2f} tok/s"
        )
    print(f"Memory         {benchmark['peak_memory_gb']:.2f} GB peak")
    print(f"Quality        {benchmark['quality']:.3f}")
    if benchmark.get("quality_judge") is not None:
        print(
            f"  keyword      {benchmark.get('quality_keyword')}    "
            f"rubric {benchmark['quality_judge']}"
        )
    print()
    print(f"Score          {state['score']:.2f}/100")
    print("Status         READY")
    if state.get("simulated"):
        print()
        print("WARNING        Simulated run. Not a hardware benchmark.")


def _task_prompt(task: str, priority: str) -> str:
    phrases = {
        "coding": "Deploy a completely local code review AI.",
        "chat": "Deploy a completely local chat AI.",
        "agentic": "Deploy a completely local agent that can call tools.",
        "vision": "Deploy a completely local image understanding model.",
        "audio": "Deploy a completely local speech model.",
        "embedding": "Deploy a completely local embedding model.",
    }
    suffixes = {
        "latency": " Response speed is the priority.",
        "throughput": " Serve many concurrent users.",
        "quality": " Quality is the priority.",
        "low_memory": " Low memory usage is the priority.",
        "long_context": " A very long context is the priority.",
        "balanced": " Balance speed and quality.",
    }
    return phrases.get(task, phrases["chat"]) + suffixes.get(
        priority, suffixes["balanced"]
    )


# ----------------------------------------------------------------------
# commands


def command_doctor(args) -> int:
    profile = HardwareProfiler().profile(simulate=False)
    if args.json:
        _print_json(profile)
        return 0

    accelerator = profile.accelerator
    print("LocalPilot doctor")
    print()
    print(f"OS              {profile.os['name']} {profile.os['release']} "
          f"({profile.os['machine']})")
    print(f"CPU             {profile.cpu['model']} "
          f"({profile.cpu['logical_cores']} cores)")
    print(f"Platform        {profile.platform_id}")
    memory_kind = "unified" if profile.unified_memory else "host"
    print(f"Memory          {profile.memory.get('total_gb')} GB {memory_kind}")
    if profile.memory_bandwidth_gbps:
        print(f"Bandwidth       {profile.memory_bandwidth_gbps} GB/s (published)")
    print(f"Accelerator     {accelerator.get('device_count', 0)} device(s) "
          f"via {accelerator.get('source') or 'none'}")
    for name in accelerator.get("names") or []:
        print(f"                {name}")
    if accelerator.get("driver_version"):
        print(f"Driver          {accelerator['driver_version']}")

    print()
    print("Engines")
    for engine_id, report in profile.engines.items():
        mark = "yes" if report.get("available") else "no "
        detail = report.get("detail") or report.get("binary_path") or ""
        print(f"  [{mark}] {engine_id:14s} {detail}")

    print()
    print(f"Real execution ready: {profile.real_execution_ready}")
    for note in profile.notes:
        print(f"  - {note}")
    return 0


def command_profile(args) -> int:
    _print_json(HardwareProfiler().profile(simulate=args.simulate))
    return 0


def command_engines(args) -> int:
    registry = EngineRegistry()
    probed = registry.probe_all()
    if args.json:
        _print_json(
            [
                {**spec.to_dict(), "probe": probed[spec.engine_id]}
                for spec in registry.all()
            ]
        )
        return 0
    for spec in registry.all():
        report = probed[spec.engine_id]
        mark = "yes" if report["available"] else "no "
        print(f"[{mark}] {spec.engine_id:14s} {spec.display_name}")
        print(f"      knobs: {', '.join(spec.knobs) or 'none'}")
        features = [name for name, value in spec.supports.items() if value]
        print(f"      supports: {', '.join(features) or 'none'}")
        if report.get("detail"):
            print(f"      note: {report['detail']}")
    return 0


def command_registry(args) -> int:
    """Shows the catalogue sized against this machine.

    A checkpoint list on its own is not decision-useful. The same model is
    a good pick or an impossible one depending on the memory pool it has to
    live in, so the sizing is computed here rather than left to the reader.
    """
    hardware = HardwareProfiler().profile(simulate=args.simulate)
    policies = PolicyEngine()
    memory_model = MemoryModel(policies.memory_config)
    registry = ModelRegistry()
    models = registry.for_task(args.task) if args.task else registry.all()

    rows = []
    for model in models:
        estimate = memory_model.estimate(
            model, hardware, context_length=args.context, concurrency=1
        )
        ceiling = decode_roofline_tokens_s(
            model.active_parameter_count_b,
            model.precision,
            hardware.memory_bandwidth_gbps,
            efficiency=policies.bandwidth_efficiency,
            memory_model=memory_model,
        )
        rows.append((model, estimate, ceiling))

    if args.json:
        _print_json(
            [
                {
                    **model.to_dict(),
                    "sizing": estimate.to_dict(),
                    "roofline_decode_tokens_s": ceiling,
                }
                for model, estimate, ceiling in rows
            ]
        )
        return 0

    budget = memory_model.budget_gb(hardware)
    print(f"Registry sized for {hardware.platform_id} "
          f"(budget {budget:.1f} GB, context {args.context})"
          if budget else "Registry (memory pool unknown)")
    print()
    header = (
        f"{'model':40s} {'params':>14s} {'prec':>6s} {'weights':>8s} "
        f"{'total':>8s} {'tok/s':>7s}  fits"
    )
    print(header)
    print("-" * len(header))
    for model, estimate, ceiling in rows:
        params = (
            f"{model.parameter_count_b:g}B/"
            f"{model.active_parameter_count_b:g}B"
            if model.is_mixture_of_experts
            else f"{model.parameter_count_b:g}B"
        )
        print(
            f"{model.model_id:40s} {params:>14s} {model.precision:>6s} "
            f"{model.weights_gb:>7.1f}G {estimate.total_gb:>7.1f}G "
            f"{(f'{ceiling:.0f}' if ceiling else '-'):>7s}  "
            f"{'yes' if estimate.fits else 'NO'}"
        )
    print()
    print("params shows total/active where they differ. Active parameters set")
    print("decode speed; total parameters set capacity. tok/s is a bandwidth")
    print("ceiling, not a measurement.")
    return 0


def command_recommend(args) -> int:
    orchestrator = Orchestrator()
    resolved = orchestrator.resolve_mode(args.mode)
    hardware = HardwareProfiler().profile(simulate=resolved == "mock")
    intent = parse_intent(args.goal or _task_prompt(args.task, args.priority))
    planner = Planner()
    candidates = planner.plan(
        intent, hardware, ModelRegistry().for_task(intent.task), resolved
    )
    selection = planner.last_selection

    if args.json:
        _print_json(
            {
                "intent": intent,
                "hardware": hardware,
                "runtime": resolved,
                "candidates": candidates,
                "rejected": [item.to_dict() for item in (selection.rejected if selection else [])],
                "warning": "SIMULATED candidates" if hardware.simulated else None,
            }
        )
        return 0

    print(f"Intent    {intent.task} / {intent.priority} / "
          f"{intent.context_length} tokens / x{intent.concurrency}")
    print(f"Runtime   {resolved}"
          + ("  (SIMULATED)" if hardware.simulated else ""))
    print()
    print("Candidates")
    for index, candidate in enumerate(candidates, start=1):
        print(f"  {index}. {candidate.candidate_id}")
        print(f"     {candidate.reason}")
    if selection and selection.rejected:
        print()
        print("Ruled out")
        for rejection in selection.rejected:
            print(f"  - {rejection.model_id} [{rejection.gate}] {rejection.reason}")
    return 0


def command_autopilot(args) -> int:
    result = Orchestrator().autopilot(
        args.goal, mode=args.mode, reuse_profile=not args.no_reuse
    )
    if args.json:
        _print_json(result)
        return 0
    if args.trace:
        print("Agent trace")
        for step in result.agent_trace:
            print(f"  [{step.agent:9s}] {step.action:22s} {step.detail}")
        print()
    if result.profile_reused:
        print("PROFILE HIT: reused a verified configuration, search skipped")
        print()
    _print_status(ProfileStore().current())
    return 0


def command_task_autopilot(args) -> int:
    result = Orchestrator().autopilot(
        _task_prompt(args.task, args.priority),
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
    candidate = _candidate_from_state(current["candidate"])
    runtime = runtime_factory(candidate.runtime)()
    try:
        runtime.load_model(candidate)
        runtime.start_model()
        metrics = BenchmarkRunner().run(runtime, candidate, args.task)
    finally:
        try:
            runtime.stop_model()
        except Exception:
            pass
    _print_json(metrics)
    return 0


def command_status(args) -> int:
    state = ProfileStore().current()
    if args.json:
        _print_json(state)
    else:
        _print_status(state)
    return 0


def command_profiles(args) -> int:
    profiles = ProfileStore().list_profiles()
    if args.json:
        _print_json(profiles)
        return 0
    if not profiles:
        print("No profiles remembered yet.")
        return 0
    for profile in profiles:
        flag = " SIMULATED" if profile.get("simulated") else ""
        print(f"{profile['task']}/{profile['priority']}{flag}")
        print(f"  {profile['candidate_id']}")
        print(f"  score {profile['score']}  ttft {profile['ttft_ms']} ms  "
              f"verified {profile['last_verified_at']}")
    return 0


def command_report(args) -> int:
    """Exports a run to results/ for review and for the write-up.

    profiles/ is machine state and is not version controlled. results/ is
    curated evidence that is, because a measurement nobody can review is
    not evidence.
    """
    store = ProfileStore()
    run_id = args.run_id or latest_run_id(store)
    if not run_id:
        print("No runs recorded yet. Run localpilot autopilot first.")
        return 2
    run = store.load_run(run_id)
    if run is None:
        print(f"Unknown run: {run_id}")
        return 2

    markdown_path, json_path = export_run(run)
    if args.json:
        _print_json(
            {
                "run_id": run_id,
                "markdown": str(markdown_path),
                "json": str(json_path),
                "simulated": (run.get("hardware") or {}).get("simulated"),
            }
        )
        return 0

    simulated = (run.get("hardware") or {}).get("simulated")
    print(f"Exported run {run_id[:8]} ({'SIMULATED' if simulated else 'MEASURED'})")
    print(f"  {markdown_path.relative_to(results_root().parent)}")
    print(f"  {json_path.relative_to(results_root().parent)}")
    if simulated:
        print()
        print("This run is simulated. Commit it as orchestration evidence, not")
        print("as a hardware measurement.")
    return 0


def command_stop(args) -> int:
    ProfileStore().stop()
    print("LocalPilot state is STOPPED. Saved profiles were retained.")
    return 0


def command_serve(args) -> int:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed. Install the API extra: "
              "pip install -e '.[api]'")
        return 2
    from localpilot.api.server import create_app

    host = os.environ.get("LOCALPILOT_HOST", args.host)
    port = int(os.environ.get("LOCALPILOT_PORT", args.port))
    print(f"LocalPilot dashboard: http://{host}:{port}/")
    uvicorn.run(create_app(), host=host, port=port)
    return 0


def command_demo(args) -> int:
    demo = run_demo(mode=args.mode, goal=args.goal)
    if args.json:
        _print_json(demo)
    else:
        print_demo(demo)
    return 0


# ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="localpilot",
        description="AI compute autopilot for local inference on NVIDIA hardware",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="inspect this machine")
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(func=command_doctor)

    profile = subparsers.add_parser("profile", help="raw hardware profile JSON")
    profile.add_argument("--simulate", action="store_true")
    profile.set_defaults(func=command_profile)

    engines = subparsers.add_parser("engines", help="list inference engines")
    engines.add_argument("--json", action="store_true")
    engines.set_defaults(func=command_engines)

    registry = subparsers.add_parser(
        "registry", help="model catalogue sized for this machine"
    )
    registry.add_argument("--task", default=None, choices=TASKS)
    registry.add_argument("--context", type=int, default=8192)
    registry.add_argument("--simulate", action="store_true", default=True)
    registry.add_argument(
        "--real", dest="simulate", action="store_false",
        help="size against the detected machine instead of the simulated one",
    )
    registry.add_argument("--json", action="store_true")
    registry.set_defaults(func=command_registry)

    recommend = subparsers.add_parser(
        "recommend", help="plan candidates without executing them"
    )
    recommend.add_argument("--goal", default=None)
    recommend.add_argument("--task", default="coding", choices=TASKS)
    recommend.add_argument("--priority", default="balanced", choices=PRIORITIES)
    recommend.add_argument("--mode", default="auto", choices=MODES)
    recommend.add_argument("--json", action="store_true")
    recommend.set_defaults(func=command_recommend)

    for name in ("deploy", "optimize"):
        task_command = subparsers.add_parser(name)
        task_command.add_argument("--task", default="coding", choices=TASKS)
        task_command.add_argument(
            "--priority", default="balanced", choices=PRIORITIES
        )
        task_command.add_argument("--mode", default="auto", choices=MODES)
        task_command.add_argument("--no-reuse", action="store_true")
        task_command.add_argument("--json", action="store_true")
        task_command.set_defaults(func=command_task_autopilot)

    autopilot = subparsers.add_parser(
        "autopilot", help="the full loop from a natural-language goal"
    )
    autopilot.add_argument("goal")
    autopilot.add_argument("--mode", default="auto", choices=MODES)
    autopilot.add_argument("--no-reuse", action="store_true")
    autopilot.add_argument(
        "--trace", action="store_true", help="print the agent trace"
    )
    autopilot.add_argument("--json", action="store_true")
    autopilot.set_defaults(func=command_autopilot)

    benchmark = subparsers.add_parser(
        "benchmark", help="re-measure the active profile"
    )
    benchmark.add_argument("--task", default="coding", choices=TASKS)
    benchmark.set_defaults(func=command_benchmark)

    status = subparsers.add_parser("status")
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=command_status)

    profiles = subparsers.add_parser("profiles", help="list remembered profiles")
    profiles.add_argument("--json", action="store_true")
    profiles.set_defaults(func=command_profiles)

    report = subparsers.add_parser(
        "report", help="export a run to results/ as Markdown and JSON"
    )
    report.add_argument(
        "run_id", nargs="?", default=None, help="defaults to the latest run"
    )
    report.add_argument("--json", action="store_true")
    report.set_defaults(func=command_report)

    stop = subparsers.add_parser("stop")
    stop.set_defaults(func=command_stop)

    serve = subparsers.add_parser("serve", help="dashboard and OpenAI-compatible API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=command_serve)

    demo = subparsers.add_parser("demo", help="the A/B story")
    demo.add_argument("--goal", default=DEFAULT_DEMO_GOAL)
    demo.add_argument("--mode", default="mock", choices=MODES)
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
                {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            )
        else:
            print(f"LocalPilot failed: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
