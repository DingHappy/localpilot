from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from localpilot.demo import _describe_difference
from localpilot.utils import atomic_write_json, project_home, utc_now


def results_root() -> Path:
    return project_home() / "results"


def _slug(text: str, limit: int = 40) -> str:
    keep = [
        character if character.isalnum() or character in "-_" else "-"
        for character in text.lower()
    ]
    slug = "".join(keep).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug[:limit] or "run"


def _table(rows: List[List[str]], headers: List[str]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    lines = [
        "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |",
        "|" + "|".join("-" * (width + 2) for width in widths) + "|",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"
        )
    return "\n".join(lines)


def _number(value: Any, digits: int = 1, suffix: str = "") -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}{suffix}"


def render_markdown(run: Dict[str, Any]) -> str:
    """Renders one run as a reviewable report.

    Written so the evidence travels with the claim: the configuration, the
    numbers, whether they were measured or modelled, what was rejected and
    why, and for each comparison the axes on which the pair differed.
    """
    intent = run.get("intent") or {}
    hardware = run.get("hardware") or {}
    best = run.get("best_profile") or {}
    winner = best.get("candidate") or {}
    simulated = bool(hardware.get("simulated"))
    candidates = run.get("candidates") or []

    lines: List[str] = []
    label = "SIMULATED" if simulated else "MEASURED"
    if not best and not simulated and not any(
        (item.get("benchmark") or {}).get("simulated") is False for item in candidates
    ):
        label = "NOT MEASURED"
    lines.append(f"# LocalPilot run — {intent.get('task')} / {intent.get('priority')}")
    lines.append("")
    lines.append(f"**{label}** · run `{run.get('run_id', '')[:8]}` · "
                 f"{run.get('finished_at') or run.get('started_at') or ''}")
    lines.append("")

    if simulated:
        lines.append("> These numbers come from a roofline model, not from this "
                     "machine. They describe what the search space implies, not "
                     "what the hardware did, and must not be quoted as a "
                     "benchmark.")
        lines.append("")

    lines.append("## Goal")
    lines.append("")
    lines.append(f"> {intent.get('raw_text', '')}")
    lines.append("")
    lines.append(
        f"Parsed as **{intent.get('task')}** / **{intent.get('priority')}**, "
        f"{intent.get('context_length')} tokens, concurrency "
        f"{intent.get('concurrency')}, privacy `{intent.get('privacy')}`."
    )
    lines.append("")

    lines.append("## Machine")
    lines.append("")
    memory_kind = "unified" if hardware.get("unified_memory") else "dedicated"
    accelerator = hardware.get("accelerator") or {}
    lines.append(_table(
        [
            ["Platform", str(hardware.get("platform_id"))],
            ["Accelerator", ", ".join(accelerator.get("names") or []) or "-"],
            ["Memory", f"{(hardware.get('memory') or {}).get('total_gb')} GB {memory_kind}"],
            ["Bandwidth", f"{hardware.get('memory_bandwidth_gbps')} GB/s"],
            ["Driver", str(accelerator.get("driver_version") or "-")],
            ["Engines found", ", ".join((hardware.get("stack") or {}).get("engines_available") or []) or "none"],
            ["Real execution ready", str(hardware.get("real_execution_ready"))],
        ],
        ["field", "value"],
    ))
    lines.append("")

    lines.append("## Candidates")
    lines.append("")
    rows = []
    for item in candidates:
        candidate = item.get("candidate") or {}
        benchmark = item.get("benchmark")
        mark = " **<-**" if candidate.get("candidate_id") == winner.get("candidate_id") else ""
        flags = []
        if (candidate.get("runtime_config") or {}).get("speculative"):
            flags.append("spec")
        if candidate.get("kv_cache_dtype") not in (None, "auto"):
            flags.append(f"kv-{candidate['kv_cache_dtype']}")
        if candidate.get("concurrency", 1) > 1:
            flags.append(f"x{candidate['concurrency']}")
        if candidate.get("recovery_action"):
            flags.append("recovery")
        if not benchmark:
            rows.append([
                f"{candidate.get('model_id')}{mark}",
                str(candidate.get("engine")),
                str(candidate.get("precision")),
                " ".join(flags) or "-",
                "failed", "-", "-", "-",
                str(item.get("error"))[:40],
            ])
            continue
        score = item.get("score")
        memory_source = (benchmark.get("raw") or {}).get("peak_memory_source")
        memory_suffix = (
            " GB est."
            if memory_source == "planner_estimate_no_readable_source"
            else " GB"
        )
        rows.append([
            f"{candidate.get('model_id')}{mark}",
            str(candidate.get("engine")),
            str(candidate.get("precision")),
            " ".join(flags) or "-",
            _number(benchmark.get("ttft_ms"), 0, " ms"),
            _number(benchmark.get("throughput_tokens_s")),
            _number(benchmark.get("aggregate_throughput_tokens_s")),
            _number(benchmark.get("peak_memory_gb"), 1, memory_suffix),
            _number(score, 1) if score is not None
            else "gated: " + "; ".join(item.get("gate_failures") or []),
        ])
    lines.append(_table(rows, [
        "model", "engine", "precision", "flags", "ttft",
        "tok/s/stream", "tok/s aggregate", "memory", "score",
    ]))
    lines.append("")

    comparisons = comparison_notes(candidates)
    if comparisons:
        lines.append("## Comparisons")
        lines.append("")
        lines.append("Each ratio lists every axis on which the pair differed. A "
                     "gap between configurations that differ in three things is "
                     "not evidence about any one of them.")
        lines.append("")
        for note in comparisons:
            lines.append(f"- {note}")
        lines.append("")

    rejections = [
        step for step in run.get("agent_trace") or []
        if step.get("action") == "reject_model"
    ]
    if rejections:
        lines.append("## Ruled out before anything ran")
        lines.append("")
        for step in rejections:
            data = step.get("data") or {}
            lines.append(f"- **{data.get('model_id')}** [{data.get('gate')}] "
                         f"{data.get('reason')}")
        lines.append("")

    if not best:
        lines.extend(["## No accepted configuration", "",
                      f"Status: **{run.get('status', 'REJECTED')}**", "",
                      str(run.get("error") or "No candidate passed acceptance."), "",
                      "No profile was selected or activated by this run.", ""])
        acceptance = run.get("acceptance") or {}
        if acceptance:
            lines.extend(["## Acceptance policy", "",
                          _table([[key, str(value)] for key, value in acceptance.items()], ["constraint", "value"]), "",
                          "The latency budget applies to mean complete response time, not TTFT.", ""])
        lines.extend(["## Agent trace", "", "```"])
        for step in run.get("agent_trace") or []:
            lines.append(f"[{step.get('agent')}] {step.get('action')}: {step.get('detail')}")
        lines.extend(["```", "", "## Next step", "",
                      "Review the failed gates and candidate errors above. Select one supported change "
                      "for a new bounded run, or explicitly revise the acceptance budget. "
                      "Do not treat this rejection as a deployment or silently relax its gates.", "",
                      f"_Exported {utc_now()} by `localpilot report`._"])
        return "\n".join(lines) + "\n"

    lines.append("## Chosen configuration")
    lines.append("")
    benchmark = best.get("benchmark") or {}
    memory_source = (benchmark.get("raw") or {}).get(
        "peak_memory_source", "-"
    )
    memory_label = (
        "Estimated memory"
        if memory_source == "planner_estimate_no_readable_source"
        else "Peak memory"
    )
    detail = [
        ["Model", str(winner.get("model_id"))],
        ["Source", f"`{winner.get('source_id')}`"],
        ["Engine", str(winner.get("engine"))],
        ["Precision", str(winner.get("precision"))],
        ["Context", str(winner.get("context_length"))],
        ["Concurrency", str(winner.get("concurrency"))],
        ["KV cache", str(winner.get("kv_cache_dtype"))],
    ]
    speculative = (winner.get("runtime_config") or {}).get("speculative")
    if speculative:
        detail.append(["Speculative draft", f"`{speculative.get('draft_source_id')}`"])
    detail.extend([
        ["TTFT", _number(benchmark.get("ttft_ms"), 2, " ms")],
        ["Complete response (mean)", _number(benchmark.get("total_latency_ms"), 2, " ms")],
        ["Decode", _number(benchmark.get("throughput_tokens_s"), 2, " tok/s")],
        [memory_label, _number(benchmark.get("peak_memory_gb"), 2, " GB")],
        ["Memory source", str(memory_source)],
        ["Quality (blended)", _number(benchmark.get("quality"), 3)],
        ["Quality (keyword)", _number(benchmark.get("quality_keyword"), 3)],
        ["Quality (rubric)", _number(benchmark.get("quality_judge"), 3)],
        ["Score", _number(best.get("score"), 2) + " / 100"],
        ["Profile key", f"`{best.get('profile_key')}`"],
    ])
    lines.append(_table(detail, ["field", "value"]))
    lines.append("")

    acceptance = (benchmark.get("raw") or {}).get("acceptance")
    if acceptance:
        lines.append("## Acceptance policy")
        lines.append("")
        lines.append(_table([[key, str(value)] for key, value in acceptance.items()], ["constraint", "value"]))
        lines.append("")
        lines.append("Latency limits apply to the measured mean complete response, not TTFT or a tail-latency guarantee. "
                     "A single accepted candidate does not demonstrate an optimization gain. "
                     "Reproduce with the same LOCALPILOT_BENCHMARK_CONFIG.")
        lines.append("")
    structured = (benchmark.get("raw") or {}).get("structured_quality")
    if structured:
        lines.append("## Document field acceptance")
        lines.append("")
        lines.append(_table([
            ["Documents", str(structured["documents"])],
            ["Fields correct / total", f"{structured['fields_correct']} / {structured['fields_total']}"],
            ["Field exact match", _number(structured["field_accuracy"], 3)],
            ["Document exact match", _number(structured["document_accuracy"], 3)],
            ["Valid JSON rate", _number(structured["json_valid_rate"], 3)],
            ["Request failures", str(structured["request_failures"])],
            ["Dataset SHA256", structured["dataset_sha256"]],
        ], ["metric", "value"]))
        lines.append("")
        lines.append("Quality uses exact JSON document matching against fixed labels. "
                     "Failed requests count as incorrect. This is dataset-specific "
                     "field evidence, not an independent semantic judge score.")
        lines.append("Reproduction requires the same benchmark configuration: set "
                     "`LOCALPILOT_BENCHMARK_CONFIG` to its path before running the "
                     "command below. The full config hash is recorded in the raw JSON.")
        lines.append("")
    elif benchmark.get("quality_judge") is None:
        lines.append("Quality is the keyword signal alone: no judge model was "
                     "reachable, so the figure is weak and is reported as such.")
        lines.append("")

    if memory_source == "planner_estimate_no_readable_source":
        lines.append("Memory is a planner estimate because the engine exposed "
                     "no readable allocation metric; it is not an observed peak.")
        lines.append("")

    lines.append("## Agent trace")
    lines.append("")
    lines.append("```")
    memory_sources = {
        (item.get("candidate") or {}).get("candidate_id"):
        ((item.get("benchmark") or {}).get("raw") or {}).get(
            "peak_memory_source"
        )
        for item in candidates
    }
    for step in run.get("agent_trace") or []:
        detail_text = str(step.get("detail"))
        step_data = step.get("data") or {}
        if (
            step.get("action") == "measure_candidate"
            and memory_sources.get(step_data.get("candidate_id"))
            == "planner_estimate_no_readable_source"
        ):
            detail_text = detail_text.replace(" GB peak", " GB estimated memory")
        lines.append(f"[{step.get('agent'):9s}] {step.get('action'):22s} "
                     f"{step.get('status'):9s} {detail_text}")
    lines.append("```")
    lines.append("")

    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    mode = (best.get("runtime_versions") or {}).get("runtime", "mock")
    lines.append(f"localpilot autopilot \"{intent.get('raw_text', '')}\" \\")
    lines.append(f"    --mode {mode} --no-reuse --json")
    lines.append("```")
    lines.append("")
    lines.append(f"_Exported {utc_now()} by `localpilot report`._")
    return "\n".join(lines) + "\n"


def comparison_notes(candidates: List[Dict[str, Any]]) -> List[str]:
    """Pairwise speed ratios, each carrying its confounders."""
    measured = [item for item in candidates if item.get("benchmark")]

    def rate(item):
        return item["benchmark"]["throughput_tokens_s"] or 0.0

    pairs = []
    for index, left in enumerate(measured):
        for right in measured[index + 1:]:
            faster, slower = (
                (left, right) if rate(left) >= rate(right) else (right, left)
            )
            difference = _describe_difference(
                faster["candidate"], slower["candidate"]
            )
            if not difference:
                continue
            axes = difference.count(",") + 1
            pairs.append((axes, -rate(faster) / max(rate(slower), 1e-6),
                          faster, slower, difference))

    pairs.sort(key=lambda entry: (entry[0], entry[1]))
    notes = []
    for _, negative_ratio, faster, slower, difference in pairs[:4]:
        notes.append(
            f"`{faster['candidate']['candidate_id']}` reached "
            f"{rate(faster):.0f} tok/s against {rate(slower):.0f} "
            f"({-negative_ratio:.1f}x), differing in {difference}"
        )
    return notes


def export_run(
    run: Dict[str, Any], root: Path = None
) -> Tuple[Path, Path]:
    """Writes a run to results/ as Markdown plus its raw JSON.

    ``profiles/`` is machine state and stays out of version control.
    ``results/`` is curated evidence meant to be committed, because a
    measurement nobody can review is not evidence.
    """
    target = root or results_root()
    target.mkdir(parents=True, exist_ok=True)

    intent = run.get("intent") or {}
    hardware = run.get("hardware") or {}
    stamp = (run.get("finished_at") or utc_now())[:19].replace(":", "").replace("-", "")
    prefix = "sim" if hardware.get("simulated") else "real"
    name = (
        f"{stamp}-{prefix}-{_slug(str(intent.get('task')))}-"
        f"{_slug(str(intent.get('priority')))}"
    )

    markdown_path = target / f"{name}.md"
    json_path = target / f"{name}.json"
    markdown_path.write_text(render_markdown(run), encoding="utf-8")
    atomic_write_json(json_path, run)
    return markdown_path, json_path


def latest_run_id(store) -> Optional[str]:
    runs = store.list_runs(limit=1)
    return runs[0]["run_id"] if runs else None
