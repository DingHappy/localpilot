---
name: local-ai-autopilot
description: Turn a local or SSH inference goal into an evidence-labelled serving profile with LocalPilot. Use for Apple Silicon, DGX Spark, and CUDA targets when choosing or validating model, engine, precision, context, concurrency, acceptance, or re-planning decisions. Attempt real validation when the target is ready and clearly distinguish measured, estimated, and simulated evidence. Do not use for hosted API selection, training or fine-tuning, generic accelerator setup, Jetson workflows, or cloud deployment.
---

# Local AI Autopilot

Use LocalPilot as the inspection, planning, acceptance, and profile-memory
tool. Interpret the user's goal and the CLI evidence; never replace a required
check with intuition or invent a measurement.

## Preconditions

Before using a LocalPilot workflow, verify that its executable is available:

```bash
command -v localpilot
localpilot --version
```

This Skill targets the LocalPilot `0.2.x` CLI contract. Report another major
or minor series as incompatible until its command contract is reviewed.

For an SSH target, verify the node through LocalPilot's target mechanism:

```bash
localpilot --target ssh://user@host --version
```

If the CLI is absent or incompatible, report that prerequisite and stop. Do
not install it implicitly. If an engine is absent, mock mode may validate the
orchestration, but real target acceptance remains outstanding.

## Route by intent

Choose the smallest workflow that answers the request:

| User goal | Workflow |
|---|---|
| Inspect the machine or engines | `doctor` and/or `engines` |
| Analyse capacity and candidates | `doctor` → `registry` → `recommend` |
| Run complete acceptance | `autopilot` |
| Re-measure an active profile | `status` → `benchmark` |
| Detect workload or requirement drift | `watch` or `reconcile` |
| Apply a validated configuration | `stage` → `drain` → `activate` → `resume` |
| Interpret an existing report | Read the supplied JSON or Markdown only |
| Explain a performance difference | Inspect every differing axis; run a controlled pair only if needed |

Add `--target ssh://user@host` to every command that belongs on a remote node.
Use `registry --simulate` only when the user explicitly wants the simulated
DGX Spark model. Registry otherwise sizes against the detected target.

## Evidence contract

- A result is measured only when both `hardware.simulated` and
  `benchmark.simulated` are false.
- Preserve capacity rejections, gate failures, recovery actions, and the
  exact target that produced the evidence.
- Invalidate profile reuse when task, privacy, priority, quality, context,
  concurrency, capabilities, modalities, or languages change.
- A successful candidate merely ran; it wins only after gates and ranking.
- Do not infer that one knob caused a difference when other axes changed.
- If `quality_judge` is null, describe quality evidence as weak.
- Treat peak memory as observed only when `peak_memory_source` identifies an
  engine or sampled pool; otherwise it is an estimate.

Read [references/evidence.md](references/evidence.md) before interpreting or
publishing results.

## Target and safety rules

- The controller may be any machine with the Agent, Python, LocalPilot, and
  SSH. Hardware and benchmark evidence must come from the inference target.
- Never download weights, install drivers, change PATH or system Python, or
  modify system services implicitly.
- Bind the API to `127.0.0.1` unless the user authorizes exposure.
- Keep prompts, user code, model output, credentials, and SSH passwords out of
  logs, reports, and target URIs.
- Use a separate deployment for the judge; a candidate must not certify
  itself.
- Do not react to one transient spike. Respect reconcile windows and cooldowns.
- Before a serving change, drain traffic. Validate and prewarm before
  activation, retain the previous profile, and always resume after success or
  an aborted attempt.
- If every candidate fails, return the structured errors and the smallest
  reversible next step. Do not expand or retry beyond the configured limit.

## References

- Read [references/cli.md](references/cli.md) for commands, targets, and
  guarded configuration changes.
- Read [references/platforms.md](references/platforms.md) when platform
  characteristics affect candidate selection.
- Read [references/composition.md](references/composition.md) only when the
  task may continue into an NVIDIA, Jetson, or engine-specific workflow.
- Read [references/installation.md](references/installation.md) when installing
  or updating this Skill.
- Read [references/evaluation.md](references/evaluation.md) only when changing,
  reviewing, or benchmarking the Skill itself.
