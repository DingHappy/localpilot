---
name: local-ai-autopilot
description: Use for LocalPilot-specific inference configuration and acceptance on one Apple Silicon or NVIDIA/CUDA target, locally or over SSH, or for explicitly requested mock workflow verification. Inspect, compare, measure and hand off a reusable configuration with evidence. Excludes engine installation, CPU-only real acceptance, AMD/ROCm, hardware purchases, generic CUDA debugging and vague local-versus-hosted comparisons. A model or GPU mention alone is insufficient.
---

# Local AI Autopilot

Help the user configure inference for their task and verify whether it meets
that task's requirements. Use LocalPilot CLI for inspection, planning,
measurement and profile memory; use the installed inference engine to run models.
Applications can call that engine directly. A LocalPilot inference gateway,
continuous monitoring or automatic traffic routing is not a prerequisite.

Apply this workflow when the user wants LocalPilot to configure or validate
inference on one supported Apple Silicon or NVIDIA/CUDA target. An explicit
request to check LocalPilot's mock orchestration may also use this Skill, with
every result labelled simulated and real-device acceptance still outstanding.
Installing an engine, buying hardware, generic CUDA troubleshooting, or a vague
local-versus-hosted quality comparison needs its own workflow. For a request
such as "Why is CUDA unavailable?" with no LocalPilot inference goal, inspect
the failing application and its CUDA environment; do not run LocalPilot
`doctor` or `engines` as a shortcut. A CPU-only or
AMD/ROCm target is outside this Skill's real-acceptance scope, even if the user
explicitly asks to use LocalPilot. Decide this from the stated goal and target
before the CLI precheck below: do not call `localpilot --help`, `doctor`,
`status` or `registry` merely to confirm an explicitly unsupported request.
Explain the boundary without substituting mock results for real acceptance.

## Preconditions

Before executing a workflow, verify the CLI independently from this Skill:

```bash
command -v localpilot
localpilot --version
```

This Skill targets the LocalPilot `0.2.x` command contract. Review the contract
before using a different major/minor series. Check newer optional commands with
`--help`; a version match alone does not prove that a particular feature exists.
For an SSH target, verify its CLI with
`localpilot --target ssh://user@host --version` and add that target to commands
that should run there. The remote node needs the CLI, not a copy of this Skill.
If the CLI is missing, stop the LocalPilot workflow and recommend the isolated
installation in [cli.md](references/cli.md#installing-a-missing-cli) on the
machine that needs it. Explain that installing the Skill alone does not install
the CLI. Do not download or install the CLI implicitly; if the user asks you to
set it up, follow the installation steps and verify the CLI before continuing.
Treat an incompatible version as a compatibility gap, not a reason to overwrite
the existing installation automatically.
Read-only interpretation of supplied reports needs no CLI or device connection.

## Route by user goal

Choose the smallest useful workflow; do not run a full search for every request.

| User goal | Workflow |
|---|---|
| Inspect hardware or engines | `doctor` and/or `engines` |
| Analyse capacity and candidates | `doctor` → `registry` → `recommend` |
| Run task acceptance | `autopilot` with the appropriate benchmark configuration |
| Validate for handoff without changing the active Profile | `stage` → `report` |
| Re-measure an active profile | `status` → `benchmark` |
| Respond to a changed task | Compare requirements and acceptance data; run a fresh bounded acceptance when needed |
| Compare speed or quality | Read [optimization.md](references/optimization.md); use a controlled pair if measurements are needed |
| Interpret results | Read existing JSON/Markdown only |
| Deliver a reusable configuration | Read [handoff.md](references/handoff.md); provide the engine endpoint, configuration, request example and evidence |

`registry` sizes against the execution target. Use `--simulate` only for an
explicitly requested simulated DGX development profile. Model entries and
engine feature declarations are candidate metadata, not proof of compatibility.

## Acceptance and evidence

- A measurement requires both hardware and benchmark `simulated` to be false.
- Preserve capacity rejections, failed requests, failed gates and their run IDs.
  `REJECTED` with no best Profile is a reviewable failure, not a deployment.
- Reject candidates missing required modalities or capabilities; do not infer
  support from a model name.
- Requirements, benchmark contents or acceptance-policy changes invalidate old
  acceptance evidence. Inspect stored requirements and hashes before reuse.
- A successful request is not a winning configuration. Apply quality and
  stability gates before ranking; use complete response time for completion budgets.
- Distinguish keyword-only evidence, independent rubric grading and deterministic
  field checks. Exact checks support claims only about their labeled dataset.
- A single candidate cannot prove optimization gain. For comparisons, name the
  changed axes, keep quality checks comparable and distinguish repeated samples
  from new inputs. Report per-stream and aggregate throughput separately.
- Describe memory as observed only when its recorded source is an engine or
  sampled pool; a planner fallback remains an estimate.

Read [evidence.md](references/evidence.md) before interpreting or publishing results.

## Scope and completion

Complete the requested inspection, proposal, acceptance or handoff. For handoff,
state what was measured, what configuration and request strategy to use, and
what remains unverified. A stored `READY` Profile does not prove a continuously
available service or automatically apply request options to the user's app.

Do not download weights, install drivers, edit system services or replace a
running configuration implicitly. Preserve existing workloads; before an
explicitly requested serving change, establish the actual engine's change and
recovery procedure. Keep secrets and user content out of exported evidence.
Use bounded candidates and recovery; report failure instead of expanding the
search or silently relaxing acceptance gates.

## Conditional references

- [cli.md](references/cli.md): core commands and local/SSH execution.
- [platforms.md](references/platforms.md): platform-specific candidate constraints.
- [composition.md](references/composition.md): another specialised Skill is needed
  for the user's requested next step and is available in this environment.
- [service-management.md](references/service-management.md): only when the user
  explicitly requests existing LocalPilot API, telemetry or guarded service control.
