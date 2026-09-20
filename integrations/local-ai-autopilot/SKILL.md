---
name: local-ai-autopilot
description: Plan, deploy, benchmark, and optimize local AI inference when requests mention local AI, OpenVINO, Intel CPU/GPU/NPU, model deployment, local model optimization, device selection, or benchmarks. Use LocalPilot for deterministic execution; do not use this skill for cloud deployment or model training.
---

# Local AI Autopilot

Turn a natural-language local AI goal into a verified LocalPilot configuration.
The Python CLI owns hardware inspection, execution, benchmarking, fallback, and
profile persistence. Use reasoning to interpret goals and explain results, not
to replace deterministic commands.

## Workflow

Follow this order:

Understand → Profile → Plan → Execute → Verify → Benchmark → Optimize → Remember

1. Confirm that the request concerns local inference and extract task, privacy,
   priority, quality, context, concurrency, language, and capabilities.
2. Run localpilot doctor --json before recommending a device or model.
3. Run localpilot recommend with the matching task and inspect every candidate's
   simulated flag, device compatibility, memory gate, and reason.
4. Run localpilot autopilot with the user's original goal.
5. Treat a run as real only when hardware.simulated and benchmark.simulated are
   both false.
6. Report failed candidates and the fallback that succeeded.
7. Report the winning configuration, measured metrics, score, profile key, and
   whether a prior profile was reused.
8. For command details and output contracts, read references/cli.md.

## Safety boundaries

- Do not install drivers, alter PATH, change the system Python, download a large
  model, or modify system services without explicit user approval.
- Doctor and profile are read-only.
- Never present Mock Runtime output as a hardware benchmark.
- Bind the API to 127.0.0.1 unless the user explicitly authorizes network
  exposure and understands the security impact.
- Do not include user code, full prompts, or model responses in logs by default.
- Stop only LocalPilot-owned processes or state. Preserve downloaded models and
  saved optimization profiles.
- If no real Intel AI PC is available, continue in mock mode for development but
  state that real OpenVINO acceptance remains outstanding.

## Stopping conditions

Stop candidate search after the configured maximum candidates. If every
candidate fails, return the structured errors and the smallest reversible next
step. Do not improvise system-level fixes.
