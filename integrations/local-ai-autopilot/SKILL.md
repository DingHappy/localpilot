---
name: local-ai-autopilot
description: Choose, validate and remember a local inference configuration on NVIDIA hardware. Use when a request involves local or on-premise LLM serving, DGX Spark, CUDA device selection, vLLM / TensorRT-LLM / SGLang / NIM engine choice, NVFP4 or FP8 quantization, KV cache sizing, speculative decoding, batch width tuning, unified-memory capacity limits, or benchmarking one serving configuration against another. Do not use for cloud deployment, for model training or fine-tuning, or for picking a hosted API model.
---

# Local AI Autopilot

Turn a natural-language goal into a serving configuration that has been
measured on the machine in front of you, and that is written down so the
next identical request does not repeat the search.

The Python CLI owns hardware inspection, candidate generation, execution,
measurement, grading, recovery and profile persistence. Use reasoning to
interpret the goal and to explain the result. Do not use reasoning to
replace a measurement or to estimate a number the CLI can measure.

## Why this is not a device-selection problem

On a discrete-GPU box the first question is which device to use. On a
unified-memory machine such as DGX Spark there is one accelerator drawing
on one pool, and the questions that decide the outcome are different:

- **Capacity is charged on total parameters; decode speed is charged only
  on the active ones.** Generating a token reads every active weight once,
  so a 30B model with 3B active decodes several times faster than a dense
  9B while holding far more capability. Do not assume a smaller model is
  faster.
- **Decode is bandwidth-bound long before it is compute-bound.** That
  inverts tuning advice carried over from HBM-equipped datacentre GPUs.
- **Batching is the one regime that escapes the bandwidth limit**, because
  a single weight read serves the whole batch. It raises aggregate
  throughput and worsens per-stream latency.
- **Speculative decoding spends idle compute to cut memory passes**, which
  is the largest single-stream win available here and stops paying once
  batching has claimed that compute.
- **Weight precision sets both capacity and speed.** NVFP4 is not only
  smaller than BF16, it decodes faster, because there are fewer bytes to
  read per token.

Report these as reasons for a ranking, never as measured results.

## Workflow

Understand → Inspect → Plan → Execute → Measure → Grade → Rank → Remember

1. Confirm the request concerns local inference. Extract task, privacy,
   priority, context length, concurrency, modalities and capabilities.
2. Run `localpilot doctor --json`. Read `real_execution_ready`,
   `platform_id`, `unified_memory`, `memory_bandwidth_gbps` and which
   engines were found. Never recommend an engine that was not found.
3. Run `localpilot registry --task <task>` to see which checkpoints fit
   this machine's memory budget, and which do not and why.
4. Run `localpilot recommend --goal "<goal>" --json` and inspect every
   candidate's engine, precision, KV dtype, batch width, memory estimate
   and `simulated` flag, plus every rejection and its gate.
5. Run `localpilot autopilot "<goal>" --json` for the full loop.
6. Treat a run as real only when `hardware.simulated` and
   `benchmark.simulated` are both false. A `READY` status with
   `simulated: true` is a development result and nothing more.
7. Report the winning configuration, its measured metrics, the candidates
   it beat and on which axes they differed, the score, the profile key,
   and whether a stored profile was reused.
8. When comparing two candidates, state every dimension on which they
   differ. A speed ratio between configurations that differ in precision,
   engine and speculative decoding at once does not establish a claim
   about any one of them.
9. Run `localpilot report` to export the run to `results/` when the user
   wants the result kept, reviewed or written up. Simulated and measured
   exports are named differently on purpose.
10. For command details and output contracts, read `references/cli.md`.

## Composing with NVIDIA's own skills

This skill is the decision and memory layer. It does not replace the
per-task NVIDIA skills, and should hand off to them:

- Serving recipes and routing for a chosen configuration belong to the
  Dynamo skills (`dynamo-router-starter`, `dynamo-recipe-runner`,
  `dynamo-troubleshoot`).
- Jetson targets have their own serving, benchmarking, memory-tuning and
  speculative-decoding skills. Use those on Jetson; this skill's platform
  model covers DGX Spark and discrete CUDA devices.
- When an engine fails to start for a reason outside LocalPilot's bounded
  recovery, hand the error to the engine's own troubleshooting skill
  rather than improvising a fix.

What this skill adds on top of them is the part none of them do: generate
a candidate set, measure the candidates against each other through one
wire protocol, grade output quality against a rubric, rank under an
explicit priority policy, and persist the winner against a hardware
fingerprint so the next run is a lookup.

## Reading a result

- `status: success` on a candidate means that candidate executed, not that
  it won.
- `score` is assigned only after the quality and stability gates pass, and
  it is normalized across the candidates in that run. It is a ranking
  within one run on one machine, never a portable benchmark figure.
- `gate_failures` explains a missing score.
- `quality` is a blend of a keyword signal and a rubric grade. When
  `quality_judge` is null no judge model was reachable, the keyword signal
  stands alone, and the quality number is weak. Say so.
- `profile_reused: true` means a stored profile passed a fresh health
  check and the search was skipped.
- Recovery candidates carry `fallback_of` and `recovery_action`. Do not
  retry past the configured recovery limit.
- `memory_estimate` is a pre-flight estimate. `benchmark.peak_memory_gb`
  is what was observed. Quote the second, and check
  `benchmark.raw.peak_memory_source`: it says whether the figure is an
  observed NVML peak or a fallback to the estimate.
- `benchmark.concurrency` above 1 means the figure is a batch measurement.
  Per-stream throughput and aggregate throughput move in opposite
  directions there, so quote whichever the user's priority is about and say
  which one it is.

## Safety boundaries

- Do not install drivers, change PATH, modify the system Python, or alter
  system services.
- **Do not download model weights implicitly.** LocalPilot refuses to
  start an engine unless the user either points at a server they already
  run (`LOCALPILOT_<ENGINE>_BASE_URL`) or sets
  `LOCALPILOT_ALLOW_ENGINE_LAUNCH=1`. Surface the exact command it would
  run and let the user decide. A checkpoint in this registry can be 328 GB.
- `doctor`, `profile`, `engines` and `registry` are read-only.
- Never present a simulated run as a hardware measurement, and never strip
  the `simulated` flag from a number when reporting it.
- Bind the API to 127.0.0.1 unless the user explicitly authorizes network
  exposure and understands the impact.
- Keep user code, full prompts and model output out of logs.
- `stop` touches only LocalPilot-owned state. Preserve downloaded weights
  and saved profiles.
- The judge must be a separate deployment from the candidate being graded.
  Do not point `LOCALPILOT_JUDGE_BASE_URL` at the candidate's own server.
- If no CUDA device is present, continue in mock mode for development but
  state plainly that real acceptance on the target is still outstanding.

## Stopping conditions

Stop after the configured maximum candidates. If every candidate fails,
return the structured errors, the bounded recovery that was attempted, and
the smallest reversible next step. Do not improvise system-level fixes and
do not widen the search past the configured budget.
