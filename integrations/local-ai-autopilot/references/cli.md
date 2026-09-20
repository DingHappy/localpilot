# LocalPilot CLI contract

Run from the repository root, or set `LOCALPILOT_HOME` to the directory
holding `config/` and `profiles/`.

Every command that prints results supports `--json`. Prefer `--json` when
the output will be parsed rather than read.

## Read-only inspection

    localpilot doctor --json
    localpilot profile [--simulate]
    localpilot engines [--json]
    localpilot registry [--task coding] [--context 8192] [--real] [--json]

`doctor` reports `real_execution_ready`. When it is false, real deployment
must not be claimed. It also reports `platform_id`, `unified_memory`,
`memory_bandwidth_gbps`, the detected accelerators and the engine probe.

`engines` lists each engine, whether it was found, which knobs it exposes
and which features it supports. An engine that was not found must not be
recommended.

`registry` sizes every checkpoint against this machine's memory budget and
shows total/active parameters, weight size, estimated total footprint, the
bandwidth-derived decode ceiling and whether it fits. It defaults to the
simulated DGX Spark profile; pass `--real` to size against the machine
actually detected.

## Planning without executing

    localpilot recommend --goal "<goal>" --mode auto --json
    localpilot recommend --task coding --priority latency --json

Returns the candidate set and, in `rejected`, every model that was gated
out with the gate name and the reason. Nothing is started.

## The full loop

    localpilot autopilot "<goal>" [--mode auto] [--no-reuse] [--trace] [--json]
    localpilot deploy --task coding --priority latency
    localpilot optimize --task coding --priority throughput

`--trace` prints the agent trace: which agent did what, in order, with the
rejections and the ranking. `--no-reuse` forces a fresh search.

### Modes

- `auto` uses the first servable engine that was found, and falls back to
  `mock` when no CUDA device or no engine is present.
- `mock` is a deterministic simulation. Its numbers come from a roofline
  model, are labelled `simulated: true`, and are not measurements.
- `vllm`, `trtllm`, `sglang`, `nim`, `llamacpp` force that engine and fail
  loudly rather than silently degrading to a simulation.

### Engine access

Real modes need a reachable server. Either:

    export LOCALPILOT_VLLM_BASE_URL=http://127.0.0.1:8000

to use a server you already run, or:

    export LOCALPILOT_ALLOW_ENGINE_LAUNCH=1

to let LocalPilot start one. Launching downloads weights, so it is opt-in.
Without either, LocalPilot refuses and prints the exact command it would
have run.

Optional:

    LOCALPILOT_ENGINE_PORT     port for a LocalPilot-managed server (default 8100)
    LOCALPILOT_ENGINE_TIMEOUT  seconds to wait for health (default 900)
    LOCALPILOT_JUDGE_BASE_URL  a separate endpoint for rubric grading
    LOCALPILOT_JUDGE_MODEL     model name to request from that endpoint
    LOCALPILOT_MODEL_PATH      local weights, for engines that need a path

The judge must not be the candidate's own server.

## State and service

    localpilot status [--json]
    localpilot profiles [--json]
    localpilot benchmark [--task coding]
    localpilot demo [--mode mock] [--json]
    localpilot stop
    localpilot serve [--host 127.0.0.1] [--port 8000]

`serve` exposes the dashboard at `/` plus:

    GET  /health
    GET  /v1/hardware          this machine, or ?simulate=true
    GET  /v1/engines           engine catalogue and probe results
    GET  /v1/registry          models sized for this machine
    GET  /v1/policies          priorities, search space, gates
    POST /v1/intent            parse a goal without running anything
    POST /v1/autopilot         start a run; returns a job_id
    GET  /v1/autopilot/{id}    poll a run
    GET  /v1/profiles          remembered profiles
    GET  /v1/runs              run history
    GET  /v1/benchmarks        the prompt set and its settings
    GET  /v1/models            OpenAI-style model list
    POST /v1/chat/completions  serve from the active profile

A real search takes minutes, so `POST /v1/autopilot` is asynchronous. Pass
`{"wait": true, "timeout": 120}` for a synchronous reply in mock mode.

`stop` preserves profile files and downloaded weights.

## Interpreting output

- `hardware.simulated` and `benchmark.simulated` must both be false before
  a result is described as measured.
- `score` is normalized across the candidates of that one run. It ranks
  within a run on a machine; it is not comparable across runs or machines.
- A candidate with no `score` has `gate_failures` explaining why.
- `quality` blends a keyword signal with a rubric grade. A null
  `quality_judge` means no judge was reachable and the quality figure is
  weak.
- `memory_estimate` is pre-flight; `benchmark.peak_memory_gb` is observed.
- `profile_reused: true` means a stored profile passed a health check and
  the search was skipped.
- Candidates with `fallback_of` came from bounded recovery. Do not retry
  past the configured limit.
- When quoting a comparison between two candidates, name every dimension
  on which they differ.
