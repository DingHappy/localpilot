# LocalPilot CLI contract

Run from the repository root or from an installed wheel. A wheel uses its
packaged configuration and writes profiles, logs and reports under the XDG
state directory. Set `LOCALPILOT_HOME` only when one directory should own
both custom `config/` and writable state, or set `LOCALPILOT_STATE_HOME` to
move installed state without replacing packaged configuration.

Every command that prints results supports `--json`. Prefer `--json` when
the output will be parsed rather than read.

Before the workflow, verify the CLI independently from the Skill files:

    command -v localpilot
    localpilot --version

## Execution target

Commands run on the current machine by default. The controller operating
system is unrestricted; it needs Python, the Agent, and SSH. To execute on a
remote DGX Spark or another inference node:

    localpilot --target ssh://pilot@dgx-spark --version
    localpilot --target ssh://pilot@dgx-spark doctor --json
    localpilot --target ssh://pilot@dgx-spark \
      autopilot "local invoice understanding, quality first" --mode vllm --json

The remote node must have LocalPilot installed. Use `--remote-command` when
its executable is not on the default remote PATH. The option is accepted
before or after the subcommand. Do not include passwords in the URI.

The remote command runs entirely on the node: hardware detection, engine
processes, profiles, logs, and exported reports are remote state. `serve` is
not proxied; start it on the node and use an SSH tunnel so it can remain bound
to `127.0.0.1`.

For real local Apple Silicon inference, run a model already installed in
Ollama through its local OpenAI-compatible endpoint:

    localpilot doctor --json
    localpilot registry --task chat --json
    localpilot autopilot "local chat assistant, quality first" \
      --mode ollama --json

LocalPilot sets Ollama `reasoning_effort` to `none` during ordinary acceptance
so a short token budget measures visible output instead of hidden reasoning.
Set `LOCALPILOT_OLLAMA_REASONING_EFFORT` to `low`, `medium`, `high`, or `max`
when reasoning tokens are part of the workload requirement.

## Read-only inspection

    localpilot doctor --json
    localpilot profile [--simulate]
    localpilot engines [--json]
    localpilot registry [--task coding] [--context 8192] [--simulate] [--json]

`doctor` reports `real_execution_ready`. When it is false, real deployment
must not be claimed. It also reports `platform_id`, `unified_memory`,
`memory_bandwidth_gbps`, the detected accelerators and the engine probe.

`engines` lists each engine, whether it was found, which knobs it exposes
and which features it supports. An engine that was not found must not be
recommended.

`registry` sizes every checkpoint against this machine's memory budget and
shows total/active parameters, weight size, estimated total footprint, the
bandwidth-derived decode ceiling and whether it fits. It defaults to the
detected execution target. Pass `--simulate` only for the explicit simulated
DGX Spark development profile.

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

- `auto` uses the first servable engine on a supported execution target, and
  falls back to `mock` when no supported accelerator or engine is present.
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
    localpilot stage "<goal>" [--mode auto] [--json]
    localpilot reconcile [--metrics FILE|-] [--goal "<current goal>"] [--json]
    localpilot watch [--url http://127.0.0.1:8000] [--goal "<current goal>"]
    localpilot drain [--url http://127.0.0.1:8000] [--timeout 30]
    localpilot resume [--url http://127.0.0.1:8000]
    localpilot activate <profile-key> [--url http://127.0.0.1:8000]
    localpilot rollback <profile-key> [--url http://127.0.0.1:8000]
    localpilot report [<run_id>] [--json]
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
    GET  /v1/telemetry         privacy-preserving request aggregates
    POST /v1/reconcile         create a plan from the live telemetry window
    GET  /v1/models            OpenAI-style model list
    POST /v1/chat/completions  serve from the active profile
    POST /internal/drain       reject new work and wait for in-flight requests
    POST /internal/resume      reopen traffic after a safe or aborted change
    POST /internal/activate    prewarm, probe, and activate a staged profile

A real search takes minutes, so `POST /v1/autopilot` is asynchronous. Pass
`{"wait": true, "timeout": 120}` for a synchronous reply in mock mode.

`report` exports a run (the latest by default) to `results/` as Markdown
plus its raw JSON. `profiles/` is machine state and is not version
controlled; `results/` is curated evidence that is. The filename carries
`-real-` or `-sim-`.

`stop` preserves profile files and downloaded weights.

## Drift reconciliation

`reconcile` evaluates one aggregated runtime observation against the active
profile. It records a plan but never changes the serving configuration:

    cat metrics.json | localpilot reconcile --metrics - --json

The metrics object may contain `request_p95_ms`, `ttft_p95_ms`, `throughput_tokens_s`,
`error_rate`, `quality`, `available_memory_gb`, `queue_depth`,
`active_requests`, `healthy`, and an ISO-8601 `timestamp`. Thresholds default
to guarded bands around the profile's measured baseline and can be overridden
with `--max-request-ms`, `--max-ttft-ms`, `--min-throughput`, `--max-error-rate`, `--min-quality`,
`--min-available-memory-gb`, and `--max-queue-depth`.

Three consecutive breaches are required by default. `--window` changes that
count and `--cooldown-seconds` prevents repeated actionable plans. The result
is `KEEP`, `REBENCH`, `RECONFIGURE`, or `SWITCH`, plus explicit drain,
validation, canary, and rollback requirements. Plans and sanitized aggregated
observations live under `profiles/control/`; prompts and request bodies are not
stored.

The API proxy records request latency, error rate, active requests, and bounded
counts without retaining prompts or model output. Poll that live window from a
controller with:

    localpilot watch --url http://127.0.0.1:8000 \
      --goal "local code assistant, latency first"

`watch` stops when an actionable plan appears. Use `--continue-after-action`
only when another component owns plan execution and deduplication. `--samples`
sets a bounded observation count for automation or demonstrations; zero keeps
watching until an actionable plan or interruption. JSON mode emits JSON Lines.

Before applying a configuration that requires a restart or model change:

    localpilot drain --timeout 30 --json
    # validate and perform the controlled change only after DRAINED
    localpilot resume --json

While drained, new chat requests receive HTTP 503 and requests already in
flight may finish. If the timeout expires, LocalPilot automatically reopens
traffic and returns a nonzero CLI status; an unsuccessful drain must never
leave the service unavailable.

Use the complete guarded change sequence for a model or runtime change:

    localpilot stage "<new goal>" --json
    localpilot drain --timeout 30 --json
    localpilot activate <staged-profile-key> --json
    localpilot resume --json

`stage` runs the normal bounded search but leaves `current.json` unchanged.
`activate` is refused unless traffic is drained. It prewarms the staged
runtime, checks health, sends a fixed `LOCALPILOT_READY` business probe, and
only then updates the active Profile. On failure it restores the previous
Profile state and attempts to restart its runtime. `rollback` runs the same
guarded activation path for a previous Profile. Activation never reopens
traffic automatically, so its result can be reviewed before `resume`.

## How the numbers are taken

- Concurrent streams are given different prompts. Identical requests would
  measure the prefix cache rather than the engine.
- `max_new_tokens` defaults to 256 so the decode rate is not dominated by
  time-to-first-token.
- Peak memory is sampled during generation via an engine-readable pool such
  as NVML when available. `benchmark.raw.peak_memory_source` says whether it
  was observed or fell back to the planner's estimate.
- The first token is excluded from the decode rate; it is TTFT.
- Warmup runs are discarded.

## Interpreting output

- `hardware.simulated` and `benchmark.simulated` must both be false before
  a result is described as measured.
- `score` is normalized across the candidates of that one run. It ranks
  within a run on a machine; it is not comparable across runs or machines.
- A candidate with no `score` has `gate_failures` explaining why.
- `quality` blends a keyword signal with a rubric grade. A null
  `quality_judge` means no judge was reachable and the quality figure is
  weak.
- `memory_estimate` is pre-flight. `benchmark.peak_memory_gb` is observed only
  when `benchmark.raw.peak_memory_source` identifies an engine or sampled
  pool; `planner_estimate_no_readable_source` means it is estimated.
- `profile_reused: true` means a stored profile passed a health check and
  the search was skipped.
- Candidates with `fallback_of` came from bounded recovery. Do not retry
  past the configured limit.
- When quoting a comparison between two candidates, name every dimension
  on which they differ.
