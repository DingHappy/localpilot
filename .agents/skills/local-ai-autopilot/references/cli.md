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

## Profiles, remeasurement and handoff

    localpilot status [--json]
    localpilot profiles [--json]
    localpilot benchmark [--task coding]
    localpilot stage "<goal>" [--mode auto] [--json]
    localpilot report [<run_id>] [--json]

`stage` measures and saves a Profile while leaving current.json unchanged.
It is suitable when the user wants a validated configuration to review or hand
off. `autopilot` normally marks its selected Profile current, but that does not
prove a persistent deployment. For an attached engine, report that it was
already running rather than claiming LocalPilot deployed it.

`report` exports a saved run as Markdown and JSON, including rejected runs.
A fully rejected search returns CLI exit code 1 and `status: REJECTED`; JSON
contains the run ID, candidates and failed gates, with no best Profile.
It does not replace the current Profile. Export the evidence before proposing
one next change; do not silently lower the user's acceptance threshold.

Use [optimization.md](optimization.md) for task-quality gates and controlled
comparisons, and [handoff.md](handoff.md) for configuration and invocation
artifacts. An engine API is sufficient for application inference. LocalPilot's
own gateway, monitoring and live switching commands are optional; read
[service-management.md](service-management.md) only when explicitly requested.

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
- Without `raw.structured_quality`, a null `quality_judge` leaves only a weak
  keyword signal. With `raw.structured_quality`, quality may be deterministic
  document exact match; report field and document accuracy with their sample
  counts. It is not an independent semantic grade or proof on unseen inputs.
- `memory_estimate` is pre-flight. `benchmark.peak_memory_gb` is observed only
  when `benchmark.raw.peak_memory_source` identifies an engine or sampled
  pool; `planner_estimate_no_readable_source` means it is estimated.
- `profile_reused: true` means a stored profile passed a health check and
  the search was skipped.
- Candidates with `fallback_of` came from bounded recovery. Do not retry
  past the configured limit.
- When quoting a comparison between two candidates, name every dimension
  on which they differ.
