# Optional LocalPilot service management

Read this only for an explicit request concerning LocalPilot's existing API,
telemetry, continuous monitoring or guarded service control. These are auxiliary
features, not the default configuration-selection and handoff workflow.
Do not start `serve`, add a gateway, or redirect application traffic as part of
ordinary acceptance. Applications may call vLLM or Ollama directly.

The drain/activate/resume endpoints below belong to a LocalPilot API instance,
not to the underlying vLLM/Ollama server. Verify the endpoint owner before using
them; these commands do not drain traffic that bypasses that API. Starting a
LocalPilot gateway merely to use these commands expands the task and is not
implied by configuration handoff.

The current LocalPilot chat proxy does not preserve multimodal message content
or forward response_format. Do not route an image/JSON-Schema workload through
it and claim equivalence with the validated direct-engine request path.

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
