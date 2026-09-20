# LocalPilot

AI that optimizes AI.

LocalPilot is an AI compute autopilot.

Tell it what AI capability you want. It decides how to run it, validates the
decision with a real execution, benchmarks viable configurations, and remembers
the best result for the next run.

## Current status

This repository is the first runnable development slice.

- The intent, hardware, planning, benchmark, scoring, profile-memory, CLI, and
  API paths are implemented.
- Mock mode runs on Python 3.10+ and is always labelled simulated.
- The OpenVINO adapter is present but requires an Intel target machine,
  OpenVINO packages, and a local OpenVINO IR model.
- No command installs drivers, changes PATH, or modifies the system Python.
- No simulated result is eligible to be reported as a real hardware benchmark.

## Development quick start

Run directly from the repository with Python 3.10 through 3.14:

    python3 -m localpilot.cli doctor --json
    python3 -m localpilot.cli autopilot \
      "帮我部署一个完全本地运行的代码审查 AI。我的代码不能离开这台电脑。质量要够用，但响应速度优先。" \
      --mode mock
    python3 -m localpilot.cli status

Run the same autopilot command again to verify profile reuse.

## CLI

    localpilot doctor
    localpilot profile
    localpilot recommend --task coding
    localpilot deploy --task coding
    localpilot benchmark
    localpilot optimize --task coding
    localpilot status
    localpilot stop
    localpilot autopilot "我要一个本地代码审查 AI，代码不能上传云端，响应速度优先"
    localpilot demo
    localpilot serve

Every command supports machine-readable JSON where useful. Use mock mode only
for development and flow validation:

    localpilot autopilot "local coding assistant, latency first" --mode mock --json

Real execution is intentionally explicit:

    export LOCALPILOT_MODEL_PATH=/absolute/path/to/openvino-ir-model
    localpilot autopilot "local coding assistant, latency first" --mode openvino

The real path does not download or convert models automatically. Model
installation and system-driver changes require a separate, reviewable action.

## API

Install the API extra, run an autopilot selection, and start the foreground
server:

    python3 -m pip install -e ".[api]"
    localpilot autopilot "local coding assistant, latency first" --mode mock
    localpilot serve

Endpoints:

- GET /health
- GET /v1/models
- POST /v1/chat/completions

The server binds to 127.0.0.1 by default. The MVP currently supports
non-streaming chat completions.

## Competition demo

Run the A/B story without requiring Intel hardware:

    localpilot demo --mode mock

Demo A lists the manual decisions a user would otherwise make. Demo B sends the
competition prompt through Intent, Profile, Plan, Execute, Benchmark, Optimize,
and Remember. Every development number remains visibly labelled simulated.

When an Intel target is available, repeat the same command with real mode:

    export LOCALPILOT_MODEL_PATH=/absolute/path/to/openvino-ir-model
    localpilot demo --mode openvino

Fallback is bounded. If every primary candidate fails, LocalPilot first retries
the leading candidate with a smaller context and then tries the configured CPU
fallback. It records the parent candidate and recovery action in JSON.

## Architecture

The Codex skill is a thin decision and safety layer under
integrations/local-ai-autopilot. Deterministic operations live in this Python
package.

The runtime boundary allows later providers without changing the orchestration
contract:

- MockRuntime: deterministic development-only execution.
- OpenVINORuntime: real OpenVINO GenAI execution when dependencies and a model
  are available.
- Future: VLLMRuntime, SGLangRuntime, TensorRTRuntime.

The first real acceptance target is one Intel AI PC, one pinned OpenVINO stack,
one coding model artifact, and at least two measured device configurations.

## Safety and privacy

- Local-only requests never send prompt or code content to a cloud service.
- Structured logs record execution metadata, not full prompts or model output.
- Doctor and profile commands are read-only.
- Model downloads, driver installation, and system changes are never implicit.
- Stop only updates LocalPilot-owned state in this MVP; the foreground API is
  stopped with Ctrl-C.

## Tests

The standard-library suite does not require OpenVINO:

    python3 -m unittest discover -s tests -v

Real Intel integration tests will be added and marked separately when target
hardware is available.
