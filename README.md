# LocalPilot

**AI that configures AI.** Say what you want to run locally. LocalPilot picks
the engine, the precision and the serving configuration, measures the
candidates against each other on your machine, and remembers the winner.

Built for NVIDIA DGX Spark and other CUDA targets, and shipped with an
[Agent Skill](integrations/local-ai-autopilot/SKILL.md) so a coding agent can
drive the whole loop.

## The problem

A DGX Spark gives you 128 GB of unified memory and roughly 1 PFLOP of FP4
compute in a desktop box. Then it asks you nine questions before the first
token:

which engine (vLLM, TensorRT-LLM, SGLang, a NIM container, llama.cpp),
which checkpoint, at NVFP4 or FP8 or BF16, whether the weights plus the KV
cache fit a pool shared with the operating system, what context length, what
batch width, whether to run speculative decoding, whether to quantize the KV
cache, and how to benchmark any of it comparably.

**The tuning intuition you carried over from a datacentre GPU is wrong here.**
Capacity is generous and bandwidth is not, so decode is bandwidth-bound long
before it is compute-bound. Three consequences, each of which LocalPilot
turns into a searchable axis:

| Consequence | Why |
|---|---|
| A bigger model can decode faster | Generating a token reads every *active* weight once. A 30B model with 3B active outruns a dense 9B. |
| NVFP4 is faster than BF16, not just smaller | Fewer bytes per token to read. |
| Speculative decoding is the biggest single-stream win | It spends idle compute to cut memory passes — and stops paying once batching claims that compute. |

From the shipped registry, sized for a 128 GB unified pool:

```
model                                        params   prec  weights   tok/s  fits
nemotron-3.5-lightning-30b-a3b-nvfp4         30B/3B  NVFP4    20.1G     121  yes
nemotron-3.5-lightning-30b-a3b-bf16          30B/3B   BF16    61.3G      34  yes
nemotron-3-super-120b-a12b-nvfp4           120B/12B  NVFP4    74.8G      30  yes
nemotron-3-nano-4b-bf16                          4B   BF16     7.4G      26  yes
nemotron-3-ultra-550b-a55b-nvfp4           550B/55B  NVFP4   328.1G       7  NO
```

The 4B dense model is the slowest thing that fits. That is the platform, and
it is why guessing does not work.

## What LocalPilot does

```
Understand → Inspect → Plan → Execute → Measure → Grade → Rank → Remember
```

Three agents with deliberately separated authority. The planner may propose
but never measure. The bench agent may measure but never grade quality. The
judge may grade quality but never rank. Nothing marks its own homework.

```
$ localpilot autopilot "帮我部署一个完全本地运行的代码审查 AI，代码不能离开这台电脑，响应速度优先"

  configuration                                    ttft    tok/s   mem   score
  ------------------------------------------------------------------------------
 *nemotron-3.5-lightning-30b-a3b-nvfp4-trtllm-spec  63m    215.4  23.7G   96.7
  nemotron-3-nano-30b-a3b-nvfp4-trtllm              62m    121.8  20.4G   84.8
  nemotron-3-super-120b-a12b-nvfp4-trtllm           63m     30.2  82.6G   58.6
  nemotron-3.5-lightning-30b-a3b-bf16-vllm          76m     31.5  68.6G   27.5

What the run showed
  - 215 vs 122 tok/s (1.8x) with speculative decoding on vs off
  - 122 vs 30 tok/s (4.0x) with active parameters 3 vs 12, and 62 GB less memory
  - nemotron-3-ultra-550b-a55b-nvfp4 was rejected before anything ran:
    needs 348.0 GB but only 102.4 GB is safely available
```

Every comparison names the axes on which the pair differed. A ratio between
configurations that differ in three things is not evidence about any one of
them, and LocalPilot will not present it as such.

The winner is written to a profile keyed on a hardware fingerprint plus the
task and priority. **The next identical request is a lookup, not a search.**

## Quick start

No CUDA device needed to see the whole loop:

```bash
python3 -m localpilot.cli doctor                  # what this machine is
python3 -m localpilot.cli registry --task coding  # what fits, and what does not
python3 -m localpilot.cli demo --mode mock        # the A/B story
python3 -m localpilot.cli autopilot "local code review AI, latency first" \
    --mode mock --trace
```

Mock mode is a roofline simulation: its numbers are derived from memory
bandwidth and active parameters per token, are labelled `simulated: true`
throughout, and are never reported as measurements.

### Dashboard

```bash
python3 -m pip install -e ".[api]"
python3 -m localpilot.cli serve          # http://127.0.0.1:8000/
```

Hardware panel, live agent trace, candidate comparison, the rejections and
their reasons, and the remembered profiles. Everything is inline — a
local-only tool whose UI needs a CDN would contradict itself.

### Real hardware

LocalPilot measures every engine through its OpenAI-compatible HTTP surface,
which is what makes a cross-engine comparison mean something: TTFT and
inter-token latency are observed identically whether the server is vLLM,
TensorRT-LLM, SGLang or a NIM container.

Point it at a server you already run:

```bash
export LOCALPILOT_VLLM_BASE_URL=http://127.0.0.1:8000
localpilot autopilot "local code review AI, latency first" --mode vllm
```

Or let it start one, which is opt-in because starting an engine downloads
weights:

```bash
export LOCALPILOT_ALLOW_ENGINE_LAUNCH=1
```

Without either, LocalPilot refuses and prints the exact command it would have
run. One checkpoint in this registry is 328 GB; nothing gets pulled by
accident.

Rubric grading needs a separate endpoint, because a model cannot certify its
own answers:

```bash
export LOCALPILOT_JUDGE_BASE_URL=http://127.0.0.1:8009
```

See [`.env.example`](.env.example) for every variable.

## Commands

```
localpilot doctor        this machine: platform, memory, bandwidth, engines
localpilot engines       engine catalogue, knobs, and what was found
localpilot registry      models sized against this machine's budget
localpilot recommend     plan candidates without executing anything
localpilot autopilot     the full loop from a natural-language goal
localpilot deploy        autopilot by task and priority
localpilot optimize      the same, for re-tuning
localpilot benchmark     re-measure the active profile
localpilot status        the active configuration
localpilot profiles      what has been remembered
localpilot demo          the A/B story
localpilot serve         dashboard and OpenAI-compatible API
localpilot stop          release LocalPilot state, keep weights and profiles
```

`--json` everywhere output is meant to be parsed.
[`references/cli.md`](integrations/local-ai-autopilot/references/cli.md) is the
full contract.

## The Agent Skill

`integrations/local-ai-autopilot/` is the decision and memory layer for a
coding agent. It is designed to **compose with** NVIDIA's own skills rather
than duplicate them: serving recipes and routing belong to the Dynamo skills,
Jetson targets have their own, and an engine that fails outside LocalPilot's
bounded recovery is handed to that engine's troubleshooting skill.

What it adds is the part none of those do: generate a candidate set, measure
the candidates against each other, grade output quality against a rubric,
rank under an explicit priority policy, and persist the winner so the next run
is a lookup.

## Architecture

```
localpilot/
  intent/        natural language → task, priority, context, concurrency, modality
  hardware/      CUDA and DGX Spark detection, engine probing
  sizing.py      memory model and the bandwidth roofline
  engines/       engine catalogue: knobs, features, launch templates
  models/        registry and gating, with every rejection kept
  planner/       candidate generation over engine x precision x context x batch
  runtime/       one OpenAI-compatible harness per engine, plus the simulation
  executor/      per-candidate lifecycle: load, start, verify, measure, stop
  agents/        planner, bench and judge, with separated authority
  optimizer/     priority-weighted ranking behind quality and stability gates
  profiles/      profile memory keyed on a hardware fingerprint
  api/           dashboard, jobs, OpenAI-compatible endpoints
  web/           the dashboard, no external assets
config/
  models.yaml    checkpoints, with provenance for every number
  engines.yaml   engines, knobs, launch templates
  devices.yaml   platform detection and characteristics
  policies.yaml  priority weights, search space, gates, roofline constants
  benchmark.yaml prompt set, and the canned answers mock mode returns
```

The runtime boundary is an abstract six-method provider, so a new engine is a
subclass and a config entry rather than a change to the orchestration.

## What is honest about the numbers

- `hardware.simulated` and `benchmark.simulated` must both be false before a
  result is a measurement. `READY` with `simulated: true` is a development
  result.
- `score` is normalized across the candidates of one run. It ranks within that
  run on that machine and is not portable.
- `memory_estimate` is pre-flight. `benchmark.peak_memory_gb` is observed.
- `quality` blends a keyword signal with a rubric grade. A null
  `quality_judge` means no judge was reachable and the quality figure is weak —
  the profile records that rather than hiding it.
- `config/models.yaml` carries a `_provenance` block: which numbers are
  measured file sizes, which are derived from `config.json`, and which are
  priors. Every entry stays `unverified_on_target` until a real run on the
  target writes a profile.

## Safety

- No implicit weight downloads, driver installs, PATH changes or system
  service edits.
- `doctor`, `profile`, `engines` and `registry` are read-only.
- The API binds to 127.0.0.1.
- Logs record execution metadata, not prompts, code or model output.
- `stop` touches only LocalPilot state; weights and profiles survive.

## Tests

```bash
python3 -m unittest discover -s tests     # 108 tests, standard library only
```

Real-hardware integration tests will be added and marked separately once the
target is available.

## Status

The orchestration, search space, sizing model, agents, scoring, profile
memory, CLI, API and dashboard are implemented and tested. The real-execution
path is implemented against the OpenAI-compatible surface of vLLM,
TensorRT-LLM, SGLang, NIM and llama.cpp, and is **pending acceptance on a
DGX Spark** — every number in this repository today is simulated and labelled
as such.
