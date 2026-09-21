# LocalPilot run — chat / quality

**MEASURED** · run `0d05f894` · 2026-09-21T04:05:28.640781+00:00

## Goal

> local chat assistant, quality first

Parsed as **chat** / **quality**, 8192 tokens, concurrency 1, privacy `local_only`.

## Machine

| field                | value           |
|----------------------|-----------------|
| Platform             | apple_silicon   |
| Accelerator          | Apple M1 Max    |
| Memory               | 64.0 GB unified |
| Bandwidth            | None GB/s       |
| Driver               | -               |
| Engines found        | ollama          |
| Real execution ready | True            |

## Candidates

| model                  | engine | precision | flags | ttft   | tok/s/stream | tok/s aggregate | memory       | score |
|------------------------|--------|-----------|-------|--------|--------------|-----------------|--------------|-------|
| qwen3.8-27b-mlx **<-** | ollama | NVFP4     | -     | 598 ms | 14.3         | 10.8            | 20.0 GB est. | 75.0  |

## Chosen configuration

| field             | value                               |
|-------------------|-------------------------------------|
| Model             | qwen3.8-27b-mlx                     |
| Source            | `qwen3.8:27b-mlx`                   |
| Engine            | ollama                              |
| Precision         | NVFP4                               |
| Context           | 8192                                |
| Concurrency       | 1                                   |
| KV cache          | auto                                |
| TTFT              | 597.66 ms                           |
| Decode            | 14.27 tok/s                         |
| Estimated memory  | 20.00 GB                            |
| Memory source     | planner_estimate_no_readable_source |
| Quality (blended) | 1.000                               |
| Quality (keyword) | 1.000                               |
| Quality (rubric)  | -                                   |
| Score             | 75.00 / 100                         |
| Profile key       | `edb77cab9283c5bf1cbc`              |

Quality is the keyword signal alone: no judge model was reachable, so the figure is weak and is reported as such.

Memory is a planner estimate because the engine exposed no readable allocation metric; it is not an observed peak.

## Agent trace

```
[planner  ] read_hardware          success   apple_silicon: 64.0 GB unified, unknown GB/s
[planner  ] restrict_to_served     degraded  Attached to a server already running ['qwen3.8:27b-mlx'], so the plan can only measure that. Dropped 10 model(s) this server cannot answer for: nemotron-3.5-lightning-30b-a3b-nvfp4, nemotron-3.5-lightning-30b-a3b-bf16, nemotron-3-nano-30b-a3b-nvfp4, nemotron-3-super-120b-a12b-nvfp4, nemotron-3-ultra-550b-a55b-nvfp4, nemotron-nano-9b-v2-fp8, nemotron-3-nano-4b-bf16, step-3.7-flash-nvfp4, step3-vl-10b-fp8, nemotron-nano-12b-v2-vl-fp8. Comparing across models needs one server per model.
[planner  ] propose_candidate      success   Ollama ranks #1 for quality; 20.0 GB of a 44.8 GB budget
[planner  ] plan_complete          success   1 candidates across 1 models and 1 engines
[judge    ] judge_unavailable      degraded  No judge model answered; the keyword score stands alone and the profile records that.
[bench    ] measure_candidate      success   qwen3.8-27b-mlx-ollama-nvfp4: TTFT 598 ms, 14.3 tok/s/stream, 20.0 GB estimated memory, 10.8 tok/s aggregate
[optimizer] rank_candidates        success   qwen3.8-27b-mlx-ollama-nvfp4 wins with 75.00/100 under priority 'quality'
[memory   ] save_profile           success   Profile edb77cab9283c5bf1cbc stored for chat/quality on this fingerprint
```

## Reproduce

```bash
localpilot autopilot "local chat assistant, quality first" \
    --mode ollama --no-reuse --json
```

_Exported 2026-09-21T05:10:42.953341+00:00 by `localpilot report`._
