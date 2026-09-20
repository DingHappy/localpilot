# LocalPilot run — vision / balanced

**MEASURED** · run `3be3abda` · 2026-09-20T17:07:25.258232+00:00

## Goal

> 本地图片理解，帮我认发票，图片不能离开这台机器

Parsed as **vision** / **balanced**, 8192 tokens, concurrency 1, privacy `local_only`.

## Machine

| field                | value            |
|----------------------|------------------|
| Platform             | dgx_spark        |
| Accelerator          | NVIDIA GB10      |
| Memory               | 119.7 GB unified |
| Bandwidth            | 273 GB/s         |
| Driver               | 580.82.09        |
| Engines found        | vllm             |
| Real execution ready | True             |

## Candidates

| model                   | engine | precision | flags | ttft  | tok/s/stream | tok/s aggregate | peak memory | score |
|-------------------------|--------|-----------|-------|-------|--------------|-----------------|-------------|-------|
| step3-vl-10b-fp8 **<-** | vllm   | FP8       | -     | 55 ms | 20.1         | 20.1            | 55.0 GB     | 65.0  |

## Chosen configuration

| field              | value                                     |
|--------------------|-------------------------------------------|
| Model              | step3-vl-10b-fp8                          |
| Source             | `stepfun-ai/Step3-VL-10B-FP8`             |
| Engine             | vllm                                      |
| Precision          | FP8                                       |
| Context            | 8192                                      |
| Concurrency        | 1                                         |
| KV cache           | auto                                      |
| TTFT               | 55.45 ms                                  |
| Decode             | 20.11 tok/s                               |
| Peak memory        | 55.04 GB                                  |
| Peak memory source | engine_reported_weights_plus_allocated_kv |
| Quality (blended)  | 1.000                                     |
| Quality (keyword)  | 1.000                                     |
| Quality (rubric)   | -                                         |
| Score              | 65.00 / 100                               |
| Profile key        | `649ea2d9967a5cac6b59`                    |

Quality is the keyword signal alone: no judge model was reachable, so the figure is weak and is reported as such.

## Agent trace

```
[planner  ] read_hardware          success   dgx_spark: 119.7 GB unified, 273 GB/s
[planner  ] restrict_to_served     degraded  Attached to a server already running ['step3-vl-10b-fp8'], so the plan can only measure that. Dropped 2 model(s) this server cannot answer for: step-3.7-flash-nvfp4, nemotron-nano-12b-v2-vl-fp8. Comparing across models needs one server per model.
[planner  ] reconcile_with_server  degraded  The attached server runs one configuration (kv=auto, util=0.5, max_len=8192), so 1 candidate(s) whose knobs differ were dropped rather than measured under a label they do not match: step3-vl-10b-fp8-vllm-fp8-kvfp8 [KV cache dtype declared fp8, server runs auto]
[planner  ] adopt_server_knobs     degraded  Adopted the server's own setting rather than measuring under a label it does not match: step3-vl-10b-fp8-vllm-fp8: utilization 0.7 -> 0.5; step3-vl-10b-fp8-vllm-fp8-kvfp8: utilization 0.7 -> 0.5
[planner  ] propose_candidate      success   vLLM ranks #1 for balanced; 16.7 GB of a 83.8 GB budget
[planner  ] plan_complete          success   1 candidates across 1 models and 1 engines
[judge    ] judge_unavailable      degraded  No judge model answered; the keyword score stands alone and the profile records that.
[bench    ] measure_candidate      success   step3-vl-10b-fp8-vllm-fp8: TTFT 55 ms, 20.1 tok/s/stream, 55.0 GB peak, 20.1 tok/s aggregate
[optimizer] rank_candidates        success   step3-vl-10b-fp8-vllm-fp8 wins with 65.00/100 under priority 'balanced'
[memory   ] save_profile           success   Profile 649ea2d9967a5cac6b59 stored for vision/balanced on this fingerprint
```

## Reproduce

```bash
localpilot autopilot "本地图片理解，帮我认发票，图片不能离开这台机器" \
    --mode vllm --no-reuse --json
```

_Exported 2026-09-20T17:08:45.701315+00:00 by `localpilot report`._
