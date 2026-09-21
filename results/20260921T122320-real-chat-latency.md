# LocalPilot run — chat / latency

**MEASURED** · run `b1742101` · 2026-09-21T12:23:20.700474+00:00

## Goal

> 完全本地运行的中文技术助手，数据不得离开节点，单用户低延迟优先

Parsed as **chat** / **latency**, 8192 tokens, concurrency 1, privacy `local_only`.

## Machine

| field                | value            |
|----------------------|------------------|
| Platform             | dgx_spark        |
| Accelerator          | NVIDIA GB10      |
| Memory               | 119.7 GB unified |
| Bandwidth            | 273 GB/s         |
| Driver               | 580.82.09        |
| Engines found        | ollama, vllm     |
| Real execution ready | True             |

## Candidates

| model                   | engine | precision | flags | ttft  | tok/s/stream | tok/s aggregate | memory  | score |
|-------------------------|--------|-----------|-------|-------|--------------|-----------------|---------|-------|
| step3-vl-10b-fp8 **<-** | vllm   | FP8       | -     | 58 ms | 20.2         | 20.2            | 55.0 GB | 55.0  |

## Chosen configuration

| field             | value                                     |
|-------------------|-------------------------------------------|
| Model             | step3-vl-10b-fp8                          |
| Source            | `stepfun-ai/Step3-VL-10B-FP8`             |
| Engine            | vllm                                      |
| Precision         | FP8                                       |
| Context           | 8192                                      |
| Concurrency       | 1                                         |
| KV cache          | auto                                      |
| TTFT              | 57.72 ms                                  |
| Decode            | 20.19 tok/s                               |
| Peak memory       | 55.04 GB                                  |
| Memory source     | engine_reported_weights_plus_allocated_kv |
| Quality (blended) | 1.000                                     |
| Quality (keyword) | 1.000                                     |
| Quality (rubric)  | -                                         |
| Score             | 55.00 / 100                               |
| Profile key       | `dc517d8e44595ff512e9`                    |

Quality is the keyword signal alone: no judge model was reachable, so the figure is weak and is reported as such.

## Agent trace

```
[planner  ] read_hardware          success   dgx_spark: 119.7 GB unified, 273 GB/s
[planner  ] restrict_to_served     degraded  Attached to a server already running ['step3-vl-10b-fp8'], so the plan can only measure that. Dropped 10 model(s) this server cannot answer for: nemotron-3.5-lightning-30b-a3b-nvfp4, nemotron-3.5-lightning-30b-a3b-bf16, nemotron-3-nano-30b-a3b-nvfp4, nemotron-3-super-120b-a12b-nvfp4, nemotron-3-ultra-550b-a55b-nvfp4, nemotron-nano-9b-v2-fp8, nemotron-3-nano-4b-bf16, qwen3.8-27b-mlx, step-3.7-flash-nvfp4, nemotron-nano-12b-v2-vl-fp8. Comparing across models needs one server per model.
[planner  ] reconcile_with_server  degraded  The attached server runs one configuration (kv=auto, util=0.5, max_len=8192), so 1 candidate(s) whose knobs differ were dropped rather than measured under a label they do not match: step3-vl-10b-fp8-vllm-fp8-kvfp8 [KV cache dtype declared fp8, server runs auto]
[planner  ] adopt_server_knobs     degraded  Adopted the server's own setting rather than measuring under a label it does not match: step3-vl-10b-fp8-vllm-fp8: utilization 0.7 -> 0.5; step3-vl-10b-fp8-vllm-fp8-kvfp8: utilization 0.7 -> 0.5
[planner  ] propose_candidate      success   vLLM ranks #1 for latency; 16.7 GB of a 83.8 GB budget
[planner  ] plan_complete          success   1 candidates across 1 models and 1 engines
[judge    ] judge_unavailable      degraded  No judge model answered; the keyword score stands alone and the profile records that.
[bench    ] measure_candidate      success   step3-vl-10b-fp8-vllm-fp8: TTFT 58 ms, 20.2 tok/s/stream, 55.0 GB peak memory, 20.2 tok/s aggregate
[optimizer] rank_candidates        success   step3-vl-10b-fp8-vllm-fp8 wins with 55.00/100 under priority 'latency'
[memory   ] save_profile           success   Profile dc517d8e44595ff512e9 activated for chat/latency on this fingerprint
```

## Reproduce

```bash
localpilot autopilot "完全本地运行的中文技术助手，数据不得离开节点，单用户低延迟优先" \
    --mode vllm --no-reuse --json
```

_Exported 2026-09-21T12:26:41.840844+00:00 by `localpilot report`._
