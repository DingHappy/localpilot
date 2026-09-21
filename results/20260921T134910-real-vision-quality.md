# LocalPilot run — vision / quality

**MEASURED** · run `d4e95a9e` · 2026-09-21T13:49:10.628013+00:00

## Goal

> 本地发票图片字段提取，单用户质量优先

Parsed as **vision** / **quality**, 8192 tokens, concurrency 1, privacy `local_only`.

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
| step3-vl-10b-fp8 **<-** | vllm   | FP8       | -     | 97 ms | 20.1         | 19.8            | 55.0 GB | 100.0 |

## Chosen configuration

| field                    | value                                     |
|--------------------------|-------------------------------------------|
| Model                    | step3-vl-10b-fp8                          |
| Source                   | `stepfun-ai/Step3-VL-10B-FP8`             |
| Engine                   | vllm                                      |
| Precision                | FP8                                       |
| Context                  | 8192                                      |
| Concurrency              | 1                                         |
| KV cache                 | auto                                      |
| TTFT                     | 97.13 ms                                  |
| Complete response (mean) | 3069.11 ms                                |
| Decode                   | 20.08 tok/s                               |
| Peak memory              | 55.04 GB                                  |
| Memory source            | engine_reported_weights_plus_allocated_kv |
| Quality (blended)        | 1.000                                     |
| Quality (keyword)        | -                                         |
| Quality (rubric)         | -                                         |
| Score                    | 100.00 / 100                              |
| Profile key              | `52b256d0a4915bbff292`                    |

## Acceptance policy

| constraint  | value            |
|-------------|------------------|
| min_quality | 1.0              |
| objective   | fastest_complete |

Latency limits apply to the measured mean complete response, not TTFT or a tail-latency guarantee. A single accepted candidate does not demonstrate an optimization gain. Reproduce with the same LOCALPILOT_BENCHMARK_CONFIG.

## Document field acceptance

| metric                 | value                                                            |
|------------------------|------------------------------------------------------------------|
| Documents              | 6                                                                |
| Fields correct / total | 30 / 30                                                          |
| Field exact match      | 1.000                                                            |
| Document exact match   | 1.000                                                            |
| Valid JSON rate        | 1.000                                                            |
| Request failures       | 0                                                                |
| Dataset SHA256         | f9d3edf9ba95ec21190e3c30ef8ff9244fbb2b16e818de05c09a2d2faf7c584f |

Quality uses exact JSON document matching against fixed labels. Failed requests count as incorrect. This is dataset-specific field evidence, not an independent semantic judge score.
Reproduction requires the same benchmark configuration: set `LOCALPILOT_BENCHMARK_CONFIG` to its path before running the command below. The full config hash is recorded in the raw JSON.

## Agent trace

```
[planner  ] read_hardware          success   dgx_spark: 119.7 GB unified, 273 GB/s
[planner  ] restrict_to_served     degraded  Attached to a server already running ['step3-vl-10b-fp8'], so the plan can only measure that. Dropped 3 model(s) this server cannot answer for: qwen3.8-27b-mlx, step-3.7-flash-nvfp4, nemotron-nano-12b-v2-vl-fp8. Comparing across models needs one server per model.
[planner  ] reconcile_with_server  degraded  The attached server runs one configuration (kv=auto, util=0.5, max_len=8192), so 1 candidate(s) whose knobs differ were dropped rather than measured under a label they do not match: step3-vl-10b-fp8-vllm-fp8-kvfp8 [KV cache dtype declared fp8, server runs auto]
[planner  ] adopt_server_knobs     degraded  Adopted the server's own setting rather than measuring under a label it does not match: step3-vl-10b-fp8-vllm-fp8: utilization 0.7 -> 0.5; step3-vl-10b-fp8-vllm-fp8-kvfp8: utilization 0.7 -> 0.5
[planner  ] propose_candidate      success   vLLM ranks #1 for quality; 16.7 GB of a 83.8 GB budget
[planner  ] plan_complete          success   1 candidates across 1 models and 1 engines
[judge    ] grade_fields           success   Deterministic JSON field acceptance
[bench    ] measure_candidate      success   step3-vl-10b-fp8-vllm-fp8: TTFT 97 ms, 20.1 tok/s/stream, 55.0 GB peak memory, 19.8 tok/s aggregate
[optimizer] rank_candidates        success   step3-vl-10b-fp8-vllm-fp8 wins with 100.00/100 under priority 'quality'
[memory   ] save_profile           success   Profile 52b256d0a4915bbff292 activated for vision/quality on this fingerprint
```

## Reproduce

```bash
localpilot autopilot "本地发票图片字段提取，单用户质量优先" \
    --mode vllm --no-reuse --json
```

_Exported 2026-09-21T13:49:10.629560+00:00 by `localpilot report`._
