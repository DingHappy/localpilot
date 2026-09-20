# LocalPilot run — coding / latency

**SIMULATED** · run `fefd7afb` · 2026-09-20T07:45:07.637703+00:00

> These numbers come from a roofline model, not from this machine. They describe what the search space implies, not what the hardware did, and must not be quoted as a benchmark.

## Goal

> 帮我部署一个完全本地运行的代码审查 AI，代码不能离开这台电脑，响应速度优先

Parsed as **coding** / **latency**, 8192 tokens, concurrency 1, privacy `local_only`.

## Machine

| field                | value                                             |
|----------------------|---------------------------------------------------|
| Platform             | dgx_spark                                         |
| Accelerator          | NVIDIA DGX Spark (GB10)                           |
| Memory               | 128.0 GB unified                                  |
| Bandwidth            | 273 GB/s                                          |
| Driver               | -                                                 |
| Engines found        | llamacpp, nim, sglang, transformers, trtllm, vllm |
| Real execution ready | False                                             |

## Candidates

| model                                       | engine | precision | flags | ttft  | tok/s/stream | tok/s aggregate | peak memory | score |
|---------------------------------------------|--------|-----------|-------|-------|--------------|-----------------|-------------|-------|
| nemotron-3.5-lightning-30b-a3b-nvfp4 **<-** | trtllm | NVFP4     | spec  | 63 ms | 215.4        | 217.8           | 23.7 GB     | 96.7  |
| nemotron-3-nano-30b-a3b-nvfp4               | trtllm | NVFP4     | -     | 62 ms | 121.8        | 121.0           | 20.4 GB     | 84.8  |
| nemotron-3-super-120b-a12b-nvfp4            | trtllm | NVFP4     | -     | 63 ms | 30.2         | 30.2            | 82.6 GB     | 58.6  |
| nemotron-3.5-lightning-30b-a3b-bf16         | vllm   | BF16      | -     | 76 ms | 31.5         | 31.6            | 68.6 GB     | 27.5  |

## Comparisons

Each ratio lists every axis on which the pair differed. A gap between configurations that differ in three things is not evidence about any one of them.

- `nemotron-3.5-lightning-30b-a3b-nvfp4-trtllm-nvfp4-spec` reached 215 tok/s against 122 (1.8x), differing in speculative decoding on vs off
- `nemotron-3-nano-30b-a3b-nvfp4-trtllm-nvfp4` reached 122 tok/s against 30 (4.0x), differing in active parameters 3 vs 12, total parameters 30 vs 120
- `nemotron-3-nano-30b-a3b-nvfp4-trtllm-nvfp4` reached 122 tok/s against 31 (3.9x), differing in precision NVFP4 vs BF16, engine trtllm vs vllm
- `nemotron-3.5-lightning-30b-a3b-nvfp4-trtllm-nvfp4-spec` reached 215 tok/s against 30 (7.1x), differing in speculative decoding on vs off, active parameters 3 vs 12, total parameters 30 vs 120

## Ruled out before anything ran

- **nemotron-3-ultra-550b-a55b-nvfp4** [memory] needs 348.0 GB but only 102.4 GB is safely available (328.1 GB weights + 0.2 GB KV at 8192 tokens x1)

## Chosen configuration

| field              | value                                                       |
|--------------------|-------------------------------------------------------------|
| Model              | nemotron-3.5-lightning-30b-a3b-nvfp4                        |
| Source             | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`        |
| Engine             | trtllm                                                      |
| Precision          | NVFP4                                                       |
| Context            | 8192                                                        |
| Concurrency        | 1                                                           |
| KV cache           | auto                                                        |
| Speculative draft  | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark` |
| TTFT               | 63.21 ms                                                    |
| Decode             | 215.42 tok/s                                                |
| Peak memory        | 23.73 GB                                                    |
| Peak memory source | -                                                           |
| Quality (blended)  | 0.911                                                       |
| Quality (keyword)  | 1.000                                                       |
| Quality (rubric)   | 0.851                                                       |
| Score              | 96.67 / 100                                                 |
| Profile key        | `2d6e0bc8f27848f9e685`                                      |

## Agent trace

```
[planner  ] read_hardware          success   dgx_spark: 128.0 GB unified, 273 GB/s
[planner  ] reject_model           rejected  nemotron-3-ultra-550b-a55b-nvfp4: needs 348.0 GB but only 102.4 GB is safely available (328.1 GB weights + 0.2 GB KV at 8192 tokens x1)
[planner  ] propose_candidate      success   TensorRT-LLM ranks #1 for latency; 3B of 30B parameters are read per token; speculative decoding trades compute for bandwidth; 23.0 GB of a 102.4 GB budget
[planner  ] propose_candidate      success   TensorRT-LLM ranks #1 for latency; 3B of 30B parameters are read per token; 19.6 GB of a 102.4 GB budget
[planner  ] propose_candidate      success   TensorRT-LLM ranks #1 for latency; 12B of 120B parameters are read per token; 79.5 GB of a 102.4 GB budget
[planner  ] propose_candidate      success   vLLM ranks #2 for latency; 3B of 30B parameters are read per token; 65.1 GB of a 102.4 GB budget
[planner  ] plan_complete          success   4 candidates across 4 models and 2 engines
[judge    ] grade_candidate        success   nemotron-3.5-lightning-30b-a3b-nvfp4-trtllm-nvfp4-spec: keyword 1.00, rubric 0.85, blended 0.91
[judge    ] grade_candidate        success   nemotron-3-nano-30b-a3b-nvfp4-trtllm-nvfp4: keyword 1.00, rubric 0.85, blended 0.91
[judge    ] grade_candidate        success   nemotron-3-super-120b-a12b-nvfp4-trtllm-nvfp4: keyword 1.00, rubric 0.85, blended 0.91
[judge    ] grade_candidate        success   nemotron-3.5-lightning-30b-a3b-bf16-vllm-bf16: keyword 1.00, rubric 0.85, blended 0.91
[bench    ] measure_candidate      success   nemotron-3.5-lightning-30b-a3b-nvfp4-trtllm-nvfp4-spec: TTFT 63 ms, 215.4 tok/s/stream, 23.7 GB peak, 217.8 tok/s aggregate
[bench    ] measure_candidate      success   nemotron-3-nano-30b-a3b-nvfp4-trtllm-nvfp4: TTFT 62 ms, 121.8 tok/s/stream, 20.4 GB peak, 121.0 tok/s aggregate
[bench    ] measure_candidate      success   nemotron-3-super-120b-a12b-nvfp4-trtllm-nvfp4: TTFT 63 ms, 30.2 tok/s/stream, 82.6 GB peak, 30.2 tok/s aggregate
[bench    ] measure_candidate      success   nemotron-3.5-lightning-30b-a3b-bf16-vllm-bf16: TTFT 76 ms, 31.5 tok/s/stream, 68.6 GB peak, 31.6 tok/s aggregate
[optimizer] rank_candidates        success   nemotron-3.5-lightning-30b-a3b-nvfp4-trtllm-nvfp4-spec wins with 96.67/100 under priority 'latency'
[memory   ] save_profile           success   Profile 2d6e0bc8f27848f9e685 stored for coding/latency on this fingerprint
```

## Reproduce

```bash
localpilot autopilot "帮我部署一个完全本地运行的代码审查 AI，代码不能离开这台电脑，响应速度优先" \
    --mode mock --no-reuse --json
```

_Exported 2026-09-20T07:45:07.741918+00:00 by `localpilot report`._
