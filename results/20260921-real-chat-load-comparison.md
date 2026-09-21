# DGX Spark chat workload comparison

This comparison records how the same attached vLLM deployment behaved when
the acceptance requirement changed from one interactive user to eight
concurrent users. It is a real workload-scaling comparison on the assigned
DGX Spark. It is not a configuration A/B test: model, engine, precision, KV
cache type, context limit, memory allocation and container remained the same.

## Controlled setup

| Axis | Both runs |
|---|---|
| Hardware | NVIDIA GB10, 119.7 GB unified memory |
| Model | `step3-vl-10b-fp8` |
| Engine | vLLM in the existing `lp-vllm` container |
| Precision | FP8 |
| Context limit | 8192 |
| KV cache | `auto` |
| GPU memory utilization | 0.5 |
| Endpoint | Existing loopback service; no new public listener |

The changed axes were the requested concurrency and ranking priority:
latency at concurrency 1 versus throughput at concurrency 8. LocalPilot did
not restart the service, download weights, or change the container.

## Measured results

| Metric | 1 user | 8 concurrent users | Change |
|---|---:|---:|---:|
| TTFT | 57.72 ms | 107.96 ms | +87.0% |
| TTFT P95 | 62.74 ms | 153.27 ms | +144.3% |
| Per-stream throughput | 20.19 tok/s | 22.24 tok/s | +10.2% |
| Aggregate throughput | 20.17 tok/s | 176.69 tok/s | 8.76x |
| Requests | 3 | 24 | 8x |
| Output tokens | 768 | 6144 | 8x |
| Engine-reported weights plus allocated KV | 55.04 GB | 55.04 GB | unchanged allocation |

Both hardware and benchmark records have `simulated: false`. Memory is the
engine-reported weight allocation plus allocated KV capacity, not a sampled
incremental peak caused by these requests. It therefore does not prove that
runtime memory use was identical under both loads.

## Interpretation

The existing deployment scaled aggregate output almost linearly to eight
streams while TTFT worsened materially. This supports two different saved
profiles for two different requirements:

- `dc517d8e44595ff512e9`: concurrency 1, latency priority
- `fcacbe300e2500e9a4a1`: concurrency 8, throughput priority

The comparison proves requirement-sensitive measurement and profile
separation. It does not yet prove that LocalPilot found a better serving
configuration, because both runs used the same server configuration. A true
optimization result requires a controlled second configuration or model.

Quality remains weak evidence in both runs. The independent judge was
unavailable and the reported `quality=1.0` came from keyword checks. One smoke
response included an inaccurate cloud-service self-description, so this score
must not be presented as verified assistant quality.

## Evidence

- [One-user latency run](20260921T122320-real-chat-latency.json)
- [Eight-user throughput run](20260921T122943-real-chat-throughput.json)
- [One-user human-readable report](20260921T122320-real-chat-latency.md)
- [Eight-user human-readable report](20260921T122943-real-chat-throughput.md)
