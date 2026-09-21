# DGX Spark real vision input validation

On 2026-09-21, LocalPilot completed a real multimodal acceptance run against the
existing loopback-only vLLM service on the assigned DGX Spark node. The measured
model was `step3-vl-10b-fp8`.

## What was tested

- Input: a 183-byte inline PNG generated for this test, with red on the left,
  blue on the right, and a green square in the center.
- Acceptance rule: the response had to contain all three expected colors.
- Requests: 3 measured image requests after warmup.
- Execution: real hardware and real inference; both hardware and benchmark
  simulation flags are false.
- Network path: LocalPilot used the existing `127.0.0.1:8000` vLLM endpoint on
  the node. No external judge was configured.

## Result

The model identified the left region as red, the right region as blue, and the
center square as green. The all-fields keyword check passed.

| Metric | Result |
|---|---:|
| TTFT | 98.76 ms |
| TTFT P95 | 98.99 ms |
| Decode throughput | 20.19 tok/s per stream |
| Aggregate throughput | 19.96 tok/s |
| Peak memory | 55.04 GB |
| Image requests measured | 3 |

Peak memory came from `engine_reported_weights_plus_allocated_kv`. The run used
one candidate attached to an already running service, so it is evidence of the
image path and this configuration's behavior rather than evidence that
LocalPilot selected a better model or configuration.

## Evidence boundary

This run proves that LocalPilot preserved image content through prompt loading,
OpenAI-compatible request construction, remote execution, measurement, and
report export. It does not prove invoice OCR accuracy, document-field accuracy,
independent semantic quality, enforced network isolation, or multi-candidate
optimization. Those require a labeled document set, a stronger evaluator, and
at least two genuinely different served configurations.

Raw and generated evidence:

- [`20260921T124614-real-vision-quality.md`](20260921T124614-real-vision-quality.md)
- [`20260921T124614-real-vision-quality.json`](20260921T124614-real-vision-quality.json)
