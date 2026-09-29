# Evidence interpretation

Use these rules when reading, comparing, or publishing LocalPilot results.

## Evidence levels

- **Measured:** both hardware and benchmark `simulated` fields are false and
  the named target executed the run.
- **Estimated:** a planner, registry, model metadata, or roofline calculation
  produced the value before execution.
- **Simulated:** mock mode produced the value. It proves workflow behavior, not
  target-device performance.

`READY` alone does not make a result measured.

## Metrics

- `score` is normalized among candidates in one run on one machine. Do not
  compare it across runs or machines.
- `gate_failures` explains why a candidate has no score.
- `memory_estimate` is pre-flight evidence.
- `benchmark.peak_memory_gb` is observed only when
  `benchmark.raw.peak_memory_source` identifies an engine or sampled pool. If
  the source is `planner_estimate_no_readable_source`, the value is estimated.
- For concurrency above one, distinguish per-stream throughput from aggregate
  throughput.
- A null `quality_judge` with no `raw.structured_quality` leaves only weak
  keyword evidence. Deterministic field acceptance instead records exact field
  and document matches against a labeled dataset. State its sample count and
  scope; passing a fixed dataset does not establish general semantic quality.
- Quality and latency should describe the same responses in a controlled pair.
  Count failed or truncated responses; do not count repeated images as new inputs.
- Distinguish first-token latency from mean complete response and from P95.
  A `max_total_latency_ms` gate uses the measured mean, not a tail guarantee.
- Benchmark and acceptance-policy hashes identify the evidence used for reuse.
  Changing either requires fresh acceptance; `READY` alone is not evidence of
  continuous serving or that request options were applied by an application.
- A recovery candidate records `fallback_of` and `recovery_action`.

## Comparisons

Name every axis that differs between candidates: model, engine, precision, KV
cache dtype, context, concurrency, batching, speculative decoding, prompts,
and target. A causal claim requires a controlled pair that changes only the
axis being tested.

Reports must retain the real or simulated label and the profile/run identity
needed to trace the result back to its raw JSON.
