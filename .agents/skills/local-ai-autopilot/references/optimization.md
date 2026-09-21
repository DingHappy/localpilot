# Task acceptance and controlled optimization

Use this for an explicit speed/quality comparison or configuration-selection task.
Keep the user's task, device and resource budget fixed; do not turn handoff into
an open-ended tuning platform or online request router.

## Explicit acceptance gates

Select a benchmark configuration through `LOCALPILOT_BENCHMARK_CONFIG`. The
configuration and referenced data must exist on the execution target; copying
this Skill alone does not copy the LocalPilot repo's example datasets.

The optional `acceptance` object supports:

```json
{"objective": "fastest_complete", "min_quality": 1.0}
```

This chooses the lowest mean complete-response latency among candidates passing
quality and stability gates. For the deterministic document evaluator, 1.0 means
all tested documents exactly match; it is not a universal production-accuracy
claim. For another task, first identify what its quality score actually means.

```json
{"objective": "highest_quality", "min_quality": 0.95, "max_total_latency_ms": 10000}
```

This selects the highest measured quality within the complete-response budget,
using lower completion time to break ties. Thresholds here are examples; use the
user's goals or an explicitly agreed baseline, not invented acceptance targets.
`weighted` preserves priority-based ranking while allowing explicit gates.

For structured fields, the dataset supplies `text`, `image_url`,
`expected_fields` and optionally `response_format`. Expected values belong only
to the evaluator, never in the inference payload or per-document schema enums.
Schema validity alone does not prove the extracted values correct. Missing fields,
request failures and incomplete responses must not disappear from the denominator.

Read `benchmark.raw.structured_quality` and per-sample verdicts for standard
acceptance. The paired experiment instead records `summaries` and `samples`.
Quality and latency in standard acceptance may come from separate requests;
paired measurement evaluates the response it timed.

## Bounded response-format A/B

For the existing single-stream vLLM JSON-Schema experiment, verify that the
installed module is available:

```bash
python3 -m localpilot.benchmark.paired --help
```

Then, on the inference node with user-supplied or discovered existing artifacts:

```bash
python3 -m localpilot.benchmark.paired \
  --profile-report BASELINE_RUN_JSON \
  --dataset DATASET_JSON \
  --response-format SCHEMA_JSON \
  --output NEW_COMPARISON_JSON
```

These names are file placeholders, not bundled assets. This module attaches to a
loopback vLLM endpoint, verifies hardware/model configuration against a real
baseline, alternates AB/BA, and changes only response_format. It uses bounded
repeats and output tokens, records progress, rejects failed/truncated responses,
and does not start or stop the engine. Use tmux when a remote task may outlast
the connection. Report unsupported parameters instead of assuming that every
engine or model supports a declared feature.

The module's current fixed selection rule requires every document to pass and
at least 10% lower mean completion time; it is not a general optimizer for all
user thresholds. Respect stricter user requirements as well. If this experiment
does not fit the task, explain that limit rather than substituting its gates.

## Interpretation and handoff

Name every changed axis. Model changes can support choosing a model but not a
claim that one engine parameter caused the difference. Lower output-token count
can reduce waiting time without increasing per-token decode speed. State warmup,
cache state, unique inputs, repetitions, failures and the tested quality scope.
Use previously unseen examples to check transfer beyond the tuning set.

Deliver the selected request/configuration file and engine invocation example
using [handoff.md](handoff.md). A selected format is not automatically applied to
all future requests or to LocalPilot's optional proxy. If no candidate passes,
retain the rejected run and recommend the smallest supported next change.
