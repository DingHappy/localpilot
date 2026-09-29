# Platform decision rules

## Apple Silicon

Treat system RAM as the unified model-memory pool. Use a real Metal-capable
local engine such as Ollama. Only measure models already present on the
attached server; never pull a model implicitly.

## DGX Spark and CUDA

- Capacity depends on total parameters; decode bandwidth depends on active
  parameters. A smaller dense model is not automatically faster than a larger
  sparse model.
- Decode is often bandwidth-bound at low concurrency. Batching may improve
  aggregate throughput while worsening per-stream latency.
- Speculative decoding may help low-concurrency serving but competes with
  batching for compute.
- Weight precision changes capacity and bytes read per generated token.

Use these rules to form candidates and explain why a test is useful. Do not
present them as measured outcomes.
