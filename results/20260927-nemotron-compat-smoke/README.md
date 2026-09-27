# Nemotron 30B GB10 compatibility smoke test — 2026-09-27

This is a **failed compatibility precheck**, not an inference benchmark. No
latency, throughput, quality, or speculative-decoding result was produced.

The assigned GB10 node has a local 21 GB directory for
`nemotron-30b-a3b-nvfp4`. Its `config.json` declares `model_type: nemotron_h`,
`architectures: [NemotronHForCausalLM]`, and one next-token prediction layer.
The model card's DGX Spark vLLM recipe names vLLM `0.27.1`; the cached,
currently serving NVIDIA image reports vLLM `0.15.1+befbc472`.

I attempted only a bounded baseline load in a separate container with no
network, read-only root and weights, a 40 GiB memory limit, 12 CPUs, and an
unpublished loopback endpoint. It exited after four seconds with code `1`:

> The checkpoint you are trying to load has model type `nemotron_h` but Transformers does not recognize this architecture.

The application failed during model configuration, before loading weights or
serving a request. The existing `lp-vllm` and `node-exporter` containers were
running before and after the smoke test. Node disk use was 41%, leaving about
2.1 TB free. No host driver, system package, firewall, SSH configuration, or
running service was changed.

The proposed MTP on/off comparison is **not yet executable with this cached
image**. Next, validate a model-card-compatible vLLM/Transformers image in an
isolated container, then run baseline and MTP under the same model, prompts,
context, output budget, warmup, concurrency, and quality checks. Do not fill
the essay's speculative-decoding speedup from registry simulations.

Evidence: [preflight](preflight.log), [server log](server.log),
[result](result.txt), [exit code](docker-run-exit-code.txt), and
[services after](services-after.txt). The image ID and model config/index hashes
are in the preflight log. This result does not verify every weight shard hash.
