"""The knobs LocalPilot searches on a CUDA target.

Kept as documentation of the axes, not as a live default: the real values
come from ``config/policies.yaml`` so a run can be reproduced from config
alone.
"""

SEARCH_AXES = {
    "engine": ["vllm", "trtllm", "sglang", "nim"],
    "weight_precision": ["NVFP4", "FP8", "BF16"],
    "kv_cache_dtype": ["auto", "fp8"],
    "context_length": [8192, 32768, 131072, 1048576],
    "max_num_seqs": [1, 8, 32],
    "speculative_decoding": [False, True],
    "prefix_caching": [False, True],
}

# Why these axes and not device selection: a DGX Spark has one accelerator
# drawing on one memory pool. Nothing is gained by asking which device to
# use. What decides the outcome is how the engine is configured against a
# bandwidth ceiling, and that is what these axes cover.
AXIS_NOTES = {
    "engine": "Different schedulers and kernels for identical weights.",
    "weight_precision": (
        "Sets both capacity and decode speed, because decode reads every "
        "active weight once per token."
    ),
    "kv_cache_dtype": "Trades cache precision for context length.",
    "context_length": "The dominant memory term once weights are placed.",
    "max_num_seqs": (
        "Batching is the only regime where a weight read is amortized across "
        "sequences, so it converts a bandwidth limit into a compute limit."
    ),
    "speculative_decoding": (
        "Spends idle compute to cut the number of memory passes. The biggest "
        "single-stream win on a bandwidth-bound machine, and worthless once "
        "batching has claimed that compute."
    ),
    "prefix_caching": "Removes repeated prefill for shared system prompts.",
}
