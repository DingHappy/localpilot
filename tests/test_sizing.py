import unittest

from helpers import discrete_profile, model, spark_profile
from localpilot.sizing import MemoryModel, decode_roofline_tokens_s


class WeightSizingTests(unittest.TestCase):
    def test_measured_weight_size_is_preferred_over_derivation(self):
        """A published file size beats a formula.

        Quantized repos keep an unpredictable tail at higher precision, so
        params * bytes_per_param drifts from reality by tens of gigabytes
        on a large checkpoint.
        """
        spec = model(parameter_count_b=200.0, weights_gb=120.4, precision="NVFP4")
        self.assertAlmostEqual(MemoryModel().weights_gb(spec), 120.4, places=2)

    def test_derivation_is_used_when_no_size_is_recorded(self):
        spec = model(parameter_count_b=10.0, weights_gb=0.0, precision="BF16")
        derived = MemoryModel().weights_gb(spec)
        self.assertGreater(derived, 15)
        self.assertLess(derived, 22)

    def test_speculative_decoding_adds_the_draft_weights(self):
        spec = model(weights_gb=20.0, draft_source_id="test/draft")
        memory = MemoryModel()
        without = memory.weights_gb(spec)
        with_draft = memory.weights_gb(spec, speculative_decoding=True)
        self.assertAlmostEqual(with_draft - without, spec.draft_weights_gb, places=2)


class KVCacheTests(unittest.TestCase):
    def test_kv_scales_with_context_and_concurrency(self):
        spec = model(kv_bytes_per_token=100_000)
        memory = MemoryModel()
        single = memory.kv_cache_gb(spec, context_length=10_000, concurrency=1)
        batched = memory.kv_cache_gb(spec, context_length=10_000, concurrency=4)
        self.assertAlmostEqual(batched / single, 4.0, places=3)

    def test_fp8_kv_cache_halves_the_cost(self):
        spec = model(kv_bytes_per_token=100_000)
        memory = MemoryModel()
        auto = memory.kv_cache_gb(spec, 10_000, 1, "auto")
        fp8 = memory.kv_cache_gb(spec, 10_000, 1, "fp8")
        self.assertAlmostEqual(fp8 / auto, 0.5, places=3)

    def test_hybrid_state_is_charged_per_sequence_not_per_token(self):
        """Mamba blocks hold a recurrent state, constant in context length."""
        spec = model(kv_bytes_per_token=0, tasks=("chat",))
        spec.state_mb_per_sequence = 54.0
        memory = MemoryModel()
        short = memory.kv_cache_gb(spec, context_length=1_000, concurrency=1)
        long = memory.kv_cache_gb(spec, context_length=100_000, concurrency=1)
        self.assertAlmostEqual(short, long, places=6)
        self.assertAlmostEqual(
            memory.kv_cache_gb(spec, 1_000, 2) / short, 2.0, places=3
        )


class BudgetTests(unittest.TestCase):
    def test_unified_memory_budget_comes_from_system_memory(self):
        budget = MemoryModel().budget_gb(spark_profile(memory_gb=128.0))
        self.assertAlmostEqual(budget, 128.0 * 0.8, places=2)

    def test_discrete_gpu_budget_is_board_vram_not_host_ram(self):
        """Host RAM is irrelevant to a discrete card's capacity."""
        profile = discrete_profile(vram_gb=24.0)
        self.assertEqual(profile.memory["total_gb"], 256.0)
        budget = MemoryModel().budget_gb(profile)
        self.assertAlmostEqual(budget, 24.0 * 0.8, places=2)

    def test_oversized_model_is_rejected_with_the_numbers(self):
        spec = model(parameter_count_b=550.0, weights_gb=328.1)
        estimate = MemoryModel().estimate(
            spec, spark_profile(), context_length=8192
        )
        self.assertFalse(estimate.fits)
        self.assertIn("328", estimate.detail)

    def test_max_context_respects_the_declared_ceiling(self):
        spec = model(kv_bytes_per_token=7168, context_length=131072)
        headroom = MemoryModel().max_context_for(spec, spark_profile())
        self.assertLessEqual(headroom, spec.context_length)
        self.assertEqual(headroom % 1024, 0)

    def test_max_context_is_zero_when_weights_alone_overflow(self):
        spec = model(parameter_count_b=550.0, weights_gb=328.1)
        self.assertEqual(
            MemoryModel().max_context_for(spec, spark_profile()), 0
        )


class RooflineTests(unittest.TestCase):
    def test_decode_speed_tracks_active_parameters(self):
        """The core platform claim, as an assertion.

        Two models of identical total size differ in decode speed by their
        active-parameter ratio, because a token reads the active weights
        once.
        """
        sparse = decode_roofline_tokens_s(3.0, "NVFP4", 273.0)
        dense = decode_roofline_tokens_s(30.0, "NVFP4", 273.0)
        self.assertAlmostEqual(sparse / dense, 10.0, places=3)

    def test_nvfp4_decodes_faster_than_bf16_for_the_same_weights(self):
        nvfp4 = decode_roofline_tokens_s(3.0, "NVFP4", 273.0)
        bf16 = decode_roofline_tokens_s(3.0, "BF16", 273.0)
        self.assertGreater(nvfp4 / bf16, 3.0)

    def test_no_bandwidth_figure_means_no_estimate(self):
        self.assertIsNone(decode_roofline_tokens_s(3.0, "NVFP4", None))


if __name__ == "__main__":
    unittest.main()
