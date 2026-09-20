import unittest

from helpers import model, spark_profile
from localpilot.models.registry import ModelRegistry
from localpilot.models.selector import ModelSelector
from localpilot.planner.planner import Planner
from localpilot.schemas import Intent


def intent(**overrides) -> Intent:
    defaults = {
        "task": "coding",
        "priority": "latency",
        "context_length": 8192,
        "concurrency": 1,
    }
    defaults.update(overrides)
    return Intent(**defaults)


class SelectorTests(unittest.TestCase):
    def test_oversized_model_is_rejected_and_the_reason_is_kept(self):
        """A rejection is information, not noise.

        "This needs 328 GB and you have 128" tells the user something a
        silently shortened list does not.
        """
        big = model("too-big", parameter_count_b=550.0, weights_gb=328.1)
        small = model("fits", weights_gb=20.0)
        outcome = ModelSelector().select(
            intent(), spark_profile(), [big, small], ["vllm"]
        )
        self.assertEqual([m.model_id for m in outcome.selected], ["fits"])
        self.assertEqual(len(outcome.rejected), 1)
        self.assertEqual(outcome.rejected[0].gate, "memory")
        self.assertIn("328", outcome.rejected[0].reason)

    def test_uninstalled_engine_rejects_a_model(self):
        spec = model("trt-only", engines=("trtllm",))
        outcome = ModelSelector().select(
            intent(), spark_profile(), [spec], ["vllm"]
        )
        self.assertFalse(outcome.selected)
        self.assertEqual(outcome.rejected[0].gate, "engine")

    def test_context_shorter_than_requested_is_rejected(self):
        spec = model("short", context_length=4096)
        outcome = ModelSelector().select(
            intent(context_length=131072), spark_profile(), [spec], ["vllm"]
        )
        self.assertEqual(outcome.rejected[0].gate, "context")

    def test_missing_modality_is_rejected(self):
        spec = model("text-only")
        outcome = ModelSelector().select(
            intent(task="chat", modalities=["text", "image"]),
            spark_profile(),
            [spec],
            ["vllm"],
        )
        self.assertEqual(outcome.rejected[0].gate, "modality")

    def test_latency_priority_prefers_the_sparser_model(self):
        dense = model("dense-9b", parameter_count_b=9.0,
                      active_parameter_count_b=9.0, weights_gb=9.6)
        sparse = model("sparse-30b", parameter_count_b=30.0,
                       active_parameter_count_b=3.0, weights_gb=20.0)
        outcome = ModelSelector().select(
            intent(priority="latency"), spark_profile(),
            [dense, sparse], ["vllm"],
        )
        self.assertEqual(outcome.selected[0].model_id, "sparse-30b")


class CandidateGenerationTests(unittest.TestCase):
    def setUp(self):
        self.hardware = spark_profile()
        self.planner = Planner()
        self.models = ModelRegistry().for_task("coding")

    def test_candidates_cover_distinct_models_before_knob_variants(self):
        """Spending the whole budget on one model answers a narrower question.

        Four near-identical configurations of the same checkpoint measure
        the knobs but never tell the user whether a different model would
        have been better.
        """
        candidates = self.planner.plan(
            intent(), self.hardware, self.models, "mock"
        )
        distinct = {item.model_id for item in candidates}
        self.assertGreaterEqual(len(distinct), 2)
        self.assertLessEqual(len(candidates), self.planner.policies.max_candidates)

    def test_engine_preference_follows_the_priority_policy(self):
        latency = self.planner.plan(
            intent(priority="latency"), self.hardware, self.models, "mock"
        )
        quality = self.planner.plan(
            intent(priority="quality"), self.hardware, self.models, "mock"
        )
        self.assertEqual(latency[0].engine, "trtllm")
        self.assertEqual(quality[0].engine, "vllm")

    def test_speculative_decoding_is_dropped_under_heavy_batching(self):
        """Drafting needs spare compute that batching has already claimed."""
        single = self.planner.plan(
            intent(concurrency=1), self.hardware, self.models, "mock"
        )
        batched = self.planner.plan(
            intent(priority="throughput", concurrency=32),
            self.hardware,
            self.models,
            "mock",
        )
        self.assertTrue(
            any(item.runtime_config.get("speculative") for item in single)
        )
        self.assertFalse(
            any(item.runtime_config.get("speculative") for item in batched)
        )

    def test_every_candidate_carries_a_memory_estimate_that_fits(self):
        for candidate in self.planner.plan(
            intent(), self.hardware, self.models, "mock"
        ):
            self.assertIsNotNone(candidate.memory_estimate)
            self.assertTrue(candidate.memory_estimate.fits)
            self.assertLessEqual(
                candidate.expected_memory_gb,
                candidate.memory_estimate.budget_gb,
            )

    def test_candidate_ids_are_unique(self):
        candidates = self.planner.plan(
            intent(), self.hardware, self.models, "mock"
        )
        ids = [item.candidate_id for item in candidates]
        self.assertEqual(len(ids), len(set(ids)))

    def test_no_engine_available_is_an_explicit_failure(self):
        hardware = spark_profile()
        hardware.stack = dict(hardware.stack)
        hardware.stack["engines_available"] = []
        with self.assertRaises(RuntimeError) as caught:
            self.planner.plan(intent(), hardware, self.models, "vllm")
        self.assertIn("engine", str(caught.exception).lower())


class RecoveryTests(unittest.TestCase):
    def test_recovery_shrinks_the_configuration_then_switches_model(self):
        planner = Planner()
        hardware = spark_profile()
        models = ModelRegistry().for_task("coding")
        primary = planner.plan(intent(context_length=131072), hardware, models, "mock")[0]

        recovery = planner.recovery_plan([primary], hardware, models)

        self.assertTrue(recovery)
        self.assertLessEqual(len(recovery), planner.policies.recovery["max_attempts"])
        self.assertTrue(all(item.fallback_of == primary.candidate_id for item in recovery))
        self.assertTrue(all(item.recovery_action for item in recovery))
        self.assertLess(recovery[0].context_length, primary.context_length)
        self.assertEqual(recovery[0].kv_cache_dtype, "fp8")

    def test_recovery_of_nothing_is_nothing(self):
        self.assertEqual(
            Planner().recovery_plan([], spark_profile(), []), []
        )


if __name__ == "__main__":
    unittest.main()
