import unittest

from helpers import candidate, metrics, result
from localpilot.optimizer.optimizer import Optimizer
from localpilot.planner.policies import PolicyEngine
from localpilot.planner.scoring import score_results


class GateTests(unittest.TestCase):
    def test_a_fast_but_wrong_candidate_is_not_scored(self):
        """Speed cannot buy its way past the quality gate."""
        fast_bad = result(
            candidate("fast-bad"),
            metrics(ttft_ms=50, throughput_tokens_s=300, quality=0.2),
        )
        slower_good = result(
            candidate("slower-good"),
            metrics(ttft_ms=300, throughput_tokens_s=30, quality=0.85),
        )
        scored = score_results([fast_bad, slower_good], "latency", PolicyEngine())
        self.assertIsNone(scored[0].score)
        self.assertIsNotNone(scored[1].score)

    def test_a_gated_candidate_says_which_gate_it_failed(self):
        flaky = result(
            candidate("flaky"), metrics(quality=0.85, stability=0.4)
        )
        good = result(candidate("good"), metrics(quality=0.85, stability=1.0))
        score_results([flaky, good], "balanced", PolicyEngine())
        self.assertTrue(flaky.gate_failures)
        self.assertIn("stability", flaky.gate_failures[0])
        self.assertFalse(good.gate_failures)


class WeightingTests(unittest.TestCase):
    def test_priority_changes_the_winner_on_identical_measurements(self):
        """The same evidence must produce different answers per priority.

        If it did not, the priority policy would be decoration.
        """
        fast = result(
            candidate("fast-small"),
            metrics(ttft_ms=60, throughput_tokens_s=200, quality=0.60,
                    peak_memory_gb=20),
        )
        strong = result(
            candidate("slow-strong"),
            metrics(ttft_ms=200, throughput_tokens_s=30, quality=0.95,
                    peak_memory_gb=80),
        )
        policies = PolicyEngine()

        by_latency = Optimizer(policies).choose([fast, strong], "latency")
        by_quality = Optimizer(policies).choose([fast, strong], "quality")

        self.assertEqual(by_latency.candidate.candidate_id, "fast-small")
        self.assertEqual(by_quality.candidate.candidate_id, "slow-strong")

    def test_throughput_priority_ranks_on_aggregate_not_per_stream(self):
        """Batch serving cares about total output, not one stream's speed."""
        single = result(
            candidate("single", concurrency=1),
            metrics(throughput_tokens_s=120, aggregate=120, concurrency=1),
        )
        batched = result(
            candidate("batched", concurrency=32),
            metrics(throughput_tokens_s=25, aggregate=800, concurrency=32),
        )
        winner = Optimizer(PolicyEngine()).choose([single, batched], "throughput")
        self.assertEqual(winner.candidate.candidate_id, "batched")

    def test_score_components_are_recorded_for_every_dimension(self):
        results = [
            result(candidate("a"), metrics(ttft_ms=100, throughput_tokens_s=100)),
            result(candidate("b"), metrics(ttft_ms=200, throughput_tokens_s=50)),
        ]
        score_results(results, "balanced", PolicyEngine())
        for item in results:
            self.assertEqual(
                set(item.score_components),
                {"quality", "latency", "throughput", "memory", "stability"},
            )


class OptimizerTests(unittest.TestCase):
    def test_all_candidates_gated_raises_with_the_detail(self):
        gated = result(candidate("bad"), metrics(quality=0.1))
        with self.assertRaises(RuntimeError) as caught:
            Optimizer(PolicyEngine()).choose([gated], "balanced")
        self.assertIn("gate", str(caught.exception).lower())


if __name__ == "__main__":
    unittest.main()
