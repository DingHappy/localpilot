import unittest

from helpers import candidate
from localpilot.executor.executor import Executor
from localpilot.runtime.mock import MockRuntime


class FailOneEngineRuntime(MockRuntime):
    def load_model(self, plan):
        if plan.engine == "trtllm":
            raise RuntimeError("injected engine failure")
        super().load_model(plan)


class UnhealthyRuntime(MockRuntime):
    def health_check(self):
        return {"healthy": False, "detail": "injected unhealthy state"}


class ExecutorIsolationTests(unittest.TestCase):
    def test_one_failure_does_not_abort_the_remaining_candidates(self):
        results = Executor(FailOneEngineRuntime).execute(
            [candidate("a", engine="trtllm"), candidate("b", engine="vllm")],
            task="coding",
        )
        self.assertEqual(results[0].status, "failed")
        self.assertIn("injected engine failure", results[0].error)
        self.assertEqual(results[1].status, "success")
        self.assertIsNotNone(results[1].benchmark)

    def test_an_unhealthy_runtime_fails_before_being_measured(self):
        """A server that answers the port but not the health check is not up.

        Measuring it anyway would record numbers from a half-started
        engine.
        """
        results = Executor(UnhealthyRuntime).execute(
            [candidate("a")], task="coding"
        )
        self.assertEqual(results[0].status, "failed")
        self.assertIsNone(results[0].benchmark)
        self.assertIn("health", results[0].error.lower())

    def test_steps_record_the_lifecycle_in_order(self):
        results = Executor(MockRuntime).execute([candidate("a")], task="coding")
        names = [step.name for step in results[0].steps]
        self.assertEqual(
            names[:5],
            [
                "environment_check",
                "model_load",
                "model_start",
                "health_check",
                "benchmark",
            ],
        )


class QualityEvaluatorTests(unittest.TestCase):
    def test_the_evaluator_overrides_the_keyword_quality(self):
        def evaluator(runtime, plan, task):
            return {
                "keyword": 1.0,
                "judge": 0.5,
                "blended": 0.7,
                "detail": "test rubric",
                "samples": [],
            }

        results = Executor(MockRuntime, quality_evaluator=evaluator).execute(
            [candidate("a")], task="coding"
        )
        benchmark = results[0].benchmark
        self.assertAlmostEqual(benchmark.quality, 0.7, places=3)
        self.assertAlmostEqual(benchmark.quality_judge, 0.5, places=3)
        self.assertAlmostEqual(benchmark.quality_keyword, 1.0, places=3)
        self.assertEqual(benchmark.raw["quality_detail"], "test rubric")

    def test_a_failing_evaluator_degrades_instead_of_failing_the_candidate(self):
        """A measured candidate with an ungraded answer is still worth having."""

        def evaluator(runtime, plan, task):
            raise RuntimeError("judge exploded")

        results = Executor(MockRuntime, quality_evaluator=evaluator).execute(
            [candidate("a")], task="coding"
        )
        self.assertEqual(results[0].status, "success")
        self.assertIsNotNone(results[0].benchmark)
        degraded = [
            step for step in results[0].steps if step.status == "degraded"
        ]
        self.assertTrue(degraded)
        self.assertIn("judge exploded", degraded[0].message)


class MockRuntimeContractTests(unittest.TestCase):
    def test_mock_refuses_a_real_candidate(self):
        """The simulation must not be able to stand in for a measurement."""
        with self.assertRaises(ValueError):
            MockRuntime().load_model(candidate("a", simulated=False))

    def test_speculative_decoding_raises_single_stream_throughput(self):
        plain = self._measure(candidate("plain", speculative=False))
        drafted = self._measure(candidate("drafted", speculative=True))
        self.assertGreater(
            drafted.throughput_tokens_s, plain.throughput_tokens_s
        )

    def test_batching_raises_aggregate_and_lowers_per_stream(self):
        single = self._measure(candidate("single", concurrency=1))
        batched = self._measure(candidate("batched", concurrency=16))
        self.assertGreater(
            batched.aggregate_throughput_tokens_s,
            single.aggregate_throughput_tokens_s,
        )
        self.assertLess(
            batched.throughput_tokens_s, single.throughput_tokens_s
        )

    def test_every_simulated_metric_is_flagged(self):
        benchmark = self._measure(candidate("a"))
        self.assertTrue(benchmark.simulated)
        self.assertIn("SIMULATED", benchmark.raw["label"])

    def _measure(self, plan):
        from localpilot.benchmark.runner import BenchmarkRunner

        runtime = MockRuntime()
        runtime.load_model(plan)
        runtime.start_model()
        try:
            return BenchmarkRunner().run(runtime, plan, "coding")
        finally:
            runtime.stop_model()


if __name__ == "__main__":
    unittest.main()
