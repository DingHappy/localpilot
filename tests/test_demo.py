import tempfile
import unittest
from pathlib import Path

from localpilot.demo import _describe_difference, run_demo
from localpilot.orchestrator import Orchestrator
from localpilot.profiles.store import ProfileStore


class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(store=ProfileStore(Path(directory)))
            cls.demo = run_demo(mode="mock", orchestrator=orchestrator)

    def test_the_manual_path_is_described_but_never_executed(self):
        self.assertFalse(self.demo["demo_a"]["executed"])
        self.assertGreaterEqual(len(self.demo["demo_a"]["user_decisions"]), 5)

    def test_the_autopilot_path_reaches_a_ready_result(self):
        result = self.demo["demo_b"]["result"]
        self.assertEqual(result["status"], "READY")
        self.assertTrue(result["hardware"]["simulated"])

    def test_a_simulated_demo_says_so(self):
        self.assertIn("not a hardware benchmark", self.demo["disclaimer"])

    def test_findings_are_produced_from_the_run(self):
        self.assertTrue(self.demo["findings"])

    def test_every_comparison_names_its_confounders(self):
        """A ratio without its differences is not a claim.

        Crediting a gap to precision when the pair also differed in engine
        and speculative decoding attributes one cause out of three.
        """
        comparisons = [
            finding for finding in self.demo["findings"] if "tok/s (" in finding
        ]
        self.assertTrue(comparisons)
        for finding in comparisons:
            self.assertIn(" with ", finding)
            self.assertIn(" vs ", finding.split(" with ", 1)[1])


class DifferenceDescriptionTests(unittest.TestCase):
    def test_a_single_axis_difference_is_named_alone(self):
        left = {
            "precision": "NVFP4", "engine": "vllm", "concurrency": 1,
            "kv_cache_dtype": "auto", "context_length": 8192,
            "active_parameter_count_b": 3, "parameter_count_b": 30,
            "runtime_config": {"speculative": {"draft_source_id": "d"}},
        }
        right = dict(left, runtime_config={})
        description = _describe_difference(left, right)
        self.assertEqual(description, "speculative decoding on vs off")

    def test_multiple_differences_are_all_listed(self):
        left = {
            "precision": "NVFP4", "engine": "trtllm", "concurrency": 1,
            "kv_cache_dtype": "auto", "context_length": 8192,
            "active_parameter_count_b": 3, "parameter_count_b": 30,
            "runtime_config": {},
        }
        right = dict(left, precision="BF16", engine="vllm")
        description = _describe_difference(left, right)
        self.assertIn("precision NVFP4 vs BF16", description)
        self.assertIn("engine trtllm vs vllm", description)

    def test_identical_candidates_differ_on_nothing(self):
        plan = {
            "precision": "NVFP4", "engine": "vllm", "concurrency": 1,
            "kv_cache_dtype": "auto", "context_length": 8192,
            "active_parameter_count_b": 3, "parameter_count_b": 30,
            "runtime_config": {},
        }
        self.assertEqual(_describe_difference(plan, dict(plan)), "")


if __name__ == "__main__":
    unittest.main()
