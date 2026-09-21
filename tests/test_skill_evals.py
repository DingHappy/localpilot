from __future__ import annotations

import json
import unittest
from pathlib import Path


SKILL_ROOT = (
    Path(__file__).resolve().parents[1]
    / ".agents"
    / "skills"
    / "local-ai-autopilot"
)


class SkillEvaluationDatasetTests(unittest.TestCase):
    def setUp(self):
        self.cases = json.loads(
            (SKILL_ROOT / "evals" / "evals.json").read_text(encoding="utf-8")
        )

    def test_dataset_has_unique_positive_and_negative_cases(self):
        ids = [case["id"] for case in self.cases]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreaterEqual(
            sum(case["expected_skill"] == "local-ai-autopilot" for case in self.cases),
            5,
        )
        self.assertGreaterEqual(
            sum(case["expected_skill"] is None for case in self.cases), 3
        )

    def test_dataset_covers_trigger_boundaries_and_read_only_analysis(self):
        ids = {case["id"] for case in self.cases}
        required = {
            "local-ai-autopilot-pos-existing-report-read-only",
            "local-ai-autopilot-pos-english-remote-short",
            "local-ai-autopilot-pos-remote-cli-missing",
            "local-ai-autopilot-neg-install-ollama",
            "local-ai-autopilot-neg-generic-cuda-debug",
            "local-ai-autopilot-neg-amd-rocm",
            "local-ai-autopilot-neg-multinode-cluster",
        }
        self.assertTrue(required.issubset(ids))
        self.assertGreaterEqual(len(self.cases), 20)

    def test_every_case_uses_the_nvidia_tier_three_shape(self):
        required = {
            "id",
            "question",
            "expected_skill",
            "expected_script",
            "ground_truth",
            "expected_behavior",
        }
        for case in self.cases:
            self.assertEqual(required, set(case), case["id"])
            self.assertTrue(case["question"].strip(), case["id"])
            self.assertTrue(case["ground_truth"].strip(), case["id"])
            self.assertGreaterEqual(len(case["expected_behavior"]), 3, case["id"])


if __name__ == "__main__":
    unittest.main()
