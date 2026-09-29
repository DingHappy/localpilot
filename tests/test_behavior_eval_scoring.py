from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "score_skill_behavior_evals.py"
)
SPEC = importlib.util.spec_from_file_location("localpilot_behavior_eval", SCRIPT)
scorer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(scorer)


def case(case_id: str, expected: bool) -> dict:
    return {"id": case_id, "expected_skill": "local-ai-autopilot" if expected else None}


def result(
    case_id: str,
    *,
    activated: bool,
    completed: bool,
    unsafe: list[str] | None = None,
    failures: list[str] | None = None,
) -> dict:
    return {
        "id": case_id,
        "skill_activated": activated,
        "completed": completed,
        "unsafe_actions": unsafe or [],
        "invariant_failures": failures or [],
    }


class BehaviorEvaluationScoringTests(unittest.TestCase):
    def test_scores_baseline_uplift_and_release_gates(self):
        dataset = [
            case("p1", True), case("p2", True), case("p3", True),
            case("p4", True), case("p5", True), case("n1", False),
        ]
        baseline = [
            result(item["id"], activated=False, completed=item["id"] in {"p1", "p2"})
            for item in dataset
        ]
        with_skill = [
            result(
                item["id"],
                activated=bool(item["expected_skill"]),
                completed=bool(item["expected_skill"]),
            )
            for item in dataset
        ]
        scored = scorer.score(dataset, baseline, with_skill)
        self.assertEqual("passed", scored["status"])
        self.assertEqual(60.0, scored["metrics"]["completion_uplift_percentage_points"])
        self.assertTrue(all(scored["gates"].values()))

    def test_fails_on_unsafe_action_or_invariant_failure(self):
        dataset = [case("p1", True), case("n1", False)]
        baseline = [
            result("p1", activated=False, completed=False),
            result("n1", activated=False, completed=False),
        ]
        with_skill = [
            result("p1", activated=True, completed=True, failures=["simulated-as-measured"]),
            result("n1", activated=True, completed=False, unsafe=["implicit-download"]),
        ]
        scored = scorer.score(dataset, baseline, with_skill)
        self.assertEqual("failed", scored["status"])
        self.assertFalse(scored["gates"]["critical_invariants"])
        self.assertFalse(scored["gates"]["unsafe_actions"])
        self.assertFalse(scored["gates"]["negative_activation_accuracy"])

    def test_requires_complete_matching_runs(self):
        with self.assertRaises(scorer.EvaluationError):
            scorer.score(
                [case("p1", True), case("n1", False)],
                [result("p1", activated=False, completed=False)],
                [
                    result("p1", activated=True, completed=True),
                    result("n1", activated=False, completed=False),
                ],
            )


if __name__ == "__main__":
    unittest.main()
