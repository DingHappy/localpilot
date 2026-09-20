import unittest

from helpers import candidate
from localpilot.agents.judge_agent import JudgeAgent
from localpilot.planner.policies import PolicyEngine
from localpilot.runtime.mock import MockRuntime


class RubricTests(unittest.TestCase):
    """The rubric exists because keyword matching is not a quality signal.

    "This has no race condition" contains the word "race". A keyword gate
    scores that as correct, and the wrong candidate wins.
    """

    def setUp(self):
        self.judge = JudgeAgent(PolicyEngine())

    def test_a_negating_answer_scores_far_below_a_correct_one(self):
        correct = self.judge._simulated_verdict(
            "This is a race condition. Guard the counter with a lock instead.",
            "race, lock",
        )
        negating = self.judge._simulated_verdict(
            "There is no race condition here, the code is safe.",
            "race, lock",
        )
        self.assertGreater(correct["correctness"], 0.9)
        self.assertLess(negating["correctness"], 0.2)
        self.assertGreater(correct["score"], negating["score"])

    def test_keyword_matching_alone_cannot_tell_them_apart(self):
        """The failure the rubric is there to fix, asserted directly."""
        terms = ["race", "lock"]
        correct = "This is a race condition; add a lock."
        negating = "There is no race condition here."
        self.assertTrue(any(term in correct for term in terms))
        self.assertTrue(any(term in negating for term in terms))

    def test_an_actionable_answer_outscores_a_vague_one(self):
        actionable = self.judge._simulated_verdict(
            "The handle leaks. Use a with statement so it always closes.",
            "close, with",
        )
        vague = self.judge._simulated_verdict("Something about close.", "close")
        self.assertGreater(actionable["score"], vague["score"])

    def test_a_verdict_parses_only_well_formed_json(self):
        good = self.judge._parse_verdict(
            'here you go {"correctness": 1, "completeness": 0.8, '
            '"actionability": 0.6, "note": "fine"}'
        )
        self.assertAlmostEqual(good["score"], 0.8, places=3)
        self.assertIsNone(self.judge._parse_verdict("no json at all"))
        self.assertIsNone(self.judge._parse_verdict('{"correctness": 1}'))

    def test_scores_are_clamped_into_range(self):
        verdict = self.judge._parse_verdict(
            '{"correctness": 7, "completeness": -3, "actionability": 0.5}'
        )
        self.assertEqual(verdict["correctness"], 1.0)
        self.assertEqual(verdict["completeness"], 0.0)


class BlendTests(unittest.TestCase):
    def test_the_blend_follows_the_configured_weights(self):
        judge = JudgeAgent(PolicyEngine())
        keyword_weight = judge.config["keyword_weight"]
        judge_weight = judge.config["judge_weight"]
        expected = (1.0 * keyword_weight + 0.0 * judge_weight) / (
            keyword_weight + judge_weight
        )
        self.assertAlmostEqual(judge._blend(1.0, 0.0), expected, places=6)


class IntegrationTests(unittest.TestCase):
    def test_evaluating_a_simulated_candidate_grades_and_blends(self):
        judge = JudgeAgent(PolicyEngine())
        plan = candidate("a")
        runtime = MockRuntime()
        runtime.load_model(plan)
        runtime.start_model()
        try:
            evaluation = judge.evaluate(runtime, plan, "coding")
        finally:
            runtime.stop_model()

        self.assertIsNotNone(evaluation["keyword"])
        self.assertIsNotNone(evaluation["judge"])
        self.assertIsNotNone(evaluation["blended"])
        self.assertTrue(evaluation["samples"])
        self.assertTrue(
            all("answer" in sample for sample in evaluation["samples"])
        )

    def test_a_task_with_no_prompts_falls_back_rather_than_failing(self):
        judge = JudgeAgent(PolicyEngine())
        plan = candidate("a")
        runtime = MockRuntime()
        runtime.load_model(plan)
        runtime.start_model()
        try:
            evaluation = judge.evaluate(runtime, plan, "no-such-task")
        finally:
            runtime.stop_model()
        self.assertIsNotNone(evaluation["keyword"])

    def test_no_external_judge_configured_means_no_external_call(self):
        """Absent a judge endpoint, nothing is contacted."""
        judge = JudgeAgent(PolicyEngine())
        self.assertIsNone(judge._ask_external_judge("anything"))


if __name__ == "__main__":
    unittest.main()
