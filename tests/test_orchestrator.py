import tempfile
import unittest
from pathlib import Path

from localpilot.orchestrator import Orchestrator
from localpilot.profiles.store import ProfileStore

GOAL = (
    "帮我部署一个完全本地运行的代码审查 AI。"
    "我的代码不能离开这台电脑。响应速度优先。"
)


class ClosedLoopTests(unittest.TestCase):
    def test_the_loop_completes_and_labels_everything_simulated(self):
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(store=ProfileStore(Path(directory)))
            result = orchestrator.autopilot(GOAL, mode="mock")

            self.assertEqual(result.status, "READY")
            self.assertFalse(result.profile_reused)
            self.assertTrue(result.best_profile.simulated)
            self.assertGreaterEqual(len(result.candidates), 2)
            self.assertTrue(
                all(item.candidate.simulated for item in result.candidates)
            )
            self.assertTrue(
                all(
                    item.benchmark is None or item.benchmark.simulated
                    for item in result.candidates
                )
            )
            self.assertTrue(
                any("SIMULATED" in warning for warning in result.warnings)
            )

    def test_the_agent_trace_shows_each_role_in_order(self):
        """Separated authority has to be visible, or it is just structure."""
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(store=ProfileStore(Path(directory)))
            result = orchestrator.autopilot(GOAL, mode="mock")

        agents = [step.agent for step in result.agent_trace]
        for name in ("planner", "bench", "judge", "optimizer", "memory"):
            self.assertIn(name, agents, agents)
        self.assertLess(agents.index("planner"), agents.index("bench"))
        self.assertLess(agents.index("bench"), agents.index("optimizer"))
        self.assertLess(agents.index("optimizer"), agents.index("memory"))

    def test_rejections_reach_the_trace_with_their_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(store=ProfileStore(Path(directory)))
            result = orchestrator.autopilot(GOAL, mode="mock")

        rejections = [
            step for step in result.agent_trace if step.action == "reject_model"
        ]
        self.assertTrue(rejections)
        self.assertIn("gate", rejections[0].data)

    def test_the_winner_is_the_highest_scoring_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(store=ProfileStore(Path(directory)))
            result = orchestrator.autopilot(GOAL, mode="mock")

        scored = [item for item in result.candidates if item.score is not None]
        best = max(scored, key=lambda item: item.score)
        self.assertEqual(
            result.best_profile.candidate.candidate_id,
            best.candidate.candidate_id,
        )


class ProfileMemoryTests(unittest.TestCase):
    def test_an_identical_request_reuses_the_stored_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(Path(directory))
            orchestrator = Orchestrator(store=store)

            first = orchestrator.autopilot(GOAL, mode="mock")
            second = orchestrator.autopilot(GOAL, mode="mock")

            self.assertFalse(first.profile_reused)
            self.assertTrue(second.profile_reused)
            self.assertEqual(
                second.best_profile.profile_key, first.best_profile.profile_key
            )
            self.assertEqual(len(second.candidates), 1)

    def test_no_reuse_forces_a_fresh_search(self):
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(store=ProfileStore(Path(directory)))
            orchestrator.autopilot(GOAL, mode="mock")
            again = orchestrator.autopilot(GOAL, mode="mock", reuse_profile=False)
        self.assertFalse(again.profile_reused)
        self.assertGreater(len(again.candidates), 1)

    def test_a_staged_search_does_not_replace_the_active_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(Path(directory))
            orchestrator = Orchestrator(store=store)
            active = orchestrator.autopilot(GOAL, mode="mock")
            staged = orchestrator.autopilot(
                "本地代码审查 AI，8 个用户，吞吐优先",
                mode="mock",
                reuse_profile=False,
                activate_profile=False,
            )
            current = store.current()
            staged_loaded = store.load(staged.best_profile.profile_key)

        self.assertEqual(current["profile_key"], active.best_profile.profile_key)
        self.assertNotEqual(
            staged.best_profile.profile_key, active.best_profile.profile_key
        )
        self.assertIsNotNone(staged_loaded)

    def test_a_different_priority_does_not_reuse_the_profile(self):
        """A profile is keyed to what was asked for, not only to the machine."""
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(store=ProfileStore(Path(directory)))
            orchestrator.autopilot(GOAL, mode="mock")
            other = orchestrator.autopilot(
                "本地代码审查 AI，质量优先", mode="mock"
            )
        self.assertFalse(other.profile_reused)
        self.assertEqual(other.intent.priority, "quality")

    def test_a_different_context_length_does_not_reuse_the_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(store=ProfileStore(Path(directory)))
            orchestrator.autopilot("本地代码审查 AI，8k 上下文，速度优先", mode="mock")
            other = orchestrator.autopilot(
                "本地代码审查 AI，128k 上下文，速度优先", mode="mock"
            )
        self.assertFalse(other.profile_reused)
        self.assertEqual(other.intent.context_length, 128 * 1024)

    def test_a_different_concurrency_does_not_reuse_the_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(store=ProfileStore(Path(directory)))
            orchestrator.autopilot("本地代码审查 AI，速度优先", mode="mock")
            other = orchestrator.autopilot(
                "本地代码审查 AI，2 个用户，速度优先", mode="mock"
            )
        self.assertFalse(other.profile_reused)
        self.assertEqual(other.intent.concurrency, 2)

    def test_a_different_modality_does_not_reuse_the_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(store=ProfileStore(Path(directory)))
            orchestrator.autopilot("本地图片理解，速度优先", mode="mock")
            other = orchestrator.autopilot(
                "本地 PDF 文档识别，速度优先", mode="mock"
            )
        self.assertFalse(other.profile_reused)
        self.assertNotEqual(other.intent.modalities, ["image", "text"])

    def test_a_different_language_does_not_reuse_the_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(store=ProfileStore(Path(directory)))
            orchestrator.autopilot("local code review AI, latency first", mode="mock")
            other = orchestrator.autopilot("本地代码审查 AI，速度优先", mode="mock")
        self.assertFalse(other.profile_reused)
        self.assertEqual(other.intent.preferred_language, ["zh", "en"])

    def test_a_saved_profile_round_trips_through_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(Path(directory))
            result = Orchestrator(store=store).autopilot(GOAL, mode="mock")
            loaded = store.load(result.best_profile.profile_key)

        self.assertIsNotNone(loaded)
        self.assertEqual(
            loaded.candidate.candidate_id,
            result.best_profile.candidate.candidate_id,
        )
        self.assertEqual(loaded.candidate.engine, result.best_profile.candidate.engine)
        self.assertIsNotNone(loaded.candidate.memory_estimate)


class ModeTests(unittest.TestCase):
    def test_a_named_engine_fails_loudly_rather_than_simulating(self):
        """Silently downgrading a real request to a simulation is worse than
        an error: the user would read modelled numbers as measurements."""
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(store=ProfileStore(Path(directory)))
            with self.assertRaises(RuntimeError):
                orchestrator.autopilot(GOAL, mode="vllm")

    def test_other_tasks_reach_a_ready_result(self):
        cases = {
            "我要一个本地智能体，能调用工具，20个人同时用": "agentic",
            "本地图片理解，帮我认发票": "vision",
            "本地 RAG 向量检索": "embedding",
        }
        for goal, expected_task in cases.items():
            with tempfile.TemporaryDirectory() as directory:
                orchestrator = Orchestrator(store=ProfileStore(Path(directory)))
                result = orchestrator.autopilot(goal, mode="mock")
            self.assertEqual(result.intent.task, expected_task, goal)
            self.assertEqual(result.status, "READY", goal)


if __name__ == "__main__":
    unittest.main()
