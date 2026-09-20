import tempfile
import unittest
from pathlib import Path

from localpilot.orchestrator import Orchestrator
from localpilot.profiles.store import ProfileStore


GOAL = (
    "帮我部署一个完全本地运行的代码审查 AI。"
    "我的代码不能离开这台电脑。响应速度优先。"
)


class OrchestratorTests(unittest.TestCase):
    def test_mock_closed_loop_and_profile_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(Path(directory))
            orchestrator = Orchestrator(store=store)
            first = orchestrator.autopilot(GOAL, mode="mock")
            self.assertEqual(first.status, "READY")
            self.assertFalse(first.profile_reused)
            self.assertTrue(first.best_profile.simulated)
            self.assertGreaterEqual(len(first.candidates), 2)
            self.assertTrue(all(item.candidate.simulated for item in first.candidates))
            self.assertTrue(
                all(
                    item.benchmark is None or item.benchmark.simulated
                    for item in first.candidates
                )
            )

            second = orchestrator.autopilot(GOAL, mode="mock")
            self.assertTrue(second.profile_reused)
            self.assertEqual(
                second.best_profile.profile_key,
                first.best_profile.profile_key,
            )

    def test_openvino_mode_fails_without_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(store=ProfileStore(Path(directory)))
            with self.assertRaises(RuntimeError):
                orchestrator.autopilot(GOAL, mode="openvino")

    def test_mock_chat_path_is_runnable(self):
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(store=ProfileStore(Path(directory)))
            result = orchestrator.autopilot(
                "I want a completely local chat assistant.",
                mode="mock",
            )
            self.assertEqual(result.intent.task, "chat")
            self.assertEqual(result.status, "READY")
            self.assertTrue(result.best_profile.simulated)


if __name__ == "__main__":
    unittest.main()
