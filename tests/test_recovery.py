import tempfile
import unittest
from pathlib import Path

from localpilot.orchestrator import Orchestrator
from localpilot.profiles.store import ProfileStore
from localpilot.runtime.mock import MockRuntime


class FailLargeContextRuntime(MockRuntime):
    """Refuses anything above a small context, as a cache allocation would."""

    def load_model(self, plan):
        if plan.context_length > 8192:
            raise RuntimeError("injected KV cache allocation failure")
        super().load_model(plan)


class FailEverythingRuntime(MockRuntime):
    def load_model(self, plan):
        raise RuntimeError("injected unconditional failure")


class RecoveryTests(unittest.TestCase):
    def test_recovery_runs_after_every_primary_candidate_fails(self):
        def resolver(name):
            return FailLargeContextRuntime

        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(
                store=ProfileStore(Path(directory)), runtime_resolver=resolver
            )
            result = orchestrator.autopilot(
                "本地长上下文代码助手，128k 上下文",
                mode="mock",
                reuse_profile=False,
            )

        self.assertEqual(result.status, "READY")
        self.assertTrue(
            any(item.candidate.fallback_of for item in result.candidates)
        )
        recovery_steps = [
            step for step in result.agent_trace if step.action == "recovery_started"
        ]
        self.assertTrue(recovery_steps)
        self.assertEqual(recovery_steps[0].status, "degraded")
        self.assertLessEqual(result.best_profile.candidate.context_length, 8192)

    def test_recovery_is_bounded_and_then_gives_up(self):
        """Failure has to terminate. An unbounded retry loop on a machine
        that cannot serve the model is worse than a clear error."""

        def resolver(name):
            return FailEverythingRuntime

        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(
                store=ProfileStore(Path(directory)), runtime_resolver=resolver
            )
            with self.assertRaises(RuntimeError) as caught:
                orchestrator.autopilot(
                    "本地代码审查 AI，速度优先", mode="mock", reuse_profile=False
                )
        self.assertIn("injected unconditional failure", str(caught.exception))

    def test_a_stale_profile_is_rejected_and_the_search_reruns(self):
        """A remembered configuration that no longer starts must not be served."""
        state = {"fail": False}

        class SometimesRuntime(MockRuntime):
            def load_model(self, plan):
                if state["fail"]:
                    raise RuntimeError("injected post-hoc failure")
                super().load_model(plan)

        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(Path(directory))
            orchestrator = Orchestrator(
                store=store, runtime_resolver=lambda name: SometimesRuntime
            )
            first = orchestrator.autopilot("本地代码审查，速度优先", mode="mock")
            self.assertFalse(first.profile_reused)

            state["fail"] = True
            with self.assertRaises(RuntimeError):
                orchestrator.autopilot("本地代码审查，速度优先", mode="mock")


if __name__ == "__main__":
    unittest.main()
