import tempfile
import unittest
from pathlib import Path

from localpilot.api.jobs import JobRegistry
from localpilot.orchestrator import Orchestrator
from localpilot.profiles.store import ProfileStore
from localpilot.runtime.openai_compat import _distinct_prompts

GOAL = "帮我部署一个完全本地运行的代码审查 AI，响应速度优先"


class ProgressCallbackTests(unittest.TestCase):
    """A real search takes minutes. Steps have to be observable as they land."""

    def test_steps_arrive_during_the_run_not_only_at_the_end(self):
        seen = []
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(store=ProfileStore(Path(directory)))
            result = orchestrator.autopilot(
                GOAL, mode="mock", progress=seen.append
            )

        self.assertTrue(seen)
        self.assertEqual(len(seen), len(result.agent_trace))
        self.assertEqual(
            [step.action for step in seen],
            [step.action for step in result.agent_trace],
        )

    def test_the_planner_reports_before_the_bench_agent_does(self):
        seen = []
        with tempfile.TemporaryDirectory() as directory:
            Orchestrator(store=ProfileStore(Path(directory))).autopilot(
                GOAL, mode="mock", progress=seen.append
            )
        agents = [step.agent for step in seen]
        self.assertLess(agents.index("planner"), agents.index("bench"))

    def test_a_reused_profile_also_reports_its_step(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(Path(directory))
            orchestrator = Orchestrator(store=store)
            orchestrator.autopilot(GOAL, mode="mock")

            seen = []
            result = orchestrator.autopilot(
                GOAL, mode="mock", progress=seen.append
            )

        self.assertTrue(result.profile_reused)
        self.assertEqual([step.action for step in seen], ["profile_reused"])
        self.assertEqual(
            [step.action for step in result.agent_trace], ["profile_reused"]
        )

    def test_a_broken_progress_callback_cannot_fail_the_run(self):
        """Observability is not allowed to break the thing being observed."""

        def explode(step):
            raise RuntimeError("callback exploded")

        with tempfile.TemporaryDirectory() as directory:
            result = Orchestrator(store=ProfileStore(Path(directory))).autopilot(
                GOAL, mode="mock", progress=explode
            )
        self.assertEqual(result.status, "READY")
        self.assertTrue(result.agent_trace)


class JobTraceTests(unittest.TestCase):
    def test_a_job_exposes_its_trace_alongside_the_result(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(Path(directory))
            registry = JobRegistry(
                orchestrator_factory=lambda: Orchestrator(store=store)
            )
            job = registry.submit(GOAL, mode="mock")
            for _ in range(600):
                if job.status in {"succeeded", "failed"}:
                    break
                import time

                time.sleep(0.05)

        self.assertEqual(job.status, "succeeded", job.error)
        payload = job.to_dict()
        self.assertTrue(payload["trace"])
        self.assertEqual(
            len(payload["trace"]), len(payload["result"]["agent_trace"])
        )
        self.assertIn("agent", payload["trace"][0])

    def test_the_history_listing_omits_the_heavy_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(Path(directory))
            registry = JobRegistry(
                orchestrator_factory=lambda: Orchestrator(store=store)
            )
            job = registry.submit(GOAL, mode="mock")
            summary = registry.recent(1)[0].to_dict(include_result=False)
            for _ in range(600):
                if job.status in {"succeeded", "failed"}:
                    break
                import time

                time.sleep(0.05)
        self.assertNotIn("result", summary)
        self.assertNotIn("trace", summary)


class DistinctPromptTests(unittest.TestCase):
    """Firing one identical request N times measures the prefix cache.

    With prefix caching on, every stream after the first skips prefill, so
    the aggregate figure lands far above what concurrent users would see.
    """

    def setUp(self):
        self.prompts = [
            {"text": "first prompt"},
            {"text": "second prompt"},
        ]

    def test_one_prompt_per_stream(self):
        built = _distinct_prompts(self.prompts, 4)
        self.assertEqual(len(built), 4)

    def test_the_available_prompts_are_rotated_first(self):
        built = _distinct_prompts(self.prompts, 2)
        self.assertEqual(
            [item["text"] for item in built], ["first prompt", "second prompt"]
        )

    def test_reused_prompts_are_differentiated(self):
        built = _distinct_prompts(self.prompts, 5)
        self.assertEqual(len({item["text"] for item in built}), 5, built)

    def test_a_single_stream_is_left_verbatim(self):
        self.assertEqual(_distinct_prompts(self.prompts, 1)[0]["text"], "first prompt")

    def test_an_image_is_preserved_in_each_request_variant(self):
        prompts = [{"text": "colors", "image_url": "data:image/png;base64,AAAA"}]
        built = _distinct_prompts(prompts, 2)
        self.assertTrue(all(item["image_url"].endswith("AAAA") for item in built))
        self.assertEqual(len({item["text"] for item in built}), 2)

    def test_an_empty_prompt_set_does_not_crash(self):
        self.assertEqual(len(_distinct_prompts([], 3)), 3)


if __name__ == "__main__":
    unittest.main()
