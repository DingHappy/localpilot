import json
import tempfile
import unittest
from pathlib import Path

from localpilot.orchestrator import Orchestrator
from localpilot.profiles.store import ProfileStore
from localpilot.reporting import comparison_notes, export_run, render_markdown


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(Path(directory))
            result = Orchestrator(store=store).autopilot(
                "帮我部署一个完全本地运行的代码审查 AI，响应速度优先", mode="mock"
            )
            cls.run_payload = result.to_dict()
        cls.markdown = render_markdown(cls.run_payload)

    def test_a_simulated_report_says_so_at_the_top(self):
        """The label has to travel with the numbers, not sit in a footnote."""
        head = self.markdown.split("## Goal")[0]
        self.assertIn("SIMULATED", head)
        self.assertIn("must not be quoted as a benchmark", head)

    def test_the_report_carries_the_machine_and_the_candidates(self):
        self.assertIn("## Machine", self.markdown)
        self.assertIn("dgx_spark", self.markdown)
        self.assertIn("## Candidates", self.markdown)
        self.assertIn("## Chosen configuration", self.markdown)
        self.assertIn("## Agent trace", self.markdown)

    def test_the_winner_is_marked_in_the_candidate_table(self):
        winner = self.run_payload["best_profile"]["candidate"]["model_id"]
        self.assertIn(f"{winner} **<-**", self.markdown)

    def test_rejections_and_their_gates_are_included(self):
        self.assertIn("Ruled out before anything ran", self.markdown)
        self.assertIn("[memory]", self.markdown)

    def test_the_report_states_how_peak_memory_was_obtained(self):
        self.assertIn("Memory source", self.markdown)

    def test_an_estimate_is_not_labelled_as_an_observed_peak(self):
        benchmark = self.run_payload["best_profile"]["benchmark"]
        source = (benchmark.get("raw") or {}).get("peak_memory_source")
        if source == "planner_estimate_no_readable_source":
            self.assertIn("Estimated memory", self.markdown)
            self.assertIn("not an observed peak", self.markdown)

    def test_a_reproduce_command_is_included(self):
        self.assertIn("## Reproduce", self.markdown)
        self.assertIn("localpilot autopilot", self.markdown)
        self.assertIn("--no-reuse", self.markdown)

    def test_a_weak_quality_signal_is_called_out(self):
        benchmark = self.run_payload["best_profile"]["benchmark"]
        if benchmark.get("quality_judge") is None:
            self.assertIn("no judge model was", self.markdown)


class ComparisonNoteTests(unittest.TestCase):
    def test_every_note_lists_the_axes_that_differed(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Orchestrator(store=ProfileStore(Path(directory))).autopilot(
                "本地代码审查 AI，速度优先", mode="mock"
            )
        notes = comparison_notes(result.to_dict()["candidates"])
        self.assertTrue(notes)
        for note in notes:
            self.assertIn("differing in", note)
            self.assertIn(" vs ", note.split("differing in", 1)[1])


class ExportTests(unittest.TestCase):
    def test_export_writes_markdown_and_json_side_by_side(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(Path(directory))
            run = Orchestrator(store=store).autopilot(
                "本地代码审查 AI，速度优先", mode="mock"
            ).to_dict()
            root = Path(directory) / "results"
            markdown_path, json_path = export_run(run, root=root)

            self.assertTrue(markdown_path.exists())
            self.assertTrue(json_path.exists())
            self.assertEqual(
                json.loads(json_path.read_text())["run_id"], run["run_id"]
            )

    def test_a_simulated_run_is_named_so_it_cannot_be_mistaken(self):
        """A filename is what someone skims in a directory listing."""
        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(Path(directory))
            run = Orchestrator(store=store).autopilot(
                "本地代码审查 AI，速度优先", mode="mock"
            ).to_dict()
            root = Path(directory) / "results"
            markdown_path, _ = export_run(run, root=root)
        self.assertIn("-sim-", markdown_path.name)
        self.assertIn("coding", markdown_path.name)
        self.assertIn("latency", markdown_path.name)


if __name__ == "__main__":
    unittest.main()
