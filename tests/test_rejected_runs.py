import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from localpilot.orchestrator import Orchestrator, AutopilotRejected
from localpilot.profiles.store import ProfileStore
from localpilot.reporting import export_run, render_markdown
from localpilot.api.jobs import JobRegistry, AutopilotJob
from localpilot.cli import main


class RejectedRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = ProfileStore(self.root / 'profiles')
        self.orchestrator = Orchestrator(store=self.store)
        config = json.loads((Path(__file__).resolve().parents[1] / 'config/benchmark.yaml').read_text())
        config['acceptance'] = {'objective': 'highest_quality', 'min_quality': .5, 'max_total_latency_ms': 1}
        path = self.root / 'benchmark.json'
        path.write_text(json.dumps(config))
        env = patch.dict(os.environ, {'LOCALPILOT_BENCHMARK_CONFIG': str(path)})
        env.start()
        self.addCleanup(env.stop)

    def reject(self):
        with self.assertRaises(AutopilotRejected) as caught:
            self.orchestrator.autopilot('local chat', mode='mock', reuse_profile=False)
        return caught.exception.run

    def test_rejection_keeps_evidence_and_current_state(self):
        previous = {'status': 'READY', 'profile_key': 'previous-valid-profile'}
        self.store.set_current(previous)
        run = self.reject()
        self.assertEqual(self.store.current(), previous)
        self.assertEqual(self.store.list_profiles(), [])
        self.assertIsNone(run['best_profile'])
        self.assertEqual(run['status'], 'REJECTED')
        self.assertTrue(run['candidates'])
        self.assertTrue(all(c['gate_failures'] for c in run['candidates']))
        self.assertEqual(self.store.load_run(run['run_id']), run)
        self.assertEqual(self.store.list_runs()[0]['status'], 'REJECTED')
        self.assertIn(run['run_id'], str(AutopilotRejected(run)))

    def test_export_has_no_fabricated_winner(self):
        run = self.reject()
        md, raw = export_run(run, root=self.root / 'results')
        text = md.read_text()
        self.assertIn('No accepted configuration', text)
        self.assertIn('SIMULATED', text)
        self.assertIn('complete response', text)
        self.assertNotIn('## Chosen configuration', text)
        self.assertNotIn('Quality is the keyword signal alone', text)
        self.assertEqual(json.loads(raw.read_text())['run_id'], run['run_id'])

    def test_real_target_without_measurements_is_not_labelled_measured(self):
        run = self.reject()
        run = copy.deepcopy(run)
        run['hardware']['simulated'] = False
        run['candidates'] = []
        self.assertIn('**NOT MEASURED**', render_markdown(run))

    def test_cli_json_is_nonzero_and_contains_saved_run(self):
        output = io.StringIO()
        with patch('localpilot.cli.Orchestrator', return_value=self.orchestrator), redirect_stdout(output):
            code = main(['autopilot', 'local chat', '--mode', 'mock', '--no-reuse', '--json'])
        data = json.loads(output.getvalue())
        self.assertEqual(code, 1)
        self.assertEqual(data['status'], 'REJECTED')
        self.assertEqual(self.store.load_run(data['run_id']), data)

    def test_background_job_remains_failed_with_reviewable_result(self):
        registry = JobRegistry(orchestrator_factory=lambda: self.orchestrator)
        job = AutopilotJob('local chat', 'mock', False)
        registry._run(job)
        self.assertEqual(job.status, 'failed')
        self.assertEqual(job.result['status'], 'REJECTED')
        self.assertIn('run_id', job.result)
        self.assertNotIn('traceback', job.result)
        self.assertIsNotNone(job.finished_at)
