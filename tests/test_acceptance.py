import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from helpers import candidate, metrics, result
from localpilot.benchmark.acceptance import validate_acceptance
from localpilot.optimizer.optimizer import Optimizer
from localpilot.orchestrator import Orchestrator, profile_requirements
from localpilot.intent.parser import parse_intent
from localpilot.profiles.store import ProfileStore
from localpilot.reporting import render_markdown


def measured(name, first, complete, quality=1):
    item = result(candidate(name), metrics(ttft_ms=first, quality=quality))
    item.benchmark.total_latency_ms = complete
    return item


class AcceptanceTests(unittest.TestCase):
    def test_complete_latency_wins_not_fast_first_token(self):
        a = measured('fast-first', 10, 12000)
        b = measured('fast-complete', 1000, 4000)
        chosen = Optimizer().choose([a, b], 'latency', {'objective': 'fastest_complete', 'min_quality': 1})
        self.assertIs(chosen, b)

    def test_quality_loss_cannot_buy_speed(self):
        a = measured('fast-wrong', 10, 100, .99)
        b = measured('correct', 100, 10000)
        chosen = Optimizer().choose([a, b], 'latency', {'objective': 'fastest_complete', 'min_quality': 1})
        self.assertIs(chosen, b)
        self.assertIsNone(a.score)
        self.assertIn('quality', a.gate_failures[0])

    def test_best_quality_within_complete_response_budget(self):
        slow = measured('too-slow', 10, 10001, 1)
        good = measured('good', 100, 9000, .95)
        fast = measured('fast', 10, 1000, .8)
        chosen = Optimizer().choose([slow, good, fast], 'quality', {'objective': 'highest_quality', 'max_total_latency_ms': 10000})
        self.assertIs(chosen, good)
        self.assertIn('complete response', slow.gate_failures[0])

    def test_failed_gate_clears_previously_assigned_score(self):
        a = measured('previous-winner', 10, 12000)
        optimizer = Optimizer()
        optimizer.choose([a], 'quality')
        self.assertIsNotNone(a.score)
        with self.assertRaisesRegex(RuntimeError, 'complete response.*exceeds gate'):
            optimizer.choose([a], 'quality', {'max_total_latency_ms': 2000})
        self.assertIsNone(a.score)

    def test_invalid_policy_rejected(self):
        for invalid in [None, [], {'min_quality': float('nan')}, {'min_quality': True},
                        {'min_quality': 1.1}, {'max_total_latency_ms': -1}, {'objective': 'fastest_complete'},
                        {'objective': 'highest_quality'}, {'min_qualty': .9}]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_acceptance(invalid)

    def test_invalid_measurements_cannot_win(self):
        for field, value in [('quality', float('nan')), ('stability', float('nan')), ('total_latency_ms', 0)]:
            a = measured('invalid', 1, 100)
            setattr(a.benchmark, field, value)
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                Optimizer().choose([a], 'quality', {'objective': 'fastest_complete', 'min_quality': 1})

    def test_policy_saved_in_report_and_invalidates_reuse(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / 'config/benchmark.yaml').read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'benchmark.json'
            path.write_text(json.dumps(config))
            with patch.dict(os.environ, {'LOCALPILOT_BENCHMARK_CONFIG': str(path)}):
                before = profile_requirements(parse_intent('local chat'))
                config['acceptance'] = {'objective': 'fastest_complete', 'min_quality': .5}
                path.write_text(json.dumps(config))
                after = profile_requirements(parse_intent('local chat'))
                self.assertNotEqual(before, after)
                run = Orchestrator(store=ProfileStore(Path(directory) / 'state')).autopilot('local chat', mode='mock', reuse_profile=False).to_dict()
            self.assertEqual(run['best_profile']['benchmark']['raw']['acceptance'], config['acceptance'])
            self.assertIn('Acceptance policy', render_markdown(run))
            self.assertTrue(run['hardware']['simulated'])
