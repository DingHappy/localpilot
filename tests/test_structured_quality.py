import json
import unittest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from localpilot.benchmark.structured import score_fields, summarize_fields
from localpilot.agents.judge_agent import JudgeAgent
from localpilot.planner.policies import PolicyEngine
from localpilot.intent.parser import parse_intent
from localpilot.orchestrator import profile_requirements
from localpilot.runtime.openai_compat import OpenAICompatRuntime
from localpilot.orchestrator import Orchestrator
from localpilot.profiles.store import ProfileStore
from localpilot.reporting import render_markdown


class StructuredQualityTests(unittest.TestCase):
    def setUp(self):
        self.expected = {'number': '0012', 'total': '73.14', 'po': None}

    def test_fields_are_bound_to_keys_and_require_exact_types(self):
        good = score_fields(json.dumps(self.expected), self.expected)
        self.assertTrue(good['document_exact'])
        for bad in [
            {'number': '73.14', 'total': '0012', 'po': None},
            {'number': 12, 'total': 73.14, 'po': None},
            {'number': '0012', 'total': '73.14'},
        ]:
            self.assertFalse(score_fields(json.dumps(bad), self.expected)['document_exact'])

    def test_final_json_only_and_duplicate_keys_rejected(self):
        valid = json.dumps(self.expected)
        self.assertTrue(score_fields('<think>ignore</think>```json\n' + valid + '\n```', self.expected)['document_exact'])
        for bad in ['Here is ' + valid, valid[:-1], valid + valid, '[]', '{"number":"bad","number":"0012","total":"73.14","po":null}']:
            self.assertFalse(score_fields(bad, self.expected)['json_valid'])

    def test_extra_fields_fail_document_acceptance(self):
        got = score_fields(json.dumps(dict(self.expected, extra='x')), self.expected)
        self.assertEqual(got['fields_correct'], 3)
        self.assertFalse(got['document_exact'])

    def test_failed_requests_remain_in_denominator_and_answers_not_recorded(self):
        prompts = [{'id': str(i), 'expected_fields': self.expected} for i in range(2)]
        runtime = Mock()
        runtime.generate.side_effect = [json.dumps(self.expected), TimeoutError('private text')]
        judge = JudgeAgent(PolicyEngine())
        with patch('localpilot.agents.judge_agent.prompts_for_task', return_value=prompts):
            result = judge.evaluate(runtime, Mock(), 'vision')
        self.assertEqual(result['blended'], .5)
        self.assertEqual(result['structured']['fields_total'], 6)
        self.assertEqual(result['structured']['request_failures'], 1)
        self.assertIsNone(result['judge'])
        self.assertNotIn('private text', json.dumps(result))
        self.assertTrue(all('answer' not in s for s in result['samples']))

    def test_keyword_failures_also_count_as_wrong(self):
        prompts = [{'id': str(i), 'text': 'question', 'expected_terms': ['ok']} for i in range(2)]
        runtime = Mock()
        runtime.generate.side_effect = ['ok', TimeoutError()]
        judge = JudgeAgent(PolicyEngine())
        judge.config = dict(judge.config, enabled=False)
        with patch('localpilot.agents.judge_agent.prompts_for_task', return_value=prompts):
            result = judge.evaluate(runtime, Mock(), 'chat')
        self.assertEqual(result['keyword'], .5)

    def test_empty_summary_does_not_pass(self):
        self.assertEqual(summarize_fields([])['document_accuracy'], 0)

    def test_dataset_change_invalidates_reuse(self):
        intent = parse_intent('local invoice vision quality')
        with patch('localpilot.orchestrator.load_benchmark_config', return_value={'prompts': ['a']}):
            a = profile_requirements(intent)
        with patch('localpilot.orchestrator.load_benchmark_config', return_value={'prompts': ['b']}):
            b = profile_requirements(intent)
        self.assertNotEqual(a['benchmark_sha256'], b['benchmark_sha256'])

    def test_gold_labels_never_enter_inference_payload(self):
        runtime = OpenAICompatRuntime()
        runtime.engine_id = 'vllm'
        prompt = {'text': 'extract fields', 'image_url': 'data:image/png;base64,AAAA',
                  'expected_fields': {'secret_gold': 'gold_value'}, 'mock_answer': 'gold_value'}
        payload = runtime._payload(prompt, 768, False)
        encoded = json.dumps(payload)
        self.assertNotIn('gold_value', encoded)
        self.assertNotIn('expected_fields', encoded)
        self.assertIn('data:image/png;base64,AAAA', encoded)

    def test_fixture_runs_end_to_end_in_simulation(self):
        path = Path(__file__).resolve().parents[1] / 'evals/documents/benchmark.json'
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'LOCALPILOT_BENCHMARK_CONFIG': str(path)}):
            result = Orchestrator(store=ProfileStore(Path(directory))).autopilot(
                '本地发票图片字段提取，单用户质量优先', mode='mock', reuse_profile=False).to_dict()
        benchmark = result['best_profile']['benchmark']
        quality = benchmark['raw']['structured_quality']
        self.assertEqual(quality['fields_correct'], 30)
        self.assertEqual(quality['document_accuracy'], 1)
        self.assertTrue(benchmark['simulated'])
        self.assertIsNone(benchmark['quality_keyword'])
        report = render_markdown(result)
        self.assertIn('Document field acceptance', report)
        self.assertNotIn('keyword signal alone', report)
