import io
import json
import unittest
from unittest.mock import patch

from localpilot.benchmark.paired import run_pairs, run_holdout
from localpilot.runtime.openai_compat import StreamSample, VLLMRuntime
from helpers import candidate

FORMAT = {'type': 'json_schema', 'json_schema': {'name': 'fields', 'schema': {'type': 'object', 'properties': {'code': {'type': 'string'}}, 'required': ['code'], 'additionalProperties': False}}}
PROMPTS = [{'id': str(i), 'text': 'extract code', 'image_url': 'data:image/png;base64,AAAA', 'expected_fields': {'code': str(i)}} for i in range(2)]


class FakeRuntime(VLLMRuntime):
    def __init__(self, wrong=False, truncated=False):
        super().__init__()
        self.candidate = candidate(simulated=False)
        self.wrong, self.truncated = wrong, truncated
        self.order = []

    def measure_response(self, prompt, max_new_tokens):
        constrained = 'response_format' in prompt
        self.order.append((prompt['id'], constrained))
        answer = json.dumps({'code': 'wrong'} if constrained and self.wrong else prompt['expected_fields'])
        return StreamSample(100, 3000 if constrained else 12000, 60 if constrained else 240, True,
                            answer=answer, finish_reason='length' if constrained and self.truncated else 'stop', token_count_source='engine_usage')


class PairTests(unittest.TestCase):
    def test_holdout_counts_failures_without_retry_or_selection(self):
        prompts = [dict(p, response_format=FORMAT, tags=['new']) for p in PROMPTS]
        for runtime in (FakeRuntime(), FakeRuntime(wrong=True), FakeRuntime(truncated=True)):
            got = run_holdout(runtime, prompts, FORMAT)
            self.assertEqual(len(runtime.order), 2)
            self.assertIsNone(got['selected_strategy'])
            self.assertIsNone(got['relative_latency_reduction'])
            self.assertEqual(got['decision'], 'REJECTED' if runtime.wrong or runtime.truncated else 'ACCEPTED')
            self.assertEqual(got['by_tag']['new']['documents'], 2)

    def test_holdout_requires_frozen_schema_before_any_request(self):
        runtime = FakeRuntime()
        with self.assertRaises(ValueError):
            run_holdout(runtime, PROMPTS, FORMAT)
        self.assertEqual(runtime.order, [])

    def test_selects_faster_only_with_same_quality(self):
        runtime = FakeRuntime()
        got = run_pairs(runtime, PROMPTS, FORMAT)
        self.assertEqual(got['selected_strategy'], 'json_schema')
        self.assertEqual(got['relative_latency_reduction'], .75)
        self.assertEqual(got['summaries']['baseline']['documents'], 4)
        self.assertNotIn('answer', json.dumps(got))
        self.assertEqual(runtime.order[2:], [('0', False), ('0', True), ('1', True), ('1', False), ('0', True), ('0', False), ('1', False), ('1', True)])

    def test_wrong_or_truncated_faster_answer_cannot_win(self):
        for runtime in (FakeRuntime(wrong=True), FakeRuntime(truncated=True)):
            got = run_pairs(runtime, PROMPTS, FORMAT)
            self.assertEqual(got['selected_strategy'], 'baseline')
            self.assertIsNone(got['relative_latency_reduction'])

    def test_payload_does_not_contain_gold_values(self):
        runtime = FakeRuntime()
        prompt = dict(PROMPTS[0], expected_fields={'code': 'SECRET-GOLD'}, response_format=FORMAT)
        payload = runtime._payload(prompt, 768, True)
        self.assertEqual(payload['response_format'], FORMAT)
        self.assertNotIn('SECRET-GOLD', json.dumps(payload))
        self.assertNotIn('expected_fields', json.dumps(payload))

    def test_oversized_experiment_is_rejected_before_inference(self):
        runtime = FakeRuntime()
        with self.assertRaises(ValueError):
            run_pairs(runtime, PROMPTS, FORMAT, repeats=5)
        self.assertEqual(runtime.order, [])

    def test_stream_records_same_response_quality_and_engine_usage(self):
        runtime = VLLMRuntime()
        runtime.candidate = candidate(simulated=False)
        runtime.spec = runtime.engines.get('vllm')
        runtime.base_url = 'http://127.0.0.1:8000'
        events = [
            {'choices': [{'delta': {'content': '{"code":'}, 'finish_reason': None}]},
            {'choices': [{'delta': {'content': '"x"}'}, 'finish_reason': None}]},
            {'choices': [{'delta': {}, 'finish_reason': 'stop'}]},
            {'choices': [], 'usage': {'completion_tokens': 8}},
        ]
        wire = ''.join('data: ' + json.dumps(e) + '\n\n' for e in events) + 'data: [DONE]\n'
        with patch('urllib.request.urlopen', return_value=io.BytesIO(wire.encode())):
            got = runtime.measure_response('extract', 768)
        self.assertEqual(got.answer, '{"code":"x"}')
        self.assertEqual(got.finish_reason, 'stop')
        self.assertEqual(got.output_tokens, 8)
        self.assertEqual(got.token_count_source, 'engine_usage')
        self.assertNotIn('code', repr(got))
