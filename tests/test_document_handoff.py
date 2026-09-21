import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('extract', ROOT / 'examples/document-extraction/extract.py')
extract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract)


class HandoffTests(unittest.TestCase):
    def test_request_matches_frozen_strategy_without_gold(self):
        dataset = json.loads((ROOT / 'evals/documents/holdout-v1/benchmark.json').read_text())
        payload = extract.build_request(ROOT / 'evals/documents/holdout-v1/holdout-01.png', 'model')
        self.assertEqual(payload['messages'][0]['content'][0]['text'], dataset['prompts'][0]['text'])
        self.assertEqual(payload['response_format'], dataset['prompts'][0]['response_format'])
        self.assertEqual(payload['max_tokens'], dataset['max_new_tokens'])
        self.assertNotIn('expected_fields', json.dumps(payload))

    def test_rejects_incomplete_or_invalid_fields(self):
        fields = dict(invoice_number='001', issue_date='2026-01-01', total_usd='1.00', seller='test', purchase_order=None)
        response = lambda content, finish='stop': {'choices': [{'finish_reason': finish, 'message': {'content': content}}]}
        self.assertEqual(extract.validate_response(response(json.dumps(fields))), fields)
        for r in [response(json.dumps(fields), 'length'), response(json.dumps(dict(fields, total_usd=1))),
                  response('{}'), response(json.dumps(fields)[:-1]+', "seller": "other"}')]:
            with self.assertRaises(ValueError):
                extract.validate_response(r)
