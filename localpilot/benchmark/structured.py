"""Deterministic document-field scoring; never infer correctness from substrings."""
from __future__ import annotations

import json
import re
from typing import Any, Dict


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def score_fields(answer: str, expected: Dict[str, Any]) -> Dict[str, Any]:
    if not expected:
        raise ValueError("expected_fields must not be empty")
    # Some engines return a closed reasoning section before the final answer.
    final = answer.rsplit("</think>", 1)[-1].strip()
    fenced = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", final)
    if fenced:
        final = fenced.group(1)
    try:
        actual = json.loads(final, object_pairs_hook=_unique_object)
        valid = isinstance(actual, dict)
    except (ValueError, TypeError):
        actual, valid = {}, False
    if not valid:
        actual = {}
    matches = {
        key: key in actual and type(actual[key]) is type(value) and actual[key] == value
        for key, value in expected.items()
    }
    return {
        "json_valid": valid,
        "field_matches": matches,
        "fields_correct": sum(matches.values()),
        "fields_total": len(expected),
        "document_exact": valid and set(actual) == set(expected) and all(matches.values()),
    }


def summarize_fields(samples):
    count = len(samples)
    fields = sum(s["fields_total"] for s in samples)
    correct = sum(s["fields_correct"] for s in samples)
    return {
        "method": "json_exact_fields_v1",
        "documents": count,
        "fields_correct": correct,
        "fields_total": fields,
        "field_accuracy": correct / fields if fields else 0.0,
        "document_accuracy": sum(s["document_exact"] for s in samples) / count if count else 0.0,
        "json_valid_rate": sum(s["json_valid"] for s in samples) / count if count else 0.0,
        "request_failures": sum("error" in s for s in samples),
    }
