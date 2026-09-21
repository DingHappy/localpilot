"""Bounded, alternating response-format experiment against an existing local vLLM."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from statistics import mean
from urllib.parse import urlparse

from localpilot.benchmark.structured import score_fields, summarize_fields
from localpilot.engines.introspect import introspect, reconcile
from localpilot.hardware.profiler import HardwareProfiler
from localpilot.runtime.openai_compat import VLLMRuntime
from localpilot.schemas import SavedProfile
from localpilot.utils import atomic_write_json, stable_hash, utc_now


def run_holdout(runtime, prompts, response_format, max_new_tokens=768, checkpoint=None):
    """Validate the frozen strategy once per new input, without strategy selection."""
    if not 1 <= len(prompts) <= 20 or not 1 <= max_new_tokens <= 2048:
        raise ValueError('Use 1-20 documents and 1-2048 output tokens')
    if len({p.get('id') for p in prompts}) != len(prompts):
        raise ValueError('Document IDs must be unique')
    for prompt in prompts:
        if not prompt.get('expected_fields') or not prompt.get('image_url') or prompt.get('response_format') != response_format:
            raise ValueError('Every input needs labels, an image and the frozen response format')
    rows = []
    for prompt in prompts:
        sample = runtime.measure_response(prompt, max_new_tokens)
        verdict = score_fields(sample.answer if sample.ok else '', prompt['expected_fields'])
        completed = sample.ok and sample.finish_reason == 'stop'
        verdict['document_exact'] = verdict['document_exact'] and completed
        rows.append(dict(verdict, prompt_id=prompt['id'], tags=prompt.get('tags', []),
                         request_ok=sample.ok, completed=completed, finish_reason=sample.finish_reason,
                         total_ms=sample.total_ms, ttft_ms=sample.ttft_ms,
                         output_tokens=sample.output_tokens, token_count_source=sample.token_count_source))
        if checkpoint:
            checkpoint(rows)
    def summary(group):
        return dict(summarize_fields(group), complete_rate=sum(r['completed'] for r in group)/len(group),
                    mean_total_ms=mean(r['total_ms'] for r in group),
                    min_total_ms=min(r['total_ms'] for r in group), max_total_ms=max(r['total_ms'] for r in group))
    result = summary(rows)
    return dict(samples=rows, summaries={'json_schema': result},
                by_tag={tag: summary([r for r in rows if tag in r['tags']]) for tag in sorted({t for r in rows for t in r['tags']})},
                tested_strategy='json_schema', selected_strategy=None, relative_latency_reduction=None,
                decision='ACCEPTED' if result['document_accuracy'] == 1 and result['complete_rate'] == 1 else 'REJECTED')


def pair_prompts(prompt, response_format):
    baseline = copy.deepcopy(prompt)
    if 'response_format' in baseline:
        raise ValueError('Baseline already has a response format')
    constrained = copy.deepcopy(baseline)
    constrained['response_format'] = copy.deepcopy(response_format)
    return baseline, constrained


def run_pairs(runtime, prompts, response_format, repeats=2, max_new_tokens=768, checkpoint=None):
    if not 1 <= repeats <= 4 or not 1 <= len(prompts) <= 20:
        raise ValueError('Use 1-4 repetitions and 1-20 documents')
    if not 1 <= max_new_tokens <= 2048:
        raise ValueError('Output budget must be 1-2048 tokens')
    if not all(p.get('expected_fields') and p.get('image_url') for p in prompts):
        raise ValueError('Every document needs an image and labeled fields')
    if len({p.get('id') for p in prompts}) != len(prompts):
        raise ValueError('Document IDs must be unique')
    # Exactly one wire-level change, checked before any inference.
    for prompt in prompts:
        a, b = pair_prompts(prompt, response_format)
        pa, pb = runtime._payload(a, max_new_tokens, True), runtime._payload(b, max_new_tokens, True)
        if pb.pop('response_format') != response_format or pa != pb:
            raise ValueError('The pair differs beyond response_format')
    warmup = []
    for arm, prompt in zip(('baseline', 'json_schema'), pair_prompts(prompts[0], response_format)):
        sample = runtime.measure_response(prompt, max_new_tokens)
        warmup.append({'arm': arm, 'ok': sample.ok, 'finish_reason': sample.finish_reason,
                       'total_ms': sample.total_ms})
        if not sample.ok:
            raise RuntimeError('Warmup request failed for ' + arm)
    rows = []
    for repeat in range(repeats):
        for index, prompt in enumerate(prompts):
            pair = pair_prompts(prompt, response_format)
            order = (0, 1) if (repeat + index) % 2 == 0 else (1, 0)
            for arm_index in order:
                sample = runtime.measure_response(pair[arm_index], max_new_tokens)
                verdict = score_fields(sample.answer if sample.ok else '', prompt['expected_fields'])
                completed = sample.ok and sample.finish_reason == 'stop'
                if not completed:
                    verdict['document_exact'] = False
                row = dict(verdict, arm=('baseline', 'json_schema')[arm_index],
                           prompt_id=prompt['id'], repeat=repeat, order=len(rows),
                           request_ok=sample.ok, completed=completed,
                           ttft_ms=round(sample.ttft_ms, 2), total_ms=round(sample.total_ms, 2),
                           output_tokens=sample.output_tokens, token_count_source=sample.token_count_source,
                           finish_reason=sample.finish_reason)
                if not sample.ok:
                    row['error'] = 'request_failed'
                rows.append(row)
                if checkpoint:
                    checkpoint(rows)
    summaries = {}
    for arm in ('baseline', 'json_schema'):
        group = [r for r in rows if r['arm'] == arm]
        summaries[arm] = dict(summarize_fields(group),
                              complete_rate=sum(r['completed'] for r in group) / len(group),
                              mean_total_ms=mean(r['total_ms'] for r in group),
                              mean_ttft_ms=mean(r['ttft_ms'] for r in group),
                              mean_output_tokens=mean(r['output_tokens'] for r in group))
    a, b = summaries['baseline'], summaries['json_schema']
    acceptable = lambda s: s['document_accuracy'] == 1 and s['complete_rate'] == 1
    gain = 1 - b['mean_total_ms'] / a['mean_total_ms'] if a['mean_total_ms'] > 0 and acceptable(a) and acceptable(b) else None
    choice = 'json_schema' if gain is not None and gain >= .10 else ('baseline' if acceptable(a) else None)
    return {'warmup': warmup, 'samples': rows, 'summaries': summaries,
            'minimum_gain': .10, 'relative_latency_reduction': gain,
            'selected_strategy': choice, 'decision': 'SELECTED' if choice else 'REJECTED'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile-report', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--response-format', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--holdout', action='store_true', help='Test only the frozen schema strategy once per new input')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; choose a new path')
    url = urlparse(args.base_url)
    if url.scheme != 'http' or url.hostname not in {'127.0.0.1', 'localhost', '::1'} or url.username or url.password:
        parser.error('Run on the inference node against a loopback HTTP service')
    previous = json.loads(args.profile_report.read_text())
    profile = SavedProfile.from_dict(previous['best_profile'])
    if profile.simulated or profile.benchmark.simulated or previous['hardware'].get('simulated') is not False:
        parser.error('A real baseline profile is required')
    if profile.candidate.engine != 'vllm' or profile.candidate.concurrency != 1:
        parser.error('This bounded experiment requires a single-stream vLLM profile')
    hardware = HardwareProfiler().profile(simulate=False)
    if hardware.fingerprint != profile.hardware_fingerprint:
        parser.error('Execution hardware does not match the baseline profile')
    os.environ['LOCALPILOT_VLLM_BASE_URL'] = args.base_url
    runtime = VLLMRuntime()
    runtime.load_model(profile.candidate)
    server = introspect('vllm', base_url=args.base_url)
    if not server or not server.served_models or not server.raw_available or reconcile(profile.candidate, server):
        parser.error('Cannot verify the running model configuration')
    if profile.candidate.model_id not in server.served_models and profile.candidate.source_id not in server.served_models:
        parser.error('The baseline model is not served at this endpoint')
    config = json.loads(args.dataset.read_text())
    response_format = json.loads(args.response_format.read_text())
    prompts = config['prompts']
    evidence = {'status': 'RUNNING', 'started_at': utc_now(), 'hardware': hardware.to_dict(),
                'benchmark': {'simulated': False}, 'baseline_run_id': previous['run_id'],
                'candidate': profile.candidate.to_dict(), 'server_before': server.to_dict(),
                'changed_axis': 'response_format', 'response_format': response_format,
                'dataset_sha256': stable_hash(prompts), 'config_sha256': stable_hash(config),
                'max_new_tokens': config['max_new_tokens'], 'repeats': args.repeats,
                'concurrency': 1, 'managed_by_localpilot': False,
                'cache_policy': 'existing warm service, alternating AB/BA, no cache reset',
                'quality_policy': 'every document exact, no failed or truncated response',
                'scope': 'selection among two request strategies; no deployment change'}
    if args.holdout:
        evidence.update(changed_axis=None, repeats=1,
                        cache_policy='existing warm service; no warmup, retries or cache reset',
                        scope='new-input acceptance of frozen json_schema strategy; no optimization comparison')
    def checkpoint(rows):
        evidence['samples'] = rows
        atomic_write_json(args.output, evidence)
        print('completed requests', len(rows), flush=True)
    try:
        runtime.start_model()
        evidence['engine_version'] = runtime._get('/version')
        atomic_write_json(args.output, evidence)
        if args.holdout:
            evidence.update(run_holdout(runtime, prompts, response_format, config['max_new_tokens'], checkpoint))
        else:
            evidence.update(run_pairs(runtime, prompts, response_format, args.repeats, config['max_new_tokens'], checkpoint))
        after = introspect('vllm', base_url=args.base_url)
        evidence['server_after'] = after.to_dict() if after else None
        keys = ('served_models', 'cache_dtype', 'gpu_memory_utilization', 'max_model_len', 'num_gpu_blocks', 'block_size')
        unchanged = after is not None and all(evidence['server_before'].get(k) == evidence['server_after'].get(k) for k in keys)
        evidence['configuration_unchanged'] = unchanged
        evidence['status'] = 'COMPLETED' if unchanged else 'INVALID_COMPARISON'
        if not unchanged:
            evidence['selected_strategy'] = None
            evidence['decision'] = 'REJECTED'
    except Exception as exc:
        evidence['status'] = 'FAILED'
        evidence['error_type'] = type(exc).__name__
        raise
    finally:
        evidence['finished_at'] = utc_now()
        atomic_write_json(args.output, evidence)
        runtime.stop_model()  # Attached mode never owns or stops the server.
    print(json.dumps({k: evidence[k] for k in ('status', 'selected_strategy', 'summaries', 'relative_latency_reduction')}))
    return 0 if evidence['status'] == 'COMPLETED' and evidence['decision'] in {'SELECTED', 'ACCEPTED'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
