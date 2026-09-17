#!/usr/bin/env python3
"""Audit the three completed BurstGPT common-engine runs before Azure."""
import argparse, hashlib, json, re
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def audit_run(path, expected_kind, expected_capacity, expected_quantized, expected_count):
    path = Path(path)
    metadata = json.loads((path / 'run-metadata.json').read_text())
    summary = json.loads((path / 'summary.json').read_text())
    records = [json.loads(line) for line in (path / 'raw_requests.jsonl').read_text().splitlines() if line.strip()]
    system = [json.loads(line) for line in (path / 'system_telemetry.jsonl').read_text().splitlines() if line.strip()]
    events = json.loads((path / 'controller-events.json').read_text())
    log_text = ''.join((path / name).read_text(errors='replace') for name in ('run.stdout.log', 'run.stderr.log'))
    checks = {
        'metadata_passed': metadata.get('passed') is True,
        'kind_matches': metadata['configuration']['kind'] == expected_kind,
        'request_count_expected': len(records) == expected_count and summary['request_count'] == expected_count,
        'unique_request_ids': len({r['request_id'] for r in records}) == expected_count,
        'all_done': all(r['status'] == 'done' and not r['timeout'] and r['error'] is None for r in records),
        'all_outputs_exact_512': all(r['generated_token_count'] == 512 and len(r['token_ids']) == 512 for r in records),
        'telemetry_nonempty': bool(system),
        'telemetry_has_memory_and_kv': all({'gpu_memory_used_bytes','kv_capacity_blocks','kv_used_blocks'} <= set(s) for s in system),
        'summary_matches_records': summary['completed_count'] == expected_count and summary['error_count'] == 0 and summary['timeout_count'] == 0,
        'base_or_expanded_capacity_observed': max(int(s['kv_capacity_blocks']) for s in system) >= expected_capacity,
        'final_clean': summary['final_active_w4_layers'] == [] and summary['final_pending_layer_events'] == [] and not summary['final_executor_poisoned'],
        'no_oom_or_traceback_in_logs': not re.search(r'(?i)(out of memory|cuda out of memory|traceback \(most recent call last\))', log_text),
        'source_revision_pinned': metadata.get('model', {}).get('snapshot') == 'd04e592bb4f6aa9cfee91e2e20afa771667e1d4b',
    }
    if expected_kind == 'morphserve-default':
        checks.update({
            'quantized_layers_observed': summary['maximum_simultaneous_quantized_layers'] >= expected_quantized,
            'morph_and_recovery_observed': summary['morph_count'] > 0 and summary['recovery_count'] > 0,
            'real_w4_module_observed': any('WQLinear_GEMM' in cls for e in events for cls in e.get('real_w4_module_classes', {}).values()),
        })
    else:
        checks['quantized_layers_observed'] = summary['maximum_simultaneous_quantized_layers'] == expected_quantized
        if expected_kind == 'static_w4':
            preflight = Path('reproduction/experiments/end-to-end-preflight/static-w4/results-v2/run-metadata.json')
            preflight_setup = json.loads(preflight.read_text()).get('configuration', {}).get('setup', '') if preflight.exists() else ''
            checks['real_static_w4_install_path'] = 'install_autoawq_layer' in Path('reproduction/scripts/end_to_end_benchmark.py').read_text() and 'WQLinear_GEMM' in str(preflight_setup) and bool(metadata.get('model', {}).get('w4_artifact'))
    return {'path': str(path), 'checks': checks, 'passed': all(checks.values()), 'summary': {
        key: summary[key] for key in ('configuration','duration_s','completed_count','completion_rate','ttft_s','tpot_s','output_token_throughput_s','peak_kv_capacity_blocks','morph_count','recovery_count')
    }}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', default='reproduction'); p.add_argument('--trace', required=True); p.add_argument('--output', required=True); p.add_argument('--include-azure', action='store_true')
    args = p.parse_args(); root = Path(args.root); trace = Path(args.trace)
    burst_runs = [
        audit_run(root / 'experiments/end-to-end-burstgpt/fp16/results-batch5-v2', 'fp16', 512, 0, 123),
        audit_run(root / 'experiments/end-to-end-burstgpt/static-w4/results-batch5-v2', 'static_w4', 1024, 32, 123),
        audit_run(root / 'experiments/end-to-end-burstgpt/morphserve-default/results-batch5', 'morphserve-default', 1736, 8, 123),
    ]
    azure_runs = []
    if args.include_azure:
        azure_runs = [
            audit_run(root / 'experiments/end-to-end-azure/fp16/results-batch5', 'fp16', 512, 0, 94),
            audit_run(root / 'experiments/end-to-end-azure/static-w4/results-batch5', 'static_w4', 1024, 32, 94),
            audit_run(root / 'experiments/end-to-end-azure/morphserve-default/results-batch5', 'morphserve-default', 1736, 8, 94),
        ]
    azure_trace = root / 'traces/figure1b-inferred/azure-code-systematic-4.75x.jsonl'
    result = {'schema_version': 1, 'audit': 'Common-engine end-to-end gate', 'pre_azure_burstgpt_gate': all(r['passed'] for r in burst_runs), 'trace': str(trace), 'trace_sha256': sha256(trace), 'azure_trace': str(azure_trace), 'azure_trace_sha256': sha256(azure_trace), 'runs': burst_runs, 'azure_runs': azure_runs, 'passed': all(r['passed'] for r in burst_runs + azure_runs)}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2)); return 0 if result['passed'] else 1

if __name__ == '__main__': raise SystemExit(main())
