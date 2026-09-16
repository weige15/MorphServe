#!/usr/bin/env python3
"""Verify REPORT.md's currently cited artifact surfaces."""

import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
required=[
 'REPORT.md','docs/paper-evidence-brief.md','docs/source-map.md','docs/claim-register.md','docs/trace-audit.md','docs/task-artifact-audit.md',
 'results/raw/environment.json','results/raw/resource-usage-summary.json','results/raw/public-trace-audit.json','results/raw/figure1b-trace-window-inference.json',
 'results/raw/full-profile-memory-feasibility.json','traces/figure1b-inferred/summary.json',
 'traces/figure1b-inferred/azure-code-systematic-4.75x.jsonl','traces/figure1b-inferred/burstgpt-v1.1-systematic-1.75x.jsonl',
 'experiments/candidate-fp16-baseline/results-attempt-3/metrics.json',
 'experiments/autoawq-layer-switch/results/metrics.json','experiments/active-kv-switch/results-attempt-2/metrics.json',
 'experiments/lis-real-8layer/results/metrics.json','experiments/real-executor/results/metrics.json',
 'experiments/multirequest-ownership/results/metrics.json','experiments/async-layer-transfer/results/metrics.json',
 'profiles/llama31-8b-wikitext2-layers24-31.json',
]
missing=[path for path in required if not (ROOT/path).is_file()]
assert not missing,missing
for path in required:
    if path.endswith('.json'):
        json.load(open(ROOT/path))
checks={
 'fp16':json.load(open(ROOT/'experiments/candidate-fp16-baseline/results-attempt-3/metrics.json'))['passed'],
 'switch':json.load(open(ROOT/'experiments/autoawq-layer-switch/results/metrics.json'))['passed'],
 'active_kv':json.load(open(ROOT/'experiments/active-kv-switch/results-attempt-2/metrics.json'))['passed'],
 'lis8':json.load(open(ROOT/'experiments/lis-real-8layer/results/metrics.json'))['passed'],
 'executor':json.load(open(ROOT/'experiments/real-executor/results/metrics.json'))['passed'],
 'ownership':json.load(open(ROOT/'experiments/multirequest-ownership/results/metrics.json'))['passed'],
}
assert all(checks.values()),checks
static=json.load(open(ROOT/'experiments/static-autoawq/results/metrics.json'))
assert static['passed'] is False and static['gate']['repeat_logits_exact'] is False
memory=json.load(open(ROOT/'results/raw/full-profile-memory-feasibility.json'))
assert memory['feasible_simultaneously_pinned'] is False
async_copy=json.load(open(ROOT/'experiments/async-layer-transfer/results/metrics.json'))['copy']
assert async_copy['bytes_exact'] and async_copy['same_address'] and async_copy['prior_unfinished_after_enqueue']
windows=json.load(open(ROOT/'results/raw/figure1b-trace-window-inference.json'))
assert windows['azure_code']['inferred_start_s']==1073 and windows['burstgpt_1_v1.1']['inferred_start_s']==1781278
trace_summary=json.load(open(ROOT/'traces/figure1b-inferred/summary.json'))
for name,digest in trace_summary['sha256'].items():
    assert hashlib.sha256((ROOT/'traces/figure1b-inferred'/name).read_bytes()).hexdigest()==digest
task_sources=ROOT/'sources/task-source-audit'
for line in (task_sources/'MANIFEST.sha256').read_text().splitlines():
    digest,name=line.split(maxsplit=1)
    assert hashlib.sha256((task_sources/name).read_bytes()).hexdigest()==digest
report=(ROOT/'REPORT.md').read_text()
for phrase in ('partial / exact reproduction blocked','no Table 9','92.45%','[25,24,26]'):
    assert phrase in report,phrase
print('report artifact verification: PASS')
print(json.dumps({'positive_gates':checks,'async_copy_gate':True,'trace_manifest_gate':True,'expected_negative_static_repeat':True,'expected_memlock_blocker':True},sort_keys=True))
