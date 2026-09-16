#!/usr/bin/env python3
"""Verify REPORT.md's currently cited artifact surfaces."""

import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
required=[
 'REPORT.md','docs/paper-evidence-brief.md','docs/source-map.md','docs/claim-register.md','docs/trace-audit.md','docs/task-artifact-audit.md','configs/claim-evidence-map.json',
 'results/raw/environment.json','results/raw/resource-usage-summary.json','results/raw/public-trace-audit.json','results/raw/figure1b-trace-window-inference.json','results/raw/claim-evidence-provenance.json','results/raw/transaction-followup-review.md','results/raw/transaction-rereview.md',
 'results/raw/full-profile-memory-feasibility.json','traces/figure1b-inferred/summary.json',
 'traces/figure1b-inferred/azure-code-systematic-4.75x.jsonl','traces/figure1b-inferred/burstgpt-v1.1-systematic-1.75x.jsonl',
 'experiments/candidate-fp16-baseline/results-attempt-3/metrics.json',
 'experiments/autoawq-layer-switch/results/metrics.json','experiments/active-kv-switch/results-attempt-2/metrics.json',
 'experiments/lis-real-8layer/results/metrics.json',
 'experiments/real-executor/results-before-atomic-repair/metrics.json','experiments/real-executor/results-before-atomic-repair/CLASSIFICATION.md',
 'experiments/multirequest-ownership/results-before-atomic-repair/metrics.json','experiments/multirequest-ownership/results-before-atomic-repair/CLASSIFICATION.md',
 'experiments/async-layer-transfer/results/metrics.json',
 'experiments/async-layer-transfer/full-model-results-attempt-1/metrics.json','experiments/async-layer-transfer/full-model-results-attempt-1/transfer-summary.json',
 'figures/transfer-diagnostics.csv','figures/transfer-diagnostics.pdf','figures/transfer-diagnostics.png',
 'experiments/async-layer-transfer/alignment-results/test.exitcode','experiments/async-layer-transfer/transaction-results/test.exitcode','experiments/async-layer-transfer/activity-analysis-results/test.exitcode',
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
}
historical_checks={
 'executor_pre_atomic':json.load(open(ROOT/'experiments/real-executor/results-before-atomic-repair/metrics.json'))['passed'],
 'ownership_pre_atomic':json.load(open(ROOT/'experiments/multirequest-ownership/results-before-atomic-repair/metrics.json'))['passed'],
}
assert all(checks.values()),checks
assert all(historical_checks.values()),historical_checks
assert 'not current-revision validation' in (ROOT/'experiments/real-executor/results-before-atomic-repair/CLASSIFICATION.md').read_text()
assert 'not current-revision validation' in (ROOT/'experiments/multirequest-ownership/results-before-atomic-repair/CLASSIFICATION.md').read_text()
static=json.load(open(ROOT/'experiments/static-autoawq/results/metrics.json'))
assert static['passed'] is False and static['gate']['repeat_logits_exact'] is False
references=json.load(open(ROOT/'configs/paper-reference-values.json'))
assert not any('llm_pq' in key or 'pyramidkv' in key for key in references['headline_claims'])
assert 'objective_only_values_not_found_in_target_pdf' in references
claim_map=json.load(open(ROOT/'configs/claim-evidence-map.json'))
assert set(claim_map['claims'])=={f'H{i}' for i in range(1,31)}
evidence_paths=set()
for claim,row in claim_map['claims'].items():
    assert row['paper_reference'] and row['comparison'] and row['limitation'],claim
    for field in ('paper_reference','commands','configs','raw','comparison'):
        for path in row[field]:
            assert (ROOT/path).is_file(),(claim,field,path)
            evidence_paths.add(path)
provenance=json.load(open(ROOT/'results/raw/claim-evidence-provenance.json'))
assert set(provenance['artifacts'])==evidence_paths
for path,row in provenance['artifacts'].items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==row['sha256'],path
    assert row['tracked'] and row['artifact_last_commit'] and row['worktree_status'] is None,path
memory=json.load(open(ROOT/'results/raw/full-profile-memory-feasibility.json'))
assert memory['feasible_simultaneously_pinned'] is False
async_copy=json.load(open(ROOT/'experiments/async-layer-transfer/results/metrics.json'))['copy']
assert async_copy['bytes_exact'] and async_copy['same_address'] and async_copy['prior_unfinished_after_enqueue']
assert (ROOT/'experiments/async-layer-transfer/alignment-results/test.exitcode').read_text().strip()=='0'
assert (ROOT/'experiments/async-layer-transfer/transaction-results/test.exitcode').read_text().strip()=='0'
assert (ROOT/'experiments/async-layer-transfer/activity-analysis-results/test.exitcode').read_text().strip()=='0'
full_async_attempt=json.load(open(ROOT/'experiments/async-layer-transfer/full-model-results-attempt-1/metrics.json'))
assert not full_async_attempt['passed'] and full_async_attempt['gate']['final_fp16_bytes_exact']
transfer_summary=json.load(open(ROOT/'experiments/async-layer-transfer/full-model-results-attempt-1/transfer-summary.json'))
assert transfer_summary['precisions']['W4']['repeats']==3 and transfer_summary['precisions']['FP16']['repeats']==3
assert 15.17 < transfer_summary['precisions']['W4']['copy_ms']['median'] < 15.22
assert 57.89 < transfer_summary['precisions']['FP16']['copy_ms']['median'] < 57.90
figure_hashes={
 'figures/transfer-diagnostics.csv':'86bfb5f61bc609574062fa761d11bc43b7c7de39cd5a1e4083341fe64a6b713f',
 'figures/transfer-diagnostics.pdf':'04eefa73e916892f131db8cbd4ae1194927d92d4d9d590b0d26d580cfaae0ac8',
 'figures/transfer-diagnostics.png':'744de99e381e68b38fa01cbbf5b1a5b93d1aba0bfd908e3e1ce54b16a2ab0e80',
}
for path,digest in figure_hashes.items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
with open(ROOT/'figures/transfer-diagnostics.csv',newline='') as handle:
    figure_rows=list(csv.DictReader(handle))
assert len(figure_rows)==len(full_async_attempt['rows'])==6
for plotted,raw in zip(figure_rows,full_async_attempt['rows']):
    assert plotted['precision']==raw['precision']
    assert int(plotted['transfer_bytes'])==raw['transfer_bytes']
    assert abs(float(plotted['copy_ms'])-raw['copy_ms']) < 1e-12
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
for phrase in ('partial / exact reproduction blocked','no Table 9','92.45%','[25,24,26]','configs/claim-evidence-map.json'):
    assert phrase in report,phrase
for index in range(1,31):
    assert report.count(f'| H{index} |')==1,index
print('report artifact verification: PASS')
print(json.dumps({'current_positive_gates':checks,'supporting_historical_gates':historical_checks,'async_copy_gate':True,'trace_manifest_gate':True,'expected_negative_static_repeat':True,'expected_memlock_blocker':True},sort_keys=True))
