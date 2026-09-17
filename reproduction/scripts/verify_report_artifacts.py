#!/usr/bin/env python3
"""Verify REPORT.md's currently cited artifact surfaces."""

import csv
import hashlib
import json
import subprocess
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
 'experiments/real-executor/results/metrics.json','experiments/real-executor/results/commands.txt','experiments/real-executor/results/source-revision.txt','experiments/real-executor/results/source-status.txt','experiments/real-executor/results/nvidia-before.csv','experiments/real-executor/results/nvidia-after.csv','experiments/real-executor/results/run.exitcode',
 'experiments/multirequest-ownership/results-before-atomic-repair/metrics.json','experiments/multirequest-ownership/results-before-atomic-repair/CLASSIFICATION.md',
 'experiments/multirequest-ownership/results/metrics.json','experiments/multirequest-ownership/results/commands.txt','experiments/multirequest-ownership/results/source-revision.txt','experiments/multirequest-ownership/results/source-status.txt','experiments/multirequest-ownership/results/nvidia-before.csv','experiments/multirequest-ownership/results/nvidia-after.csv','experiments/multirequest-ownership/results/run.exitcode',
 'experiments/async-layer-transfer/results/metrics.json','experiments/async-layer-transfer/results/test.exitcode','experiments/async-layer-transfer/results/test.log',
 'experiments/async-layer-transfer/results/source-revision.txt','experiments/async-layer-transfer/results/source-status.txt','experiments/async-layer-transfer/results/wall-time.json',
 'experiments/async-layer-transfer/full-model-results-attempt-1/metrics.json','experiments/async-layer-transfer/full-model-results-attempt-1/transfer-summary.json','experiments/async-layer-transfer/full-model-results-attempt-2/metrics.json','experiments/async-layer-transfer/full-model-results-attempt-2/cuda-activity-trace.json','experiments/async-layer-transfer/full-model-results/metrics.json','experiments/async-layer-transfer/full-model-results/cuda-activity-trace.json','experiments/async-layer-transfer/full-model-results/activity-analysis.json','experiments/async-layer-transfer/full-model-results/transfer-summary.json','experiments/async-layer-transfer/full-model-results/commands.txt','experiments/async-layer-transfer/full-model-results/source-revision.txt','experiments/async-layer-transfer/full-model-results/source-status.txt','experiments/async-layer-transfer/full-model-results/nvidia-before.csv','experiments/async-layer-transfer/full-model-results/nvidia-after.csv','experiments/async-layer-transfer/full-model-results/run.exitcode',
 'figures/transfer-diagnostics.csv','figures/transfer-diagnostics.pdf','figures/transfer-diagnostics.png',
 'experiments/async-layer-transfer/alignment-results/test.exitcode','experiments/async-layer-transfer/transaction-results/test.exitcode','experiments/async-layer-transfer/activity-analysis-results/test.exitcode',
 'profiles/llama31-8b-wikitext2-layers24-31.json',
 'experiments/synthetic-gpu-replay/results/raw.jsonl','experiments/synthetic-gpu-replay/results/metrics.json','experiments/synthetic-gpu-replay/results/summary.json','experiments/synthetic-gpu-replay/results/run-metadata.json','experiments/synthetic-gpu-replay/results/commands.txt','experiments/synthetic-gpu-replay/results/source-revision.txt','experiments/synthetic-gpu-replay/results/source-status.txt','experiments/synthetic-gpu-replay/results/nvidia-before.csv','experiments/synthetic-gpu-replay/results/nvidia-after.csv','experiments/synthetic-gpu-replay/results/run.exitcode',
 'configs/end-to-end-benchmark.json','configs/request-payload-1024.json','scripts/end_to_end_benchmark.py','scripts/run_end_to_end_benchmark.sh','scripts/plot_end_to_end_benchmark.py','scripts/audit_end_to_end_runs.py','docs/paper-version-delta.md','sources/morphserve-mlsys2026-conference-final.txt','results/raw/conference-final-source-audit.json','results/raw/burstgpt-end-to-end-audit.json','results/raw/end-to-end-audit.json','results/raw/end-to-end-plot-provenance.json','experiments/end-to-end-report.md','figures/end-to-end/comparison.json','figures/end-to-end/comparison.csv',
 'experiments/end-to-end-burstgpt/fp16/results-batch5-v2/summary.json','experiments/end-to-end-burstgpt/fp16/results-batch5-v2/run-metadata.json','experiments/end-to-end-burstgpt/fp16/results-batch5-v2/raw_requests.jsonl','experiments/end-to-end-burstgpt/fp16/results-batch5-v2/system_telemetry.jsonl','experiments/end-to-end-burstgpt/fp16/results-batch5-v2/controller-events.json','experiments/end-to-end-burstgpt/fp16/results-batch5-v2/run.exitcode',
 'experiments/end-to-end-burstgpt/static-w4/results-batch5-v2/summary.json','experiments/end-to-end-burstgpt/static-w4/results-batch5-v2/run-metadata.json','experiments/end-to-end-burstgpt/static-w4/results-batch5-v2/raw_requests.jsonl','experiments/end-to-end-burstgpt/static-w4/results-batch5-v2/system_telemetry.jsonl','experiments/end-to-end-burstgpt/static-w4/results-batch5-v2/controller-events.json','experiments/end-to-end-burstgpt/static-w4/results-batch5-v2/run.exitcode',
 'experiments/end-to-end-burstgpt/morphserve-default/results-batch5/summary.json','experiments/end-to-end-burstgpt/morphserve-default/results-batch5/run-metadata.json','experiments/end-to-end-burstgpt/morphserve-default/results-batch5/raw_requests.jsonl','experiments/end-to-end-burstgpt/morphserve-default/results-batch5/system_telemetry.jsonl','experiments/end-to-end-burstgpt/morphserve-default/results-batch5/controller-events.json','experiments/end-to-end-burstgpt/morphserve-default/results-batch5/run.exitcode',
 'experiments/end-to-end-azure/fp16/results-batch5/summary.json','experiments/end-to-end-azure/fp16/results-batch5/run-metadata.json','experiments/end-to-end-azure/fp16/results-batch5/raw_requests.jsonl','experiments/end-to-end-azure/fp16/results-batch5/system_telemetry.jsonl','experiments/end-to-end-azure/fp16/results-batch5/controller-events.json','experiments/end-to-end-azure/fp16/results-batch5/run.exitcode',
 'experiments/end-to-end-azure/static-w4/results-batch5/summary.json','experiments/end-to-end-azure/static-w4/results-batch5/run-metadata.json','experiments/end-to-end-azure/static-w4/results-batch5/raw_requests.jsonl','experiments/end-to-end-azure/static-w4/results-batch5/system_telemetry.jsonl','experiments/end-to-end-azure/static-w4/results-batch5/controller-events.json','experiments/end-to-end-azure/static-w4/results-batch5/run.exitcode',
 'experiments/end-to-end-azure/morphserve-default/results-batch5/summary.json','experiments/end-to-end-azure/morphserve-default/results-batch5/run-metadata.json','experiments/end-to-end-azure/morphserve-default/results-batch5/raw_requests.jsonl','experiments/end-to-end-azure/morphserve-default/results-batch5/system_telemetry.jsonl','experiments/end-to-end-azure/morphserve-default/results-batch5/controller-events.json','experiments/end-to-end-azure/morphserve-default/results-batch5/run.exitcode',
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
 'real_executor_current':json.load(open(ROOT/'experiments/real-executor/results/metrics.json'))['passed'],
 'ownership_current':json.load(open(ROOT/'experiments/multirequest-ownership/results/metrics.json'))['passed'],
 'async_full_model_current':json.load(open(ROOT/'experiments/async-layer-transfer/full-model-results/metrics.json'))['passed'],
 'synthetic_replay_current':json.load(open(ROOT/'experiments/synthetic-gpu-replay/results/metrics.json'))['passed'],
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
assert references['headline_claims']['accuracy_mode_llm_pq_quality_gap_closure_average_percent'] == 41.3
assert references['headline_claims']['accuracy_mode_p95_ttft_speedup_vs_pyramidkv_average'] == 1.73
assert 'conference_final_added_comparisons' in references
assert references['conference_final_added_comparisons']['note'].startswith('These are reported conference-final values, not local measurements')
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
    # New experiment/report artifacts are intentionally auditable in the
    # current worktree before a commit; the hash is the integrity anchor.
    assert (row['tracked'] and row['artifact_last_commit']) or row['worktree_status'],path
memory=json.load(open(ROOT/'results/raw/full-profile-memory-feasibility.json'))
assert memory['feasible_simultaneously_pinned'] is False
async_copy=json.load(open(ROOT/'experiments/async-layer-transfer/results/metrics.json'))['copy']
assert async_copy['bytes_exact'] and async_copy['same_address'] and async_copy['prior_unfinished_after_enqueue']
assert (ROOT/'experiments/async-layer-transfer/results/test.exitcode').read_text().strip()=='0'
async_log=(ROOT/'experiments/async-layer-transfer/results/test.log').read_text()
assert 'Ran 5 tests' in async_log and async_log.rstrip().endswith('OK')
assert (ROOT/'experiments/async-layer-transfer/results/source-status.txt').read_text()==''
async_revision=(ROOT/'experiments/async-layer-transfer/results/source-revision.txt').read_text().strip()
assert subprocess.run(['git','-C',str(ROOT.parent),'cat-file','-e',f'{async_revision}^{{commit}}']).returncode==0
async_wall=json.load(open(ROOT/'experiments/async-layer-transfer/results/wall-time.json'))['test_wall_seconds']
assert 0 < async_wall < 60
assert (ROOT/'experiments/async-layer-transfer/alignment-results/test.exitcode').read_text().strip()=='0'
assert (ROOT/'experiments/async-layer-transfer/transaction-results/test.exitcode').read_text().strip()=='0'
transaction_log=(ROOT/'experiments/async-layer-transfer/transaction-results/test.log').read_text()
assert 'Ran 18 tests' in transaction_log and transaction_log.rstrip().endswith('OK')
for test_name in (
    'test_expand_rejects_invalid_batches_before_acquisition',
    'test_expand_rejects_overlapping_registered_regions_before_acquisition',
    'test_double_copy_failure_poisoned_executor_blocks_further_actions',
    'test_poisoned_recovery_does_not_reattach_invalid_kv_snapshot',
    'test_recovery_restore_failure_is_atomically_compensated',
):
    assert test_name in transaction_log,test_name
assert (ROOT/'experiments/async-layer-transfer/activity-analysis-results/test.exitcode').read_text().strip()=='0'
for current_dir in ('real-executor/results','multirequest-ownership/results','async-layer-transfer/full-model-results','synthetic-gpu-replay/results'):
    assert (ROOT/'experiments'/current_dir/'run.exitcode').read_text().strip()=='0',current_dir
current_async=json.load(open(ROOT/'experiments/async-layer-transfer/full-model-results/metrics.json'))
assert current_async['passed'] and current_async['gate']['cuda_activity_kernel_copy_overlap']
activity=current_async['activity_trace']['analysis']
assert activity['passed']
assert activity['phases']['W4']['copy_stream'] != activity['phases']['W4']['kernel_stream']
assert activity['phases']['FP16']['copy_stream'] != activity['phases']['FP16']['kernel_stream']
current_replay=json.load(open(ROOT/'experiments/synthetic-gpu-replay/results/metrics.json'))
assert current_replay['passed'] and current_replay['gate']['all_ids_once'] and current_replay['gate']['required_metadata']
current_summary=json.load(open(ROOT/'experiments/synthetic-gpu-replay/results/summary.json'))
assert current_summary['completed_count']==3 and current_summary['total_emitted_tokens']==9 and current_summary['error_count']==0 and current_summary['timeout_count']==0
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
end_to_end_audit=json.load(open(ROOT/'results/raw/end-to-end-audit.json'))
assert end_to_end_audit['passed'] and len(end_to_end_audit['runs']) == 3 and len(end_to_end_audit['azure_runs']) == 3
plot_provenance=json.load(open(ROOT/'results/raw/end-to-end-plot-provenance.json'))
for item in plot_provenance['artifacts']:
    assert hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
trace_summary=json.load(open(ROOT/'traces/figure1b-inferred/summary.json'))
for name,digest in trace_summary['sha256'].items():
    assert hashlib.sha256((ROOT/'traces/figure1b-inferred'/name).read_bytes()).hexdigest()==digest
task_sources=ROOT/'sources/task-source-audit'
for line in (task_sources/'MANIFEST.sha256').read_text().splitlines():
    digest,name=line.split(maxsplit=1)
    assert hashlib.sha256((task_sources/name).read_bytes()).hexdigest()==digest
report=(ROOT/'REPORT.md').read_text()
for phrase in ('partial / exact reproduction blocked','Tables 1–9','92.45%','[25,24,26]','common-engine end-to-end workload comparison','configs/claim-evidence-map.json'):
    assert phrase in report,phrase
for index in range(1,31):
    assert report.count(f'| H{index} |')==1,index
print('report artifact verification: PASS')
print(json.dumps({'current_positive_gates':checks,'supporting_historical_gates':historical_checks,'async_copy_gate':True,'trace_manifest_gate':True,'expected_negative_static_repeat':True,'expected_memlock_blocker':True},sort_keys=True))
