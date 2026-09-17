#!/usr/bin/env python3
"""Regenerate common-engine end-to-end comparison metrics and plots from raw runs."""
import argparse, csv, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

CONFIGS = ('fp16', 'static_w4', 'morphserve-default')
LABELS = {'fp16': 'FP16', 'static_w4': 'static real W4', 'morphserve-default': 'MorphServe-default'}
COLORS = {'fp16': '#4c78a8', 'static_w4': '#f58518', 'morphserve-default': '#54a24b'}


def load(root, workload, config):
    path = Path(root) / f'experiments/end-to-end-{workload}/{"static-w4" if config == "static_w4" else config}/results-batch5{"-v2" if workload == "burstgpt" and config != "morphserve-default" else ""}'
    # Azure outputs and MorphServe Burst use the non-v2 directory.
    if not (path / 'summary.json').exists():
        path = Path(root) / f'experiments/end-to-end-{"burstgpt" if workload == "burstgpt" else "azure"}/{"static-w4" if config == "static_w4" else config}/results-batch5'
    return path, json.loads((path / 'summary.json').read_text()), json.loads((path / 'run-metadata.json').read_text())


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', default='reproduction'); p.add_argument('--output-dir', required=True)
    args = p.parse_args(); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []; loaded = {}
    for workload in ('burstgpt', 'azure'):
        loaded[workload] = {}
        for config in CONFIGS:
            path, summary, metadata = load(args.root, workload, config)
            loaded[workload][config] = (path, summary, metadata)
            rows.append({'workload': workload, 'configuration': LABELS[config], 'kind': config,
                         'run_path': str(path), 'request_count': summary['request_count'],
                         'completed_count': summary['completed_count'], 'completion_rate': summary['completion_rate'],
                         'duration_s': summary['duration_s'], 'ttft_p50_s': summary['ttft_s']['p50'],
                         'ttft_p95_s': summary['ttft_s']['p95'], 'ttft_p99_s': summary['ttft_s']['p99'],
                         'tpot_p50_s': summary['tpot_s']['p50'], 'tpot_p95_s': summary['tpot_s']['p95'],
                         'tpot_p99_s': summary['tpot_s']['p99'], 'output_token_throughput_s': summary['output_token_throughput_s'],
                         'completed_request_throughput_s': summary['completed_request_throughput_s'],
                         'ttft_over_2s_fraction_submitted': summary['ttft_over_2s_fraction_submitted'],
                         'peak_kv_capacity_blocks': summary['peak_kv_capacity_blocks'], 'peak_kv_used_blocks': summary['peak_kv_used_blocks'],
                         'time_weighted_kv_capacity_blocks': summary['time_weighted_kv_capacity_blocks'],
                         'time_weighted_kv_used_blocks': summary['time_weighted_kv_used_blocks'],
                         'morph_count': summary['morph_count'], 'recovery_count': summary['recovery_count'],
                         'maximum_simultaneous_quantized_layers': summary['maximum_simultaneous_quantized_layers'],
                         'valid_for_comparison': summary['valid_for_comparison']})
    (out / 'comparison.json').write_text(json.dumps({'schema_version': 1, 'classification': 'modified-condition end-to-end comparison', 'rows': rows}, indent=2, sort_keys=True) + '\n')
    with (out / 'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    # Latency panels use the same percentile definition emitted by the runner.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    x = np.arange(3); width = .25
    for ax, workload, title in zip(axes, ('burstgpt', 'azure'), ('BurstGPT v1.1 inferred window', 'Azure Code inferred window')):
        for i, q in enumerate(('p50', 'p95', 'p99')):
            vals = [loaded[workload][c][1]['ttft_s'][q] for c in CONFIGS]
            bars = ax.bar(x + (i - 1) * width, vals, width, label=q.upper())
            for bar, value in zip(bars, vals): ax.text(bar.get_x()+bar.get_width()/2, value, f'{value:.0f}', ha='center', va='bottom', fontsize=7, rotation=90)
        ax.set_title(title); ax.set_xticks(x, [LABELS[c] for c in CONFIGS], rotation=18); ax.set_ylabel('TTFT (s)'); ax.grid(axis='y', alpha=.25)
    axes[0].legend(frameon=False, ncol=3, fontsize=8); fig.suptitle('End-to-end TTFT percentiles; lower is better')
    fig.savefig(out / 'ttft-percentiles.png', dpi=180); fig.savefig(out / 'ttft-percentiles.pdf'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    x = np.arange(6); vals = [r['output_token_throughput_s'] for r in rows]
    bars = ax.bar(x, vals, color=[COLORS[r['kind']] for r in rows])
    ax.set_xticks(x, [f"{r['workload'].title()}\n{r['configuration']}" for r in rows], rotation=18); ax.set_ylabel('Completed output tokens/s'); ax.set_title('End-to-end output throughput (modified conditions)'); ax.grid(axis='y', alpha=.25)
    for bar, value in zip(bars, vals): ax.text(bar.get_x()+bar.get_width()/2, value, f'{value:.1f}', ha='center', va='bottom', fontsize=8)
    fig.savefig(out / 'throughput.png', dpi=180); fig.savefig(out / 'throughput.pdf'); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    for ax, workload in zip(axes, ('burstgpt', 'azure')):
        for config in CONFIGS:
            path, summary, _ = loaded[workload][config]
            samples = [json.loads(line) for line in (path / 'system_telemetry.jsonl').read_text().splitlines() if line.strip()]
            t = [s['timestamp_s'] for s in samples]; cap = [s['kv_capacity_blocks'] for s in samples]; used = [s['kv_used_blocks'] for s in samples]
            ax.plot(t, cap, color=COLORS[config], linestyle='--', alpha=.75, label=f'{LABELS[config]} capacity')
            ax.plot(t, used, color=COLORS[config], label=f'{LABELS[config]} used')
        ax.set_title(f'{workload.title()} inferred window'); ax.set_ylabel('KV blocks'); ax.grid(alpha=.25)
    axes[-1].set_xlabel('Benchmark time (s)'); axes[0].legend(frameon=False, ncol=3, fontsize=7); fig.suptitle('Physical KV capacity and occupancy telemetry')
    fig.savefig(out / 'kv-capacity-occupancy.png', dpi=180); fig.savefig(out / 'kv-capacity-occupancy.pdf'); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    for ax, workload in zip(axes, ('burstgpt', 'azure')):
        path, _, _ = loaded[workload]['morphserve-default']; samples = [json.loads(line) for line in (path / 'system_telemetry.jsonl').read_text().splitlines() if line.strip()]
        t = [s['timestamp_s'] for s in samples]; q = [s['queue_depth'] for s in samples]; layers = [s['active_w4_layer_count'] for s in samples]
        ax.plot(t, q, label='waiting queue depth', color='#4c78a8'); ax.plot(t, layers, label='active W4 layers', color='#54a24b'); ax.set_ylabel('count'); ax.set_title(f'MorphServe-default controller telemetry: {workload.title()}'); ax.grid(alpha=.25)
    axes[-1].set_xlabel('Benchmark time (s)'); axes[0].legend(frameon=False, fontsize=8); fig.suptitle('Reconstructed controller workload telemetry')
    fig.savefig(out / 'morphserve-controller.png', dpi=180); fig.savefig(out / 'morphserve-controller.pdf'); plt.close(fig)
    print(f'wrote {len(rows)} rows and plots to {out}')

if __name__ == '__main__': main()
