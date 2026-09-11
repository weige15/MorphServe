# Phase 1 Context

**Status:** complete. The final experiment found a reproducible mixed GPU-KV-admission and compute transition; the scheduler was not changed.

## Read only what you need

1. [`references/MORPHSERVE_PHASE1_REFERENCE.md`](../../references/MORPHSERVE_PHASE1_REFERENCE.md) — authoritative objective and evidence rules.
2. [`kv-admission-sweep-report.md`](kv-admission-sweep-report.md) — current result, configuration, plots, and reproduction command.
3. [`kv-admission-sweep-audit.md`](kv-admission-sweep-audit.md) — detailed pass/fail evidence when verification is needed.
4. [`baseline-configuration.md`](baseline-configuration.md) and [`kv-capacity-calibration.md`](kv-capacity-calibration.md) — baseline facts and configuration rationale when reproducing or modifying an experiment.
5. [`archive/`](archive/) — superseded reports; consult only for historical provenance.

Do not load every report for routine work. The final report supersedes the earlier short-workload report; the final audit supersedes the earlier completion audit.

## Current experiment facts

- Model: unquantized FP16 Llama 3.1 8B on CUDA device 3.
- Final fixed workload: 2048 prompt tokens, 16 output tokens, 20 requests per point, fixed open-loop arrivals.
- Final profiled capacity: 1768 GPU blocks / 28,288 logical KV token slots.
- KV admission pressure begins around 1 RPS, with queueing and P95 TTFT increasing through the 1.25–2 RPS points.
- Classification: mixed KV + compute; no preemption or swap was required.
- Historical short workload: 3880 GPU blocks with 8/4-token requests; compute-bound and retained only as baseline evidence.

## Working locations

- Benchmark code: `swiftLLM/benchmark/`
- Baseline smoke runs: `benchmark-results/phase-1/baseline-low-load/runs/` (`fixed-arrivals/` and `poisson-arrivals/`).
- KV calibration runs: `benchmark-results/phase-1/kv-capacity-calibration/runs/`.
- Final sweep raw runs: `benchmark-results/phase-1/final-kv-admission-sweep/runs/` (`sweep-rps-*` plus `repeat-rps-*`).
- Final sweep logs: `benchmark-results/phase-1/final-kv-admission-sweep/logs/`.
- Final derived artifacts: `benchmark-results/phase-1/final-kv-admission-sweep/`.
- Scheduler integrity target: `swiftLLM/swiftllm/server/scheduler.py`

## Documentation rule

Keep current Phase 1 documentation under `docs/phase-1/`. Keep generated data and plots under `benchmark-results/phase-1/`, grouped by experiment. Prefer updating an existing canonical document over creating another root-level Markdown file; put superseded material in `archive/` rather than deleting evidence.
