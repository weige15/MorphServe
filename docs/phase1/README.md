# Phase 1 Context

**Status:** complete. The final experiment found a reproducible mixed GPU-KV-admission and compute transition; the scheduler was not changed.

## Read only what you need

1. [`references/MORPHSERVE_PHASE1_REFERENCE.md`](../../references/MORPHSERVE_PHASE1_REFERENCE.md) — authoritative objective and evidence rules.
2. [`final-report.md`](final-report.md) — current result, configuration, plots, and reproduction command.
3. [`final-audit.md`](final-audit.md) — detailed pass/fail evidence when verification is needed.
4. [`baseline.md`](baseline.md) and [`calibration.md`](calibration.md) — baseline facts and configuration rationale when reproducing or modifying an experiment.
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
- Raw runs: `benchmark-results/phase1-final-rps-*` and `benchmark-results/phase1-final-repeat-*`
- Derived final artifacts: `benchmark-results/phase1-final/`
- Scheduler integrity target: `swiftLLM/swiftllm/server/scheduler.py`

## Documentation rule

Keep current phase documentation under `docs/phase1/`. Keep generated data and plots under `benchmark-results/`. Prefer updating an existing canonical document over creating another root-level Markdown file; put superseded material in `archive/` rather than deleting evidence.
