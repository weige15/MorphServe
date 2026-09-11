# Phase-1 Completion Audit

## Objective restatement

Complete a reproducible, open-loop RPS sweep of the unchanged full-precision SwiftLLM baseline, holding one practical workload and all serving parameters fixed while varying only offered RPS. Preserve raw request/system telemetry, regenerate all summaries/tables/plots from machine-readable data, independently verify percentile calculations, classify the observed bottleneck using logical KV and queue evidence, and report the result without claiming memory saturation unless the required correlated evidence exists.

## Prompt-to-artifact checklist

| Requirement | Evidence inspected | Status |
|---|---|---|
| Read authoritative baseline/reference inputs before experimental work | `docs/phase-1/baseline-configuration.md`, `docs/phase-1/archive/low-load-benchmark-harness-report.md`, `references/MORPHSERVE_PHASE1_REFERENCE.md` | PASS |
| Preserve full-precision validated baseline and exact baseline EngineConfig | Final run `metadata.json` files; `docs/phase-1/archive/short-workload-saturation-report.md` fixed configuration; all runs report 3,880 GPU blocks | PASS |
| Keep scheduler semantics, FCFS, admission, preemption/swap, precision, and KV policy unchanged | `benchmark-results/phase-1/historical-short-workload/scheduler-audit.txt`; before/after/baseline SHA-256 all `80e214...`; `git diff --exit-code -- swiftLLM/swiftllm/server/scheduler.py` | PASS |
| Perform only minimum calibration before final sweep | `benchmark-results/phase-1/historical-short-workload/runs/warmup-calibration-fixed/`; one fixed 8/4-token warmup calibration selected the one-request warmup policy | PASS |
| Hold model, EngineConfig, GPU, prompt/output lengths, arrival mode, seed, warmup, telemetry, and request count fixed | `sweep-summary.json` `verification.fixed_configuration_consistent=true`; 12/12 metadata records checked; only target RPS and derived schedule window/run identity differ | PASS |
| Use open-loop traffic independent of completion | `swiftLLM/benchmark/workload.py` and `run.py`; raw 128-RPS request rows show later arrival before first request completion | PASS |
| Establish underloaded point | Canonical 0.5-RPS run: zero waiting queue, 0.5306 completed RPS, TTFT P95 0.0532 s | PASS |
| Coarse monotonic sweep to knee/overload | Canonical targets: 0.5, 4, 16, 64, 128 RPS, with queue/throughput degradation visible | PASS |
| Fine points below/near/above candidate knee | Additional 24, 32, 40, 48, 80 RPS; representatives are 4, 16, and 64 RPS | PASS |
| Repeat important knee-region points | `repeat-rps-32.00` and `repeat-rps-64.00`; repeat values retained and plotted | PASS |
| Preserve raw request/system telemetry at every measured point | 12 run directories each contain `metadata.json`, `requests.jsonl`, `telemetry.jsonl`, and `summary.json`; 192/192 requests completed | PASS |
| Compute required arrival/throughput/latency metrics | `rps-sweep-metrics.csv` and report table contain target/achieved arrival, completed/token throughput, request counts, TTFT, queueing, and TPOT distributions | PASS |
| Compute queue/KV/swap/block/memory telemetry | `summary.json` `telemetry_stats`, `rps-sweep-metrics.csv`, report telemetry table, and raw JSONL fields contain waiting/running/swapped stats, decoding/GPU blocks, logical KV, swaps/preemptions, and HBM stats | PASS |
| Do not infer memory saturation from TTFT alone | Report explicitly uses logical KV and queue/swap evidence; classifies result as compute-bound | PASS |
| Generate required plots from saved data | `benchmark-results/phase-1/historical-short-workload/plots/`: P95 TTFT, P95 queueing, peak logical KV, completed throughput, plus representative time series | PASS |
| Preserve representative time series below/near/above knee | Raw `telemetry.jsonl` paths for 4, 16, and 64 RPS and `representative-queue-and-kv-over-time.png` | PASS |
| Regenerate summaries/tables/plots from raw machine-readable files | `swiftLLM/benchmark/analyze.py`; summaries regenerated with `benchmark.summarize`; `sweep-summary.json` and `rps-sweep-metrics.csv` produced by the analysis command | PASS |
| Independently recompute percentiles from raw timestamps | `sweep-summary.json` per-run `verification.per_run[*].percentiles`; TTFT/queueing/TPOT checks pass for all 12 runs | PASS |
| Verify plotted data source | `sweep-summary.json` `verification.plot_source_matches_points=true`; `plot_data` records exact source series | PASS |
| Produce required report contents | `docs/phase-1/archive/short-workload-saturation-report.md`: configuration, exact commands, complete tables, paths, estimate, causal evidence, qualitative comparison, discrepancy, uncertainty, next test | PASS |
| Report local mismatch honestly and give smallest next adjustment | Report states no KV saturation; recommends retaining baseline and increasing fixed output length toward ~15,500 tokens for a controlled KV-pressure test | PASS |
| Avoid all Phase-1 non-goals | No MorphServe, quantization, layer swapping, KV resizing, new scheduler, admission heuristic, forecasting, or mitigation changes; scheduler audit remains clean | PASS |
| Run implementation checks | `unittest benchmark.test_benchmark`: 6 tests OK; `py_compile`: PASS; `git diff --check`: PASS; raw completion/config/artifact audit: PASS | PASS |

## Final decision

The deliverables are complete for the selected practical workload. The experiment demonstrates an RPS knee and compute-bound saturation, but it does **not** demonstrate the reference memory/KV saturation phenomenon. That negative result is intentional and is reported without overclaiming; the next controlled experiment is documented in `docs/phase-1/archive/short-workload-saturation-report.md`.
