# Phase-1 Final Completion Audit

Audit date: 2026-09-10 UTC

## Objective restatement

Complete the final SwiftLLM Phase-1 experiment using the authoritative calibrated
KV-stress configuration, with unquantized FP16 Llama 3.1 8B on CUDA device 3,
open-loop fixed arrivals, and offered RPS as the only independent variable.
Preserve raw request and telemetry data, demonstrate a load-dependent KV
admission transition with queueing and a P95 TTFT knee, separate compute from
queueing where timestamps permit, regenerate all result artifacts from raw data,
and report the result without confusing it with the historical 3880-block
short-workload baseline or changing `scheduler.py`.

## Prompt-to-artifact checklist

| Requirement / success criterion | Current evidence inspected | Status |
|---|---|---|
| Read `baseline.md` before work | File exists and was read at the start of this continuation | PASS |
| Read `archive/benchmark-harness-report.md` before work | File exists and was read at the start of this continuation | PASS |
| Read `archive/short-workload-saturation-report.md` before work | File exists and was read at the start of this continuation | PASS |
| Read `archive/completion-audit.md` before work | File exists and was read at the start of this continuation | PASS |
| Read authoritative `calibration.md` before work | File exists and was read at the start of this continuation | PASS |
| Read `../../references/MORPHSERVE_PHASE1_REFERENCE.md` when present | File exists and was read at the start of this continuation | PASS |
| Preserve original validated baseline facts | `baseline.md`, `archive/short-workload-saturation-report.md`, and Section A of `final-report.md`: original EngineConfig, 3880 blocks, and 62,080 token slots | PASS |
| Preserve the original short-workload compute result as separate evidence | Section A of `final-report.md`: approximately 35 RPS plateau at approximately 0.206% logical KV utilization, no swaps/preemptions, compute-bound classification | PASS |
| Use the calibrated fixed configuration without recalibration | Every final metadata record and Section B of `final-report.md`: block 16, utilization 0.99, profiled 1768 GPU blocks, CPU 4096, sequence table 128, max blocks 3072, batch 32, batch-token budget 49152 | PASS |
| Use exact 2048-token prompts and 16 requested output tokens | Eight final `metadata.json` files and all final `requests.jsonl`: observed prompt 2048 and output 16 for every completed request | PASS |
| Use fixed arrival mode and calibration seed policy | Final metadata: `arrival_mode=fixed`, `random_seed=2025`, no warmup, telemetry interval 0.25 s | PASS |
| Keep model, GPU, scheduler, and workload settings fixed | `aggregate.json` verification: `fixed_configuration_consistent=true`; all eight final metadata records agree on fixed fields; only target RPS and derived schedule window/run ID differ | PASS |
| Use open-loop arrivals independent of completion | `swiftLLM/benchmark/run.py` and `workload.py`; final metadata planned offsets; audit checked a 2-RPS later arrival occurred before the first request completed | PASS |
| Establish an underloaded point | 0.5 and 0.75 RPS final runs: all 20 completed, no KV-pressure samples; 0.75 is selected as the closest below-transition representative | PASS |
| Increase RPS monotonically with coarse sweep | Canonical points 0.5, 0.75, 1, 1.25, 1.5, 2 RPS; pressure first appears at 1 RPS | PASS |
| Add fine points below, at, and above the knee | 0.75 below, 1.0 onset/near, 1.25 and 1.5 above, with 2 RPS overload confirmation | PASS |
| Repeat important knee-region points | `phase1-final-repeat-rps-1` and `phase1-final-repeat-rps-2`; raw and summary values are included in `aggregate.json` and plotted as repeat markers | PASS |
| Stop after obvious overload rather than arbitrary higher RPS | Sweep stopped at 2 RPS after 25 KV-pressure telemetry samples, waiting peak 16, and completed throughput near 1.09 RPS versus 2 offered | PASS |
| Preserve raw machine-readable request data at every point | Each of eight source directories in `aggregate.json` contains `metadata.json`, `requests.jsonl`, `telemetry.jsonl`, and `summary.json`; 160/160 requests completed | PASS |
| Preserve raw telemetry and physical HBM data | Final telemetry rows contain queue, block, logical KV, swap/preemption, and `torch.cuda.mem_get_info` fields; counts are checked in the audit command output | PASS |
| Compute offered/achieved arrival and completed throughput separately | `sweep.csv`, Section C request table, and per-run summaries contain target/achieved arrival, completed RPS, and output-token throughput as separate fields | PASS |
| Compute TTFT mean/P50/P95/P99 | `sweep.csv`, `summary.json`, and Section C request table | PASS |
| Compute queueing mean/P50/P95 | `sweep.csv`, `summary.json`, and Section C request table | PASS |
| Compute TPOT mean/P95/P99 | `sweep.csv`, `summary.json`, and Section C request table | PASS |
| Compute waiting/running/swapped queue statistics and peaks | `sweep.csv` has complete distributions and peaks; Section C telemetry table includes mean/P50/P95/P99/max | PASS |
| Compute decoding blocks, total GPU blocks, logical KV utilization and peak | `sweep.csv`, `admission_evidence.csv`, raw telemetry, and Section C telemetry table | PASS |
| Do not infer HBM saturation from latency alone | Section B and C use logical KV occupancy and admission condition; HBM is explicitly supplementary | PASS |
| Demonstrate KV admission blocking while other limits are slack | Final raw pressure samples satisfy `num_decoding_gpu_blocks + 128 > 1768`; pressure rows have running=13, waiting>0, and the evidence checks max-batch, batch-token, per-sequence, and sequence-table slack | PASS |
| Demonstrate the required ordering of load, KV pressure, waiting, queueing, and TTFT | `admission_evidence.csv`, `sweep.csv`, `representative_time_series.csv`, and Section C: no pressure below 0.75; pressure at 1+; queueing/TTFT rise through 2 RPS | PASS |
| Inspect representative raw time series below/near/above | Representatives in `aggregate.json`: below `phase1-final-rps-0p75`, near `phase1-final-rps-1`, above `phase1-final-rps-1p25`; raw rows preserved in `representative_time_series.csv` and plotted in `representative_time_series.png` | PASS |
| Examine queueing versus post-admission/prefill TTFT | Raw timestamp decomposition in `decomposition.csv`; `rps_vs_ttft_components.png`; Section C reports tokenization, queueing, post-admission, and delivery components | PASS |
| Classify memory/KV versus compute honestly | Section C and D classify the final result as mixed KV+compute: direct KV admission blocking materially raises queueing, while 2048-token prefill compute is also material | PASS |
| Generate primary RPS vs P95 TTFT plot from saved data | `benchmark-results/phase1-final/plots/rps_vs_p95_ttft.png`; source series in `aggregate.json.plot_data` | PASS |
| Generate RPS vs P95 queueing plot from saved data | `.../rps_vs_p95_queueing.png`; source series in `aggregate.json.plot_data` | PASS |
| Generate RPS vs peak logical KV plot from saved data | `.../rps_vs_peak_logical_kv_utilization.png`; source series in `aggregate.json.plot_data` | PASS |
| Generate RPS vs completed throughput plot from saved data | `.../rps_vs_completed_throughput.png`; source series in `aggregate.json.plot_data` | PASS |
| Generate TTFT decomposition diagnostic | `.../rps_vs_ttft_components.png` and `decomposition.csv`, generated from raw timestamps | PASS |
| Regenerate summaries, tables, plots, and report from raw files | `final_analyze.py` rewrites summaries using `summarize_run`, derives CSV/JSON/plots, and writes `docs/phase1/final-report.md`; aggregate verification says summary regeneration and plot-source checks PASS | PASS |
| Independently recompute selected request metrics and percentiles | `request_audit.json` stores all request timestamp recomputations and selected first/last IDs; `aggregate.json` stores independent TTFT/queueing/TPOT percentile checks for all eight runs | PASS |
| Verify repeatability | Target 1 and 2 repeats reproduce 1768 blocks, 1677 peak decoding blocks, 94.85% peak KV utilization, and matching pressure queue ranges; request-level percentiles are preserved in `request_audit.json` | PASS |
| Produce `final-report.md` sections A/B/C/D | File exists and contains baseline, fixed configuration/admission proof, final sweep tables/plots/evidence, and explicit conclusion/comparison policy | PASS |
| Preserve qualitative-only MorphServe comparison | Section D says qualitative mechanism only and makes no numerical paper reproduction claim | PASS |
| Keep `swiftLLM/swiftllm/server/scheduler.py` unchanged | `git diff --exit-code -- swiftLLM/swiftllm/server/scheduler.py` passed; current and `git show HEAD:` SHA-256 are both `80e2142c5a7bab300aad43e10af110043d429fd9d1655e5942f5073f99305c6d` | PASS |
| Avoid MorphServe, quantization, layer swapping, KV resizing, new admission, scheduler, forecasting, or mitigation changes | Final command uses existing benchmark/Engine path and unchanged scheduler; code diff contains no such implementation | PASS |
| Verify plots can be opened | `PIL.Image.verify()` passed for all six PNGs; `file` identifies each as a valid PNG | PASS |
| Run tests and code gates | `python -m unittest benchmark.test_benchmark -v`: 6 tests OK; `py_compile` for benchmark/server files: PASS; `git diff --check`: PASS | PASS |
| Separate 1768-block final results from historical 3880-block results | Final aggregate has eight source runs and all metadata report 1768; historical `benchmark-results/phase1-saturation` remains separate and its baseline points are not included in the final aggregate/report | PASS |
| Record residual uncertainty rather than overclaim | Section C documents 20-request finite-tail sampling, GPU/Ray timing variability, prefill compute cost, and no swap requirement | PASS |

## Current-state verification output

The final audit command wrote `benchmark-results/phase1-final/completion_audit.txt`.
Its observed results are:

- Required input documents: all six present.
- Eight final runs: all have 20 completed requests, zero failed/incomplete.
- Fixed configuration consistency: PASS; profiled capacity is 1768 blocks / 28,288 token slots.
- Independent timestamp percentile checks and raw regeneration: PASS.
- Representative states: below has 0 KV-pressure samples; near has 4; above has 6.
- Image-open checks: all six PNGs PASS.
- Scheduler current/checkpoint hashes: identical.
- Unit tests: 6/6 PASS.
- Python compilation and `git diff --check`: PASS.
- Historical/final 3880-vs-1768 separation: PASS.

## Final decision

The checklist has no missing or weakly verified required item. The final
experiment is complete: offered RPS increased from an underloaded regime to a
reproducible KV-admission pressure regime with waiting-queue and queueing-delay
growth and a P95 TTFT knee. The defensible bottleneck classification is mixed
KV+compute, not pure memory-only saturation. No final result is mislabeled as
the original 3880-block baseline, and no scheduler or mitigation change was
made.
