# Throughput confirmation v11 completion audit

Status: **artifact/completion PASS; scientific THROUGHPUT CONFIRMATION DECISION: NO-GO**.

A passing completion audit means the requested experiment, evidence, and decision were produced and independently checked. It does not convert failed scientific gates into GO.

## Objective restated as checkable deliverables

1. Preserve v10 byte-for-byte and retain its RELEASE-SIDE SYSTEMS DECISION: NO-GO.
2. Add observation-only controller/window/trace, telemetry lateness, event-loop, launch, output-consumer, GPU, and HBM timing without changing decisions.
3. Prove exact archived v7/v9/v10 policy parity before serving.
4. Preregister and execute 12 counterbalanced low-only FP16/Dynamic pairs; use paired log-ratio NI inference against 0.98 and attribute controller/tail variability.
5. Apply the frozen Phase B branch, permitting at most one exact-parity optimization only when its preconditions hold; freeze the selected implementation before Phase C.
6. Preregister and execute eight one-cycle and eight two-cycle FP16/Dynamic pairs without early stopping or v10 reuse.
7. Verify entry/release/drain/post-restore/state-order/chatter/request/KV/capacity/OOM behavior for every run.
8. Report workload-specific throughput inference, measurement/token ratios, TTFT/TPOT, controller/jitter/environment metrics, exact finite-tail decomposition, and long-run HBM behavior.
9. Produce raw/regeneration commands, independent audits, final evidence categories, and a GO/NO-GO decision without a semantic-quality experiment.

## Prompt-to-artifact checklist

| Explicit requirement | Concrete artifact/evidence | Audit result |
|---|---|---|
| V10 remains unchanged and NO-GO | Phase A/C manifests hash both v10 trees; Phase A and Phase C audits recompute hashes; `analysis/final_decision.json` and final report retain v10 NO-GO | PASS |
| Entry policy unchanged | Source hashes include frozen `closed_loop_controller.py`; 27,217-sample parity; no source edit to controller | PASS |
| `quiet_pressure_drop_5v10` unchanged | Source hash, protocol constants, exact v10 replay | PASS |
| No threshold/window/cadence change | Protocols/manifests record exact values; fixed 0.25-s grid; raw metadata confirms | PASS |
| AWQ-Marlin/layers unchanged | Manifests/raw engine config: AWQ-Marlin decoder `[0,16)` | PASS |
| Capacities 1,759/4,170 and reference 1,768 | Raw metadata/telemetry; independent capacity audit over all 56 runs | PASS |
| Scheduler, segmented KV, hot transition unchanged | V10 runner is delegated unchanged; source hashes; final CUDA 9/9 tests | PASS |
| Checkpoint/request/decode unchanged | Workload/model hashes and raw metadata; every request is 1,024 prompt / exact 512 greedy outputs | PASS |
| Observation-only evaluate wall/CPU | `instrumented_release_intent_controller.py`; every Dynamic controller row; `analysis/phase_*/controller_overhead.json` | PASS |
| Window construction/integration timing | Per-sample wall/CPU fields and history/segment lengths; aggregate files and reports | PASS |
| Controller trace bookkeeping | Decision/history construction and `trace_fields` timed; unchanged outer row copy explicitly disclosed as unseparated lower bound | PASS with declared measurement boundary |
| Telemetry lateness | Existing `sampling_jitter_s`; P95/P99 in `serving_runs.csv` | PASS |
| Event-loop lag | Same 50-ms probe in both conditions; `event_loop_lag.jsonl`; P95/P99 analysis | PASS |
| Request-launch jitter | Existing raw `arrival_jitter_s`; P95/P99 analysis | PASS |
| Output-consumer lag where measurable | First-output→first-receipt and completion→stream-completion fields/analysis; intermediate tokens explicitly not claimed | PASS |
| Instrumentation cannot alter actions | Timings are outputs only; exact original/instrumented decision equality | PASS |
| Archived v7/v9/v10 parity before results | `archived_policy_parity.json`: 36/6/6 streams, 48 total, 27,217 samples, PASS; prereg commit contains artifact | PASS |
| Phase A exact v10 low workload/process | Workload SHA-256 `f8cedda3...`; same runtime wrapper; raw metadata | PASS |
| Phase A 12 paired counterbalanced blocks | `phase-a-run-plan.json`: 24 cells, six AB/six BA; status 24/24 | PASS |
| V10 excluded from primary statistic | Protocol/manifests/confidence artifact explicitly exclude v10 | PASS |
| Paired log NI method fixed before results | `protocol.md` and Phase A manifest at `447c049`; alpha 0.025, margin 0.98, Student t fixed | PASS |
| Measurement/token throughput, P95/P99 TTFT, TPOT | `analysis/throughput_ratio_table.csv`, phase serving/pair tables | PASS |
| GPU clocks/temperature/power | 56-run NVML evidence and `hbm_environment_summary.json` | PASS |
| Controller and jitter attribution | Phase A report and controller-overhead artifacts | PASS |
| Phase B at most one optimization | Trigger false because two-sided upper ratio 1.002739; no optimization performed | PASS |
| Exact parity required before optimization | Branch not entered; no optimized evaluator/result exists | N/A, correctly not invoked |
| Freeze implementation before Phase C | Phase A result/freeze `6f14431`; Phase C preregistration `cbf0edf`; all Phase C metadata clean at `cbf0edf` | PASS |
| Phase C ≥6, preferably 8 pairs/workload | `phase-c-run-plan.json`: eight one-cycle and eight two-cycle pairs; 32/32 cells | PASS |
| Counterbalanced fresh processes | Four AB/four BA per workload; every raw metadata clean/fresh; no replacement/early stop | PASS |
| Workload-specific inference; no pooled rescue | `confidence_analysis.json`: separate low/one/two bounds; intersection-union rule | PASS |
| No false low entry | 11/12 low-only and 7/8 one-cycle preserve; `low-04` and `one-07` failures retained | SCIENTIFIC FAIL |
| One entry per high/useful release | All two-cycle and 7/8 one-cycle exact; `one-07` has two extra low entries/round trips | SCIENTIFIC FAIL |
| Real drain above base | Nine Phase C above-base restores all record `drain_required`; all legality/capacity checks pass | PASS |
| Post-restore low requests execute FP16 | All expected normal releases pass; `one-07` extra releases fail usefulness and are retained | SCIENTIFIC FAIL |
| Multi-cycle order/no chatter | 8/8 two-cycle exact; 7/8 one-cycle exact; one chatter run | SCIENTIFIC FAIL |
| No loss/re-prefill/KV/OOM | 9,728 requests, one prefill each, 4,980,736 outputs, all IDs/positions/digests/capacities valid, no OOM | PASS |
| Final planned arrival/completion/drain/tail share | Pair-level `tail_decomposition.csv`; combined `tail_summary.json`; exact zero-error closure | PASS |
| HBM retained; monotonic growth checked | Per-run telemetry, transition CSV, combined environment summary; no monotonic trend; one 512-KiB difference disclosed | PASS |
| Do not call cache variation a leak | Final report explicitly labels non-monotonic variation non-leak with capacity/KV evidence | PASS |
| No larger-N semantic-quality experiment | No quality workload/analysis in either plan | PASS |
| Controller-overhead source quantified | Low/one/two means about 1.96/10.11/23.54 s, dominated by full-history window work | PASS |
| Transition/drain/tail/other attribution | Final report reports hot/pending/drain and exact tail components; avoids additive causal overclaim | PASS |
| Raw regeneration commands | `execution_commands.json`, exact attempt ledgers/logs, executable `regenerate.sh` | PASS |
| Required final headings and decision | `final-report.md` ends with all five required sections and `THROUGHPUT CONFIRMATION DECISION: NO-GO` | PASS |

## Actual validation evidence

- Phase A execution: 24/24 selected fresh-process runs; no serving attempt replacement.
- Phase C execution: 32/32 selected fresh-process runs; no serving attempt replacement.
- Raw provenance: all Phase A runs clean at `9dc8d54`; all Phase C runs clean at `cbf0edf`.
- Archived parity: PASS, 48 streams / 27,217 samples.
- Phase A artifact audit: PASS, 12/12 checks.
- Phase C artifact audit: PASS, 14/14 checks.
- Regeneration: PASS; final scientific decision remains NO-GO.
- CPU suite: 44 tests passed, five CUDA-only cases skipped there.
- CUDA suite: 9/9 passed separately on GPU 5, including all five segmented-KV cases.
- Python compilation: PASS for benchmark/server/worker modules.
- `git diff --check`: PASS.

The first attempted Phase A command used the wrong working directory and failed workload path validation before model initialization or run-directory creation. The traceback and command are preserved, the minimal command-only repair is documented in `doc/debug-report.md`, and all 24 serving cells subsequently ran at the clean repaired commit. Two audit-only corrections are disclosed; both preserve raw selection, controller/analyzer outputs, guardrail values, and decisions.

## Coverage assessment

No explicit deliverable is absent. The requested scientific GO conditions are not all satisfied, which correctly produces NO-GO rather than an incomplete artifact. The known measurement boundary for outer dictionary-copy bookkeeping and the absence of intermediate-token producer timestamps are explicitly bounded by “where measurable”; first/final consumer lag and controller-owned trace construction are measured. Remaining scientific questions are listed in the final report and are not silently treated as achieved.
