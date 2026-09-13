# Closed-loop runtime adaptation v9 completion audit

## Objective restated as concrete deliverables

The repository must answer from actual same-process measurements whether the validated v7 crossover and v8 state-preserving morph substrate combine into the core runtime-adaptation trade-off. Completion requires:

1. a pre-result frozen primary controller using exact v7 time-weighted entry/release semantics and a separate 1,768-block policy reference;
2. same-runtime-envelope FP16, initialized-AWQ, and causal Dynamic controls;
3. immutable low, high, and low→high→low workloads with two fresh runs per cell;
4. raw request, scheduler, physical KV, batch, controller, transition, state, and per-token precision evidence;
5. regenerated latency/queue/KV/throughput/transition/quality/fidelity outputs and timelines;
6. direct validation of request/KV/capacity integrity, open-loop arrivals, controller behavior, reversibility, stability, and HBM evidence;
7. separate systems and quality–latency decisions that retain negative results without retuning; and
8. copy-pasteable raw execution/regeneration commands plus an independent completion audit.

A scientific GO is not required for task completion: a protocol-valid NO-GO is a valid answer. Missing evidence is not.

## Prompt-to-artifact checklist

### Phase A — controller frozen before results

| Explicit requirement | Evidence inspected | Audit status |
|---|---|---|
| Primary `sustained_compound_pressure`; no replacement by secondary | `docs/closed-loop-runtime-v9/protocol.md`; `protocol_manifest.json/controller`; `closed_loop_controller.py` | PASS |
| Trailing 3 s: KV>=.95 for >=80%, wait>=4 for >=50%, current wait>=4 | `closed_loop_controller.py:evaluate`; unit replay in `test_closed_loop_controller.py`; per-sample `controller.jsonl` | PASS |
| Exact v7 time weighting, not sample counting | direct imports of `telemetry_segments`, `window_stats`, `counter_value_at` from frozen `analyze_crossover.py`; helper hash in manifest/audit | PASS |
| Policy denominator 1,768; physical base 1,759 remains separate | manifest; metadata runtime envelope; every telemetry row has policy/base/physical/visible fields | PASS |
| Release: full trailing 10 s, wait mean<=.5, zero preemption, used/1768<=.70 for >=90% | controller source/tests; `controller.jsonl`; phased last-window inspection | PASS |
| Fixed 0.25-s cadence | manifest/runner; sample index/deadline/jitter in raw controller/telemetry; 12,486 samples | PASS |
| Explicit latched state machine/no duplicate transition | controller source/tests; transition count and ordered event rows | PASS |
| No cooldown/minimum residence/post-result special case | protocol/source hash; no such rule in controller | PASS |
| Prohibited controller inputs absent | manifest metadata input-exclusion list; controller accepts only causal pressure sample fields | PASS |

Temporal evidence: protocol commit `168a27d` precedes every v9 run. Every raw metadata file records that commit, clean diff SHA-256 `e3b0c442...`, identical manifest SHA-256 `34f27564...`, and identical source hashes.

### Phase B — fair same-runtime controls

| Explicit requirement | Evidence inspected | Status |
|---|---|---|
| All arms prepare runtime morphing in same process layout | all 18 `metadata.json/engine_config_actual`; identical source provenance | PASS |
| Runtime-static FP16 stays FP16 at ~1,759 | static FP16 telemetry/transitions; 1,759 stable capacity and zero transitions | PASS |
| Runtime-static AWQ morphs before clock and holds safe 4,170 | `initialization_transitions.jsonl`, measurement timestamps, AWQ telemetry; zero measured transitions | PASS |
| Dynamic starts FP16 and charges every runtime transition | controller/transition/request monotonic timestamps; measured duration and TTFT | PASS |
| No old v7 number substitutes for controls | all comparisons in `analysis/paired_performance.csv` resolve v9 selected runs | PASS |

### Phase C — pre-registered workloads/repeats

| Explicit requirement | Evidence inspected | Status |
|---|---|---|
| Low scale 5.75, high scale 4.25, exact 64 contents/1024/512/schedule per arm | input files, source and output SHA-256 values, workload metadata, run metadata hashes | PASS |
| High point not replaced | immutable `run-plan.json`; exactly scale-4.25 file in all six high cells | PASS |
| Low→high→low deterministic concatenation, no artificial gap | `input/low-high-low.jsonl`; boundaries 0/100.625/175.0; natural cohort spans documented | PASS |
| Recovery long enough to test frozen release while requests remain | 90.5625-s recovery arrivals; observed rule still became true only at/after tail | EXECUTED; scientific gate failed |
| At least two fresh processes per condition/workload | exact 3×3×2 status mapping; 18 unique processes/runs | PASS |
| Duplicated phased prompts not treated as independent quality units | analyzer uses only 64 unique headline questions and averages repeats before bootstrap | PASS |

### Phase D — controller/transition telemetry

| Required evidence | Concrete raw field/artifact | Status |
|---|---|---|
| monotonic time; controller/runtime states | `controller.jsonl:timestamp_ns,elapsed_s,controller_state,runtime_precision_state` | PASS |
| waiting/running/swapped | controller and telemetry queue fields | PASS |
| actual physical used/free/total and scheduler capacity | physical/scheduler/base/extension fields every tick | PASS |
| fixed-reference and native utilization | `fp16_equivalent_kv_utilization`, `native_current_state_kv_utilization` | PASS |
| trailing entry/release statistics and booleans | entry/release fields every controller tick | PASS |
| request/start/end, pending/drain, result/error | controller samples plus full `transitions.jsonl` engine traces | PASS |
| arrivals continue during transition | absolute launcher; request timestamps/jitter; zero observed transition-overlap arrivals due natural cohort spacing, not source pause | PASS |
| transition stall included | hot intervals fall after measurement start and intersect request service; no clock exclusion | PASS |
| per-request prefill/output precision, exposure, transition span, answer/reference | 512-element precision/position/time arrays and derived fields in `requests.jsonl` | PASS |
| all existing batch/scheduler/KV traces | 59,353 rows in `batches.jsonl`, 12,486 periodic samples | PASS |

### Phase E — performance comparison

| Requirement/gate | Evidence | Outcome |
|---|---|---|
| P50/P95/P99 TTFT, queue, TPOT, prefill→output | `analysis/serving_runs.csv` | PASS |
| strict TTFT>2 s | serving table and headline table | PASS |
| request/token throughput | serving table | PASS |
| queue/running/swapped/KV distributions+dwell | serving table from raw time segments | PASS |
| swaps/preemptions, batch/effective-M distributions | serving table/batches | PASS |
| precision occupancy by wall/forward/request/token | serving table/headline table | PASS |
| transition/pending/stall contribution | `transition_cost_contribution.csv` | PASS |
| frozen run-level noise formula | manifest/protocol/decision/paired table | PASS |
| low zero false entries and FP16-like latency | both Dynamic low runs; paired table | PASS |
| high causal entry and P95 improvement > noise in both | advantages 8.193/7.263 s > epsilon 0.776 s | PASS |
| high queue/KV/preemption/throughput mechanism gates | `decision.json/performance_gates` | FAIL: fixed-reference KV dwell repeat 1 and mean-throughput clause |
| Dynamic-versus-static-AWQ gap quantified | -16.326/+2.969 s, with static-AWQ variance and trigger/hot costs reported | PASS |

### Phase F — quality/fidelity

| Requirement | Evidence | Outcome |
|---|---|---|
| Existing DuReader character-overlap F1 on same headline requests | raw answers/references; `quality_repeat_values.csv`; `quality_fidelity_summary.csv` | PASS |
| Paired request uncertainty with explicit repeat aggregate | 64-question repeat-first aggregate; fixed 10,000-sample percentile bootstrap | PASS |
| position agreement/common prefix/exact match | `quality_request_pairs.csv`, summary | PASS |
| Dynamic AWQ-token fraction/exposure groups | requests and serving table | PASS |
| Token agreement not called semantic quality | protocol/report/table metric-role text | PASS |
| Strong semantic trade-off | Dynamic-AWQ +0.939 pp, CI [-1.582,+3.819] | UNRESOLVED; strong gate failed |

### Phase G — reversibility/stability

| Requirement | Evidence | Outcome |
|---|---|---|
| initial low remains FP16 | both phased controller traces; entry after 100.625 s | PASS |
| high causes one FP16→AWQ | requests at 133.260/110.010 s, successful transitions | PASS |
| recovery causes AWQ→FP16 under exact rule | repeat 0 none; repeat 1 request 295.769 s | FAIL in repeat 0 |
| restore before final recovery arrival/useful subsequent service | final arrival 265.5625 s; repeat 1 restore ends 297.721 s; no later arrival | FAIL both repeats |
| post-restore correctness | final active request emits seven contiguous FP16 steps in repeat 1 | PARTIAL PASS |
| no oscillation | one transition / one round trip | PASS |
| no loss/re-prefill/invalid positions/capacity error/OOM/remap failure | all 1,920 requests and all transition integrity checks | PASS |
| repeated restored-HBM bound | only one phased restore | NOT ESTABLISHED; gate fails |

### Phase H — final outputs and decisions

| Named deliverable | Path | Status |
|---|---|---|
| Three-row headline table | `analysis/headline_table5.csv`; final report | PASS |
| High latency/queue/KV/state timeline | `analysis/headline_latency_queue_kv_state_timeline.png` | PASS |
| Phased controller timeline | `analysis/phased_controller_timeline.png` | PASS |
| Paired quality/fidelity table | `analysis/quality_request_pairs.csv`, `quality_fidelity_summary.csv` | PASS |
| Controller trigger audit | `analysis/controller_trigger_audit.csv` | PASS |
| Transition-cost contribution | `analysis/transition_cost_contribution.csv` | PASS |
| Raw regeneration scripts/commands | `regenerate.sh`, `execution_commands.json` | PASS |
| Completion audit | `completion_audit.json`, this document | PASS |
| Independent raw numerical recomputation and hashes | `independent_verification.json` (103/103) | PASS |
| Required final finding/uncertainty/decision sections | `final-report.md` | PASS |

### Scope prohibitions and historical preservation

| Requirement | Evidence | Status |
|---|---|---|
| No W4-8/W4-32, alternate backend, scheduler, kernel, LIS, model/dataset, extra KV policy | manifest scope; exact source diff; all metadata state/backend/model fields | PASS |
| Preserve v2/v4/v5/v6/v7/v8 namespaces | pre-result tree SHA-256 map in manifest equals final independent audit | PASS |
| Do not retune after results | source/protocol/workload hashes remain those recorded by every run; no extra result run | PASS |

## Verification commands actually run

```bash
# CPU suites in both validated environments (31 tests, 5 expected CUDA skips each)
PYTHONPATH="$AWQ_EXT:$PWD/swiftLLM" "$AWQ_ENV/bin/python" -m unittest \
  benchmark.test_benchmark benchmark.test_inference_substrate \
  benchmark.test_crossover benchmark.test_runtime_morphing \
  benchmark.test_closed_loop_controller -v

# Opt-in v8 segmented-KV/runtime CUDA preflight (9/9 pass)
MORPHSERVE_RUN_CUDA_TESTS=1 CUDA_VISIBLE_DEVICES=5 \
  PYTHONPATH="$AWQ_EXT:$PWD/swiftLLM" "$AWQ_ENV/bin/python" \
  -m unittest benchmark.test_runtime_morphing -v

# Frozen 18-run matrix
CUDA_VISIBLE_DEVICES=5 PYTHONPATH="$AWQ_EXT:$PWD/swiftLLM" \
  "$AWQ_ENV/bin/python" -m benchmark.run_closed_loop_plan ... --resume

# Raw regeneration and independent audit
benchmark-results/closed-loop-runtime-v9/regenerate.sh

git diff --check
```

The final regeneration logs are under `benchmark-results/closed-loop-runtime-v9/logs/`. Responsibilities are explicit rather than overstated: the frozen analyzer derives all tables/gates from raw streams; the frozen completion audit verifies preregistration/history/workload hashes, lineage, raw structure/state/capacity integrity, analysis coverage, and report sections; and the post-result independent verifier hashes every selected raw file and separately recomputes request arrays, arrivals, capacity/reference invariants, controller windows, decisive serving metrics, performance gates, F1/bootstrap CI, fidelity, phased reversibility, headline values, and both decisions without changing any frozen rule.

## Missing, weak, or uncovered requirements

No requested artifact or measurement is missing. The following are **measured failed/uncertain scientific criteria**, not hidden completion gaps:

- useful restoration failed in both phased repeats;
- fixed-reference high-pressure KV dwell did not improve in repeat 1;
- Dynamic mean completed throughput was 0.18% lower than FP16;
- Dynamic semantic superiority over static AWQ was unresolved;
- two-repeat phased restored-HBM evidence could not be formed because one run never restored.

The audit does not use tests, a manifest, or a green raw validator as a proxy for those outcomes; each failed gate remains false in `analysis/decision.json` and the report.

## Completion conclusion

The repository now answers the objective from actual one-process closed-loop measurements. It confirms causal no-entry/entry, state-preserving pressure relief, and repeated high-load latency benefit, but rejects the complete predeclared systems and quality–latency claims because useful restoration and semantic separation were not demonstrated. No required follow-up run is allowed within this frozen final experiment.
