# Release-side runtime v10 completion audit

## Objective restated as concrete deliverables and success criteria

The repository must determine from preserved v7/v9 development traces plus new preregistered held-out serving runs whether v9's AWQ→FP16 failure was caused by late release detection and whether one causal release-intent/drain design restores useful FP16 service without altering or destroying the confirmed FP16→AWQ mechanism.

Completion requires:

1. exact v9 release-delay reconstruction and honest non-additive attribution;
2. at most two causal candidates and one frozen PRIMARY policy before results;
3. separation of release intent from unchanged 1,759-block allocator safety;
4. deterministic held-out one- and two-cycle workloads with continuing low arrivals and different content assignment;
5. two fresh Dynamic and matched FP16 repeats per workload, with useful static-AWQ controls;
6. exact intent/drain/hot-restore/resume/catch-up, request/KV/capacity/HBM, and GPU-environment evidence;
7. useful post-restore FP16 requests and exact repeated state order without chatter/corruption;
8. preregistered noise/non-inferiority rather than v9's brittle mean-throughput/KV-dwell clauses;
9. descriptive answer retention without semantic-significance gating;
10. raw scripts, analyzers, independent raw audit, final report, and explicit RELEASE-SIDE SYSTEMS DECISION.

Scientific GO is not required for task completion; a valid NO-GO is. The formal GO requires every frozen gate, not merely successful restoration.

## Prompt-to-artifact checklist

### Fixed prior evidence and scope

| Requirement | Evidence inspected | Status |
|---|---|---|
| V9 remains systems/quality NO-GO | unchanged `docs/closed-loop-runtime-v9/final-report.md`, v10 protocol/report | PASS |
| Preserve entry thresholds/window | v10 controller subclasses/calls v9 `SustainedCompoundController.evaluate`; unit boundary test; six held-out entries | PASS |
| Preserve AWQ-Marlin, layers `[0,16)`, checkpoint, segmented KV, capacities, scheduler ordering | protocol manifest source hashes; raw metadata/config/transitions; scheduler/model files unchanged from run commit | PASS |
| Preserve v2–v9 artifacts | frozen tree hashes for 15 result/doc namespaces; completion audit direct comparison | PASS |
| Do not alter high regime merely for result | scale-4.25 arrival-template hash in workload metadata/manifest | PASS |

### Phase A — v9 release latency diagnosis

| Explicit requirement | Concrete evidence | Status |
|---|---|---|
| Every v9 Dynamic high/phased trajectory | `development/v9_release_trajectory.csv` | PASS: four runs, every post-AWQ sample |
| Waiting trajectory/time-weighted area | trajectory plus `waiting_queue_area_post_awq_request_s` in decomposition | PASS |
| Preemption history | trajectory counters/recent deltas | PASS |
| Scheduler-used and physically allocated blocks | trajectory fields | PASS |
| `/1768` utilization and 1,759 margin | trajectory fields | PASS |
| Old rule first true | decomposition CSV/JSON: 98.751/98.751/censored/295.751 s | PASS |
| Physical shrink first feasible | exact entry context: feasible at AWQ completion in all four | PASS |
| Candidate first fire | 86.001/85.000/199.001/199.751 s | PASS |
| Quantify 10-s persistence | standalone ~10.24 s; ordered incremental contribution 0 at actual release | PASS |
| Quantify 0.70 threshold | first-crossing and release-producing-episode lags, up to 175.568 s after feasibility | PASS |
| Quantify 90% dwell | ~9 s after release-producing <=.70 episode | PASS |
| Quantify waiting/preemption | jointly ready 8.983–11.721 s after stable AWQ | PASS |
| Quantify physical drain feasibility | zero v9 pre-request delay; prospective v10 drain measured separately | PASS |
| Do not use replay as validation | explicit warning in JSON/diagnosis/protocol/report | PASS |
| Use v7 evidence as development context | `development/v7_operating_regimes.csv` | PASS |

The audit explicitly rejects additive causal attribution because threshold/dwell/window conditions interact.

### Phase B — two-stage contract and protocol freeze

| Requirement | Evidence | Status |
|---|---|---|
| At most two candidates | development JSON `candidate_count: 1` | PASS |
| Exact PRIMARY frozen | `quiet_pressure_drop_5v10` constants/source/protocol/manifest | PASS |
| Protocol commit precedes results | commit `8e7675f`; every raw run records that clean commit | PASS |
| Only causal existing observations | waiting, preemption counter, scheduler-used/1,768 recent/prior means | PASS |
| No prohibited online inputs | controller signature/source and manifest exclusion list | PASS |
| Avoid naive old rule | no `<=.70`/90%-10-s actuation in v10 controller; old stats retained only by superclass telemetry | PASS |
| Intent may precede physical feasibility | five raw intents at +85/+88/+85/+88/+284 blocks | PASS |
| Reuse, do not duplicate, allocator safety | Engine `_mark_restore_feasible`; controller never reads physical usage | PASS |
| Pause prefill/swap-in; continue decode | scheduler source, lifecycle and 17–29 decode forwards/22 requests in drains | PASS |
| Shrink/remap/publish/resume ordering | exact lifecycle timestamps and KV traces; capacity publication precedes resume | PASS |
| No sensitivity substitution | no sensitivity policy/run/result | PASS |

### Phase C — held-out reversibility workloads and matrix

| Requirement | Evidence | Status |
|---|---|---|
| Low regime scale 5.75; high scale 4.25 | input metadata template paths/hashes | PASS |
| One low→high→low workload | 192-row `input/heldout-one-cycle.jsonl` | PASS |
| At least two cycles | 320-row `input/heldout-two-cycle.jsonl` | PASS |
| Deterministic different content slice/offset | source rows 0–63 vs v9 42–105; hashes frozen | PASS with disclosed 22-question overlap |
| Exact 1024/512 shape | input/unit/raw audit | PASS |
| No artificial idle gaps | natural span boundaries in metadata/tests | PASS |
| Continuing final-low arrivals | each low has 64 requests and 90.5625-s arrival span | PASS |
| Two fresh PRIMARY repeats per alternating workload | four Dynamic raw dirs, unique fresh processes | PASS |
| Matched FP16 controls | four alternating FP16 raw dirs | PASS |
| Static AWQ useful for interpretation | four alternating AWQ raw dirs; excluded from primary release trigger | PASS |
| Low-only false-entry control | two Dynamic + two FP16 runs | PASS |
| Counterbalanced immutable matrix | 16-cell `run-plan.json`; all attempt 0 | PASS |

### Phase D — mandatory release/drain telemetry

| Required field/evidence | Raw/derived location | Status |
|---|---|---|
| release condition first true and restore request | transition event + `transition_timeline.csv` | PASS |
| Engine receipt/admission pause | `engine_trace.restore_lifecycle` + timeline | PASS |
| allocated/scheduler blocks and ±1,759 at intent | release observation + timeline | PASS |
| drain start/end and first physical legality | lifecycle + timeline | PASS |
| arrivals during pause | raw request interval recomputation; timeline (all zero) | PASS |
| requests continuing decode | batch IDs/forwards; timeline | PASS: 22 in all five real drains |
| hot restore start/end | model/controller trace + timeline | PASS |
| capacity publication/admission resume | lifecycle + timeline | PASS |
| queue during drain/pause | telemetry integral/peak; timeline | PASS: zero under natural cohort timing |
| catch-up time/area | exact three-second zero-wait definition; timeline | PASS |
| subsequent re-entry | transition sequence/next-entry fields | PASS |
| request IDs/one prefill/positions/KV integrity | raw requests/batches/transitions; run validation and audit | PASS |
| physical/scheduler capacities, HBM, transition memory | telemetry/trace/timeline/decision HBM spread | PASS |
| GPU clocks/temp/power | 5,087-row `gpu_environment.jsonl` collection; environment summary | PASS |

### Phase E — success gates

| Gate | Evidence/outcome | Status |
|---|---|---|
| No false low-only AWQ entry | zero transitions in both | PASS |
| AWQ entry under every high | six of six, exactly one per high | PASS |
| Preserve high benefit | P95 benefit 8.017–10.364 s > frozen epsilon; queue area lower; preemptions no greater | PASS |
| No request/KV corruption | 3,328/3,328 complete; all raw validation checks | PASS |
| Causal intent after pressure subsides | all six in following low phase; no online phase input | PASS |
| Safe restore completes | six of six | PASS |
| Meaningful low service remains | 51 arrivals and >62 s to last arrival after every resume | PASS |
| Post-restore requests execute FP16 | 51/51 per exit; 306 total | PASS |
| No loss/re-prefill | one prefill and 512 contiguous steps each | PASS |
| No immediate oscillation | next entries only 82.392/83.119 s later after next high; none otherwise | PASS |
| Drain/restore not obviously worse | pause/catch-up bounds and matched recovery comparison pass all six | PASS |
| Exact multi-cycle state order | both `FP16→AWQ→FP16→AWQ→FP16` | PASS |
| Throughput 0.98 non-inferiority | low rep0 0.97778; one rep1 0.97774; one aggregate 0.97944 | **FAIL** |
| Strict >2-s rate retained but not sole gate | `serving_runs.csv`/phase tables | PASS |
| Retired v9 brittle clauses not reused | protocol/analyzer decision | PASS |

Every direct useful-recovery gate passes, but the frozen overall systems GO fails on throughput non-inferiority.

### Phase F — semantic boundary

| Requirement | Evidence | Status |
|---|---|---|
| Preserve generated answers | every raw request has decoded answer/output IDs | PASS |
| Do not require semantic significance | protocol/manifest/decision excludes F1 | PASS |
| Retain descriptive quality | `analysis/quality_descriptive.csv` | PASS |
| Do not reinterpret v9 quality result | final report keeps semantic power question blocked | PASS |

### Required named outputs

| Deliverable | Path | Status |
|---|---|---|
| v9 release-delay decomposition | `docs/.../release-delay-diagnosis.md`; `development/v9_release_delay_decomposition.{csv,json}` | PASS |
| frozen release-intent specification | `docs/.../protocol.md`; `protocol_manifest.json` | PASS |
| held-out manifests | `input/*.jsonl`, `input/workload_metadata.json`, `run-plan.json` | PASS |
| per-transition timeline | `analysis/transition_timeline.csv` | PASS |
| post-restore counts/FP16 proof | `analysis/post_restore_requests.csv` | PASS |
| multi-cycle timeline | `analysis/state_timeline.csv`, `multi_cycle_state_timeline.json` | PASS |
| latency/queue/preemption comparison | `analysis/matched_phase_comparison.csv`, `phase_metrics.csv` | PASS |
| environmental GPU telemetry | per-run `gpu_environment.jsonl`, `gpu_environment_summary.csv` | PASS |
| raw regeneration scripts | `run_release_plan.py`, `regenerate.sh`, `execution_commands.json` | PASS |
| completion audit | `completion_audit.json`, this document | PASS |
| required final sections/decision | `final-report.md` | PASS |

## Analysis-correction audit

The preregistered analyzer's percentile calls used fractions against an API requiring percentages. The bug was found only after all runs. `analysis_correction.json` records preregistered/corrected hashes, both logs, and confirms:

- no raw rerun;
- no controller, threshold, workload, margin, selection, or decision-rule change;
- phase percentile calls changed from `.50/.95/.99` to a regression-tested `50/95/99` helper;
- the audit changed to bound/disclose the repair and independently recompute every decisive scientific gate from raw;
- preliminary and corrected overall decisions are both NO-GO.

Raw metadata continues to match the preregistered source map. Final audit verifies the exact protocol/manifest existed in the clean run commit and permits only those two declared post-result source differences.

## Commands actually run

```bash
# 42 CPU-safe tests in the AWQ environment (5 expected CUDA skips)
PYTHONPATH="$AWQ_EXT:$PWD/swiftLLM" "$AWQ_ENV/bin/python" -m unittest \
  benchmark.test_benchmark benchmark.test_inference_substrate \
  benchmark.test_crossover benchmark.test_runtime_morphing \
  benchmark.test_closed_loop_controller benchmark.test_release_intent_controller \
  benchmark.test_release_runtime_telemetry benchmark.test_release_workloads \
  benchmark.test_release_analysis -v

# 9/9 opt-in CUDA runtime/segmented-KV tests
MORPHSERVE_RUN_CUDA_TESTS=1 CUDA_VISIBLE_DEVICES=5 \
  PYTHONPATH="$AWQ_EXT:$PWD/swiftLLM" "$AWQ_ENV/bin/python" \
  -m unittest benchmark.test_runtime_morphing -v

# Frozen 16-run matrix, all attempt 0
CUDA_VISIBLE_DEVICES=5 PYTHONPATH="$AWQ_EXT:$PWD/swiftLLM" \
  "$AWQ_ENV/bin/python" -m benchmark.run_release_plan ... --resume

# Corrected analysis and final regeneration/audit
benchmark-results/release-side-runtime-v10/regenerate.sh

git diff --check
```

Full commands are in `execution_commands.json`; logs are retained under `logs/`.

## Missing, incomplete, weak, or uncovered requirements

No requested artifact or held-out run is missing. The following are measured limitations rather than hidden completion gaps:

- throughput non-inferiority fails narrowly, so formal GO is unavailable;
- only two fresh processes per cell limit run-noise inference;
- 22 held-out questions overlap v9 because only 106 frozen source rows exist;
- no arrival overlapped a drain under natural cohort timing, although active-request drain was exercised five times;
- driver HBM varied by up to 302 MiB while model allocated bytes/capacity were identical;
- the post-result percentile correction is transparently bounded but weakens preregistration ideality;
- generalization beyond this trace/model/checkpoint/GPU/request shape remains unmeasured;
- semantic significance remains intentionally blocked.

The 17/17 executable audit independently reconstructs all six release ticks and every decisive systems gate from raw before comparing `decision.json`. It does not treat a green validator, manifest, transition count, or successful restore as a proxy for the entire objective; it directly retains the failed throughput gate and formal NO-GO.

## Completion conclusion

The repository can now determine the narrow question from actual held-out full round trips rather than by lengthening v9 until its old rule fires. The new intent repeatedly repairs direct release usefulness and exercises v8 drain while preserving entry/high-pressure benefit and state. The formal preregistered overall decision remains NO-GO solely because the 0.98 throughput non-inferiority margin is missed in two individual comparisons and the one-cycle aggregate.
