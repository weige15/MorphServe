# Closed-loop runtime adaptation v9 protocol

Status: **pre-registered before any v9 closed-loop serving result** on 2026-09-13 UTC.

## Scientific question and fixed scope

This final reproduction experiment asks whether one causal controller can keep the same active SwiftLLM process in FP16 under low pressure, morph it to the validated AWQ-Marlin W4-16 state under sustained pressure, and restore FP16 after pressure subsides, while producing a better measured quality–latency operating point than either same-envelope static state alone.

The only runtime states are `{FP16, AWQ-Marlin W4-16}`. AWQ means decoder layers `[0,16)` from the existing AutoAWQ W4/G128/asymmetric-zero-point artifact through the existing vLLM 0.11.2 Marlin path; all other weights remain FP16. The model, tokenizer, 1,024-token prompts, exact 512-step greedy outputs, strict-FCFS scheduler, block size 16, batch/token limits, segmented KV implementation, RTX 3090 device 5, software environment, and artifacts are unchanged from v7/v8.

No W4-8/W4-32 state, alternate backend, new scheduler, kernel tuning, LIS profiling, new model/dataset, additional KV policy, or post-result threshold/workload/transition change is allowed. Existing v2/v4/v5/v6/v7/v8 files remain in their namespaces. All v9 files live under `benchmark-results/closed-loop-runtime-v9/`, `docs/closed-loop-runtime-v9/`, and new benchmark-only modules.

## Frozen primary controller

The sole actuating policy is v7 `sustained_compound_pressure`. The separately qualified `capacity_margin_queue_integral` rule is not actuated and is not used to replace an unfavorable primary result.

At each causal 0.25-second pressure tick, the controller records current/past runtime state and uses the exact piecewise-constant, time-weighted integration helpers from `benchmark.analyze_crossover`. A telemetry row applies from its monotonic timestamp until the next row; threshold dwell is elapsed-time weighted, never sample-count weighted.

### Policy reference versus physical safety

The policy reference is frozen at **1,768 FP16 blocks**, exactly the capacity under which v7 qualified the signal. Entry and release use `scheduler logical used blocks / 1768`. The v8-capable process's actual base is expected to be 1,759 blocks and its safe AWQ total is fixed at 4,170 blocks. Actual allocator used/free/total blocks, physical segments, and scheduler visibility are separately recorded and remain the only basis for v8 transition safety. The 1,768 policy denominator never authorizes physical allocation.

### Entry

While stable in FP16, request FP16→AWQ when, over the full causal trailing three seconds:

- `used / 1768 >= 0.95` for at least 80% of elapsed time;
- waiting depth `>=4` for at least 50% of elapsed time; and
- current waiting depth is `>=4`.

### Release

After an entry has been requested and while stable in AWQ, request AWQ→FP16 only at or after entry-request time plus ten seconds and when the full causal trailing ten-second window has:

- time-weighted waiting mean `<=0.5`;
- zero new preemptions; and
- `used / 1768 <=0.70` for at least 90% of elapsed time.

Native AWQ utilization is logged but never drives release.

### State machine and prohibited inputs

The explicit states are `FP16`, `ENTRY_REQUESTED`, `MORPHING_TO_AWQ`, `AWQ`, `RELEASE_REQUESTED`, `DRAINING_OR_RESTORING`, and `FAILED`. Exactly one transition may be latched. Repeated ticks cannot issue duplicates or a conflicting target. The existing Engine public APIs serialize and service transitions only between complete forwards. Restoration may hold new admission/swap-in while active KV naturally drains to the 1,759-block base; it does not discard or re-prefill requests.

There is no added cooldown, minimum residence, debounce, retry, post-result special case, or threshold. After a completed restore, the same frozen entry rule can fire again if its trailing window genuinely becomes true; extra transitions are reported as instability.

Controller inputs exclude workload class/scale/RPS, future arrivals, completion outcomes, final TTFT, DuReader answers/F1, generated tokens, and offline crossover labels. Request launches continue on their absolute open-loop deadlines during hot transitions.

## Same-runtime-envelope controls

Every condition initializes the same runtime-capable process with runtime morphing enabled, prepares both pinned weight variants before KV profiling, uses the same binary/code, and pays the same staging/layout overhead.

1. **RUNTIME-STATIC-FP16:** start and remain in runtime FP16, expected physical capacity 1,759; no controller or transition request.
2. **RUNTIME-STATIC-AWQ-W4-16:** initialize runtime FP16, manually morph and expand to exactly 4,170 blocks before warmup and before the measurement clock, then remain AWQ. This setup transition is saved as initialization evidence and excluded from request latency.
3. **CLOSED-LOOP-DYNAMIC:** begin measured service in runtime FP16 and actuate only the frozen controller. Every post-clock transition, drain, and stall is charged.

Runtime KV hash sampling is enabled for every transition. Its diagnostic cost is included in Dynamic's hot transition latency.

## Frozen workloads

The three inputs and hashes are in `benchmark-results/closed-loop-runtime-v9/input/workload_metadata.json`; the immutable matrix/order is `run-plan.json`.

### Low-pressure negative control

`input/low-scale-5p75.jsonl` reproduces the v7 scale-5.75 trace: 64 fixed questions, source sequences 42–105, last arrival 90.5625 s, approximately 0.696 requests/s. Dynamic is expected to make zero transitions. This is a false-positive test, not a tuning opportunity.

### High-pressure headline

`input/high-scale-4p25.jsonl` reproduces the preselected v7 scale-4.25 trace: the same 64 questions and generation policy, last arrival 66.9375 s, approximately 0.941 requests/s. It lies in v7's independently validated contiguous 0.889–1.000 requests/s AWQ-preferred region and cannot be replaced after results.

### Low→high→low reversibility

`input/low-high-low.jsonl` concatenates one exact scale-5.75 template, one exact scale-4.25 template, and one exact scale-5.75 template. Each next phase starts one existing source-trace cohort interval after the preceding phase's final cohort: low span 100.625 s, high span 74.375 s, high starts at 100.625 s, recovery starts at 175.0 s, and the last recovery arrival is 265.5625 s. No artificial idle gap is inserted. The 192-row input uses unique benchmark IDs but intentionally reuses the same 64 question contents; duplicated phases are not independent quality samples.

All three conditions receive byte-identical v9 workload files. There are exactly two fresh-process repeats per condition/workload, for 18 planned cells. The two passes use the predeclared counterbalanced order. Attempt 0 uses the planned run ID; after a preserved failed/incomplete directory, resume uses `-attemptN`, records planned→actual lineage in `execution_status.json`, and selects exactly one valid attempt per cell. Failed attempts and logs remain present and are never silently overwritten or analyzed as a successful repeat.

## Fixed engine and measurement envelope

- Physical GPU: RTX 3090 device 5, `CUDA_VISIBLE_DEVICES=5`.
- Environment: torch 2.9.0+cu128, vLLM 0.11.2, Transformers 4.51.3, existing torch-2.9 SwiftLLM extension.
- GPU memory utilization 0.99; 4,096 CPU blocks; block size 16.
- Maximum batch 32; maximum 49,152 input tokens/batch; 128 sequence IDs; 3,072 blocks/sequence.
- Seed 2025; one excluded eight-token warmup; exact 512 greedy output steps; timeout 7,200 s.
- Controller and telemetry cadence 0.25 s on absolute monotonic deadlines.
- Arrival integrity gate: sequence-ordered actual arrivals, exact planned/actual timestamp formulas, finite jitter, and absolute launch jitter no greater than one telemetry period (0.25 s). A larger delay means the source was effectively paused and invalidates the run rather than being tuned away.
- Every run is a fresh process. No old v7 static number substitutes for a v9 control.

Raw files per run are `metadata.json`, `requests.jsonl`, `telemetry.jsonl`, `batches.jsonl`, `controller.jsonl`, `transitions.jsonl`, `initialization_transitions.jsonl`, `runtime_preparation.json`, and `summary.json`.

## Telemetry and state-preservation evidence

Every pressure/controller tick records monotonic time; controller and runtime state; waiting/running/swapped counts; admission hold; scheduler and actual allocator used/free/total blocks; base/extension/visible capacities; fixed-reference and native utilization; exact three-/ten-second statistics; entry/release booleans; transition latch/request/start/end/pending/result/error fields; counters; and HBM.

Every forward retains v7 batch, KV, queue, prefill/effective-M, duration, state, and swap/preemption observation. Every request retains answer/reference/output IDs and exact per-output-step precision, input position, receipt time, stable engine request ID, prefill state, exposure fraction, and transition overlap. Actual arrivals are checked against hot transition intervals; the launch source is never paused to hide a stall.

A valid request has exactly one 1,024-token prefill path evidenced by exactly 512 model-output forwards, contiguous input positions 1,023–1,534, one stable engine request ID, exactly 512 valid output token IDs, and completion. It must also contain 512 allowed per-step precision labels, positions, engine IDs, and monotonic receipt timestamps whose recomputed FP16/AWQ counts, first/prefill state, and compressed precision transitions match the saved derived fields. Transitions must report success, the exact target precision, expected physical/scheduler capacity, and equal sampled logical KV digests before/after resize. An active-request transition without a digest is invalid; a digest-free transition is accepted only when the trace explicitly reports zero active requests.

## Frozen raw-derived reporting

Regenerate for every run:

- TTFT, queueing, TPOT, first-prefill→first-output, and arrival-jitter P50/P95/P99 plus maximum absolute arrival jitter;
- strict `TTFT >2 s` violation count/rate;
- completed request and generated-token throughput;
- waiting/running/swapped distributions, waiting area/dwell, fixed-reference/native KV distributions and >=0.95 dwell;
- preemption/swap counts;
- forward/prefill sequence, token/effective-M, duration, and decode-size distributions;
- precision occupancy by wall time, forwards, prefills/requests, and output tokens;
- transition request/start/end, pending/drain/hot-path cost, affected requests/arrivals, admission-hold time, and direct TTFT-interval overlap (reported as mechanism overlap, not a causal semantic metric).

The headline outputs are a three-row Table-5-style high-pressure table, performance pairs, quality/fidelity tables, high and phased timelines with transition markers, controller audit, transition-cost contribution, decision JSON, raw regeneration commands/script, and completion audit.

## Predeclared run-noise and performance gates

For Dynamic versus runtime-static FP16, separately at each workload, freeze:

`epsilon = max(0.111 s, static-FP16 two-repeat P95 range, Dynamic two-repeat P95 range, 0.05 * static-FP16 P95 mean)`.

Fresh serving run is the independent latency unit; request bootstrap does not replace two-run variability.

### Low-pressure gate

Both Dynamic low repeats must issue zero transitions, and each matched Dynamic P95 TTFT must be within `±epsilon` of runtime-static FP16. Any entry is a policy failure.

### High-pressure gate

Both Dynamic headline repeats must complete FP16→AWQ. In each matched repeat, runtime-static-FP16 P95 minus Dynamic P95 must be strictly greater than `epsilon`. Dynamic must also have lower P95 queueing, lower time-weighted waiting area, and lower fixed-reference KV>=0.95 dwell in both repeats; no more preemptions in either and fewer in at least one; at least 98% of FP16 completed throughput in each repeat and no lower two-repeat mean throughput. Dynamic is not required to beat always-AWQ latency; the gap, trigger delay, pending time, and measured hot transition are reported.

### Reversibility/stability gate

Each phased Dynamic repeat must remain FP16 before high-phase start, perform exactly one ordered FP16→AWQ transition during/after sustained high pressure, request restore only after recovery begins under the frozen ten-second rule, complete AWQ→FP16 before the final recovery arrival, and complete at least one subsequently arriving recovery request. All 192 requests must finish with contiguous positions and no loss/re-prefill evidence. More than this one round trip is instability. Every transition must preserve sampled KV and succeed; capacities must remain consistent; no CUDA OOM/remap error is allowed. Restored equal-state allocated-HBM values across the two phased repeats must differ by no more than the already used 64-MiB v8 meaningful-leak tolerance.

## Quality/fidelity analysis and decisions

For the 64 headline questions, compute existing DuReader character-overlap F1 against the best reference. Report each serving repeat separately. The aggregate first averages each question across its two serving repeats, then macro-averages the 64 unique questions. Paired percentile bootstrap uncertainty resamples only those 64 unique question aggregates (10,000 samples, fixed seeds). Duplicated phased prompts never become extra independent quality units.

Against matched runtime-static FP16, report position-aligned output-token agreement, common-prefix length, exact-output match, Dynamic AWQ-token fraction, and never/partially/fully AWQ-exposed request counts. These are fidelity/mechanism evidence, not semantic quality.

**SYSTEMS ADAPTATION GO** requires all 18 raw runs valid, both low gates, every high gate, successful causal transitions/KV preservation, both useful phased round trips, no oscillation, and bounded restored HBM spread. Otherwise it is **NO-GO**; no threshold, workload, or implementation is retuned to rescue it.

**QUALITY–LATENCY STRONG GO** additionally requires the systems GO and a strictly positive lower bound of the paired 95% unique-question bootstrap CI for Dynamic-minus-static-AWQ DuReader F1. If systems GO holds but that semantic difference is unresolved, the decision is **SUPPORTED / UNCERTAIN**. If systems adaptation is NO-GO, quality–latency is **NO-GO**.
