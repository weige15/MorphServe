# MorphServe throughput confirmation v11

Status: **complete — adaptation workloads satisfy the preregistered 0.98 paired non-inferiority inference, but the all-workload throughput confirmation is NO-GO because low-only remains just unresolved and repeated runs exposed false low-phase entry/chatter**.

V10 is untouched and remains **RELEASE-SIDE SYSTEMS DECISION: NO-GO** under its original every-repeat/two-repeat rule. V11 is a separate 56-process confirmation experiment; no v11 observation is pooled into or used to reinterpret v10.

## Question and frozen design

V11 tested whether v10's roughly 0–2.2% completed-throughput misses were repeatable closed-loop cost or two-process finite-run noise. It retained the exact v10 state pair, sustained-compound entry, `quiet_pressure_drop_5v10` release, 0.25-second grid, thresholds/windows, AWQ-Marlin `[0,16)`, 1,759/4,170 physical capacities, 1,768 policy reference, strict FCFS, segmented KV, admission drain, hot transitions, checkpoints, 1,024/512 request shape, greedy decoding, workloads, and RTX 3090 GPU 5.

Phase A fixed 12 low-only FP16/Dynamic paired blocks. Phase B preregistered an optimization only if the paired result showed a repeatable nonzero deficit **and** measured controller work explained at least 25% of positive duration excess. That conjunction failed, so no optimization was made. Phase C then fixed eight one-cycle and eight two-cycle pairs. All 28 pairs ran to the planned count without early stopping; V10 observations were excluded.

The primary estimand for each workload is the geometric mean of fresh-process paired Dynamic/FP16 completed-throughput ratios. The fixed analysis is a paired Student t interval on log ratios, one-sided alpha 0.025 (97.5% lower bound), with non-inferiority only when the lower bound is strictly above 0.98. The global claim is conjunctive; pooling cannot rescue a workload.

## Exact policy-parity audit

Before serving, `archived_policy_parity.json` compared the instrumented evaluator with the current v10 evaluator for all decision fields over:

- all 36 archived v7 streams;
- all six archived v9 Dynamic streams; and
- all six archived v10 Dynamic streams.

Coverage was 48 streams and 27,217 fixed-grid samples. Every instrumented/original boolean and requested action matched. On v10, the replay also matched every archived decision row and action sample/timestamp. The timing fields are outputs only.

## Primary throughput results

| workload | pairs | ratio range | geometric mean D/F | log-ratio SD | 97.5% one-sided lower | p vs 0.98 | NI |
|---|---:|---:|---:|---:|---:|---:|---|
| low-only | 12 | 0.94844–1.01962 | **0.990763** | 0.018911 | **0.978930** | 0.03535 | **FAIL** |
| one-cycle | 8 | 0.97276–0.99628 | **0.987017** | 0.008470 | **0.980053** | 0.02435 | PASS |
| two-cycle | 8 | 0.98519–0.99231 | **0.989464** | 0.002519 | **0.987383** | 0.00000646 | PASS |

Low-only misses the bound by 0.001070. Its two-sided 95% upper ratio is 1.002739, so it does not establish a repeatable nonzero deficit either. One- and two-cycle means show repeatable nonzero costs—their two-sided 95% upper ratios are 0.994031 and 0.991550—but those costs are about 1.30% and 1.05%, and both are statistically non-inferior to the unchanged 2% margin.

The individual Phase C ratios were:

- one-cycle: 0.985276, 0.996275, 0.994077, 0.990343, 0.992985, 0.977123, 0.987545, 0.972761;
- two-cycle: 0.990154, 0.986969, 0.992307, 0.989027, 0.985187, 0.992173, 0.991042, 0.988877.

Every run completed the same planned count and 512 outputs per request, so generated-token-throughput ratios equal completed-request-throughput ratios. `analysis/throughput_ratio_table.csv` reports every completed throughput, measurement-duration ratio, generated-token ratio, P95/P99 TTFT, and TPOT value.

This resolves the v10 two-repeat question asymmetrically: the adaptation-workload 0.98 misses were too noisy to establish a ≥2% cost, but low-only remains narrowly unresolved at the stronger preregistered 97.5% bound rather than becoming confirmed.

## Controller-overhead attribution

Low-only Dynamic controller evaluation plus measured trace work consumed 1.554–2.409 s per process (mean 1.964 s), or 1.32–1.87% of measured duration. Exact window construction/integration dominated. A post-result replay calibration measured timing instrumentation overhead at only about 0.020 s per 487-sample replay (instrumented/original ratio 1.0335), far below the serving-time totals.

The same full-history rebuilding/scanning grew with run horizon:

| workload | mean Dynamic duration | mean measured controller wall | fraction of duration |
|---|---:|---:|---:|
| low-only | about 120 s | 1.964 s | about 1.6% |
| one-cycle | 296.023 s | 10.112 s | 2.85–3.91% per run |
| two-cycle | 470.959 s | 23.536 s | 4.83–5.18% per run |

Window work was 8.23–11.39 s in one-cycle and 22.30–24.13 s in two-cycle. This is a real, horizon-growing synchronous control-plane cost. It is large enough to explain the measured duration excess in magnitude, but it is not an additive causal mediation: GPU/model execution occurs through an executor, AWQ service gains overlap it, and longer tails cause more samples and therefore more evaluation work.

The Phase B optimization trigger was frozen before Phase A. Although measured accounting exceeded its 25% materiality threshold, low-only did not meet the required repeatable-deficit condition. Consequently no post-result optimization was allowed. The exact rolling-window optimization remains an unexecuted future option, not evidence used here.

## Finite-horizon tail decomposition

For every pair, the final planned arrival, final actual arrival, last stream completion, and measurement end exactly decompose duration into planned horizon, launch timing, completion tail, and post-completion runner time. All log decompositions close with zero recorded numerical error.

| workload | final planned arrival | FP16 post-last-planned drain, median (range) | Dynamic drain, median (range) | median signed fraction of log difference from completion tail |
|---|---:|---:|---:|---:|
| low-only | 90.5625 s | 30.562 s (26.430–32.019) | 30.316 s (26.970–38.399) | 1.0011 |
| one-cycle | 265.5625 s | 26.044 s (25.276–30.652) | 29.526 s (27.008–38.927) | 0.9993 |
| two-cycle | 440.5625 s | 25.259 s (23.619–28.348) | 30.194 s (27.812–33.631) | 1.0006 |

Thus essentially all signed fixed-workload throughput difference appears in completion-tail duration; arrival timing and post-completion bookkeeping are tiny and offsetting. This does not identify whether the tail accumulated from controller blocking, transition stalls, admission drains, or service variation, and it does not redefine the primary metric.

Measured transition components help bound alternatives. One-cycle Dynamic averaged 5.10 s of hot-transition time because the chatter run made six transitions instead of two; two-cycle averaged 6.53 s across the expected four. Restore legality drain averaged 0.33 s per one-cycle run and 2.45 s per two-cycle run. These costs are partially offset by AWQ high-pressure service gains. The observable throughput mechanism is therefore finite completion tail with material controller-window and hot-transition contributors—not request loss or arrival jitter.

## Scheduling, latency, and environment

The raw tables retain P95/P99 TTFT and mean/P95 TPOT for every run. Dynamic retained substantially lower one-/two-cycle P95 TTFT than FP16 in most pairs while paying longer TPOT/finish tails. P99 event-loop and telemetry lateness were generally small but had process-level excursions: Phase A Dynamic maxima were 545/634 ms, and Phase C Dynamic maxima were 518/589 ms. FP16 Phase C maxima were 49/52 ms. Launch-jitter P99 stayed below 67 ms. First/final output-consumer lag is separately retained; it is not conflated with model TPOT.

NVML succeeded in all 56 processes. Across Phase C, mean SM clocks were 1,670/1,683 MHz (FP16/Dynamic one-cycle) and 1,654/1,679 MHz (two-cycle); mean temperatures were 69.31/69.04°C and 70.09/69.43°C; mean power was 266.26/261.93 W and 267.51/261.68 W. These overlaps do not support a condition-specific thermal explanation.

## HBM and allocator sanity

All samples retained HBM and capacity telemetry. Driver-used end HBM varied non-monotonically across fresh processes and transitions: examples include 20,171–21,849 MiB end values. It did not grow monotonically with ordinal or cycle. Twenty-six Phase C restores included nine real above-base drains; all nine waited for legality and all completed. Restore model-allocated bytes were 20,070,251,520 in 26 of 27 v11 restores and 20,070,775,808 in the final second-cycle restore, a one-time 512-KiB difference without a third-cycle trend, capacity change, digest error, or repeat across the preceding seven two-cycle runs.

No monotonic driver or model growth is established. The 1,759/4,170 capacities, allocator legality, and KV digests remained valid, so the observed non-monotonic cache/allocator variation is not labeled a leak.

## Preserved systems and reversibility audit

Across 56 fresh processes:

- 9,728/9,728 requests and 4,980,736/4,980,736 output steps completed;
- every request had exactly one prefill, a stable Engine ID, 512 precision/timestamp positions, contiguous positions, and no error;
- all 54 transitions succeeded with KV digest/capacity integrity;
- every process ended FP16 with 1,759 blocks;
- all eight two-cycle Dynamic runs followed exact `FP16 -> AWQ -> FP16 -> AWQ -> FP16` order;
- seven of eight one-cycle runs followed the expected single round trip;
- eleven of twelve low-only Dynamic runs made zero transitions.

Two real stability failures prevent preservation from passing:

1. Phase A `low-04` entered at 92.5 s during the low-only completion tail (waiting 10, scheduler-used 1,701), then restored at 110.0 s. Arithmetic and transitions were correct, but the zero-transition low-only guardrail failed.
2. Phase C `one-07` made the expected high-phase round trip, then entered twice more during recovery low at 217.268 and 267.283 s, with releases at 253.514 and 285.274 s. The extra sequence caused post-restore usefulness failure and is chatter/false low-phase entry under the fixed offline audit.

These runs are not excluded. They are exactly the instability a larger fresh-process confirmation was intended to reveal. Importantly, no request loss, duplicate prefill, KV corruption, illegal shrink, OOM, or terminal state failure accompanied them.

## Provenance, corrections, and reproduction

- Phase A preregistration: `447c049`; command-only root-path correction before any serving: `9dc8d54`.
- Phase A result/unchanged Phase B freeze: `6f14431`.
- Phase C preregistration commit: `cbf0edf`.
- All 24 Phase A raw runs record clean commit `9dc8d54`; all 32 Phase C raw runs record clean commit `cbf0edf`.
- The pre-measurement path-validation failure created no run directory and is preserved/diagnosed.
- Two bounded audit-only corrections are disclosed. Neither changed raw runs, selections, controller, analyzer, inference, thresholds, guardrail values, or scientific decisions.
- Raw and regeneration commands are in `benchmark-results/throughput-confirmation-v11/execution_commands.json` and `regenerate.sh`.
- Independent artifact audits pass 12/12 Phase A and 14/14 Phase C checks while preserving the scientific failures.

## CONFIRMED FINDINGS

- One- and two-cycle closed-loop throughput is non-inferior to FP16 at the fixed 0.98 margin under the preregistered paired-log analysis: lower bounds are 0.980053 and 0.987383.
- Their mean costs are nevertheless repeatable and nonzero at about 1.30% and 1.05%; this is below, not equal to, a confirmed ≥2% deficit.
- Full-history controller window recomputation is measurable, horizon-growing, and dominates controller time (about 2, 10, and 24 seconds over low/one/two-cycle runs).
- Fixed-workload throughput differences are almost entirely completion-tail differences; arrival timing and final bookkeeping are negligible.
- Every request, transition, KV digest, and physical/scheduler capacity remained correct; all processes restored to FP16/1,759.
- No monotonic HBM growth or leak is established.
- Larger-N execution exposed one low-only false entry and one one-cycle chatter run that two v10 repeats did not reveal.

## SUPPORTED BUT UNCERTAIN FINDINGS

- Low-only's center estimate is only a 0.92% deficit and its interval includes no deficit, so ordinary tail/scheduling variability is plausible; its 97.5% lower bound of 0.978930 still does not establish the required 0.98 NI claim.
- Controller blocking is large enough to explain duration deficits in magnitude, but overlap with GPU execution, AWQ service gains, and the endogenous number of tail samples prevents a clean additive causal estimate.
- The one-time 512-KiB second-restore allocator difference is not a repeated trend, but only two transitions per normal two-cycle process limit long-horizon leak sensitivity.

## BLOCKED QUESTIONS

- Whether an exact rolling-window/deque evaluator removes the measured horizon-growing cost without changing any action remains untested because the preregistered Phase B optimization trigger did not authorize it.
- The cause of the rare large event-loop/tail excursions is not uniquely identified as OS scheduling, Python/GIL contention, controller scanning, or another process-level effect.
- Generalization beyond this checkpoint, request shape, workload family, and RTX 3090 remains outside scope.
- No larger-N semantic-quality experiment was performed.

## REMAINING UNCERTAINTY

- The low-only lower bound is only 0.001070 below the margin and remains statistically unresolved rather than credibly ≥2% inferior.
- The one-cycle NI lower bound exceeds 0.98 by only 0.000053 and is sensitive to the fixed 97.5% small-sample inference.
- Completion-tail attribution identifies where the makespan difference appears, not a complete causal partition of controller, transition, drain, and service-time mechanisms.
- Rare false low-phase entries/chatter now have an observed rate but not a reliable population frequency from 12/8 Dynamic runs.

## THROUGHPUT CONFIRMATION DECISION: NO-GO

The larger paired experiment rejects the interpretation that v10 demonstrated a repeatable ≥2% adaptation-throughput cost: one- and two-cycle workloads pass the unchanged 0.98 margin, with measured mean costs around 1%. However, the required global claim fails because low-only does not clear its preregistered lower bound, controller evaluation shows persistent horizon-growing cost, and repeated confirmation reveals false low-pressure entry/chatter. Those are explicit NO-GO conditions. V10 remains independently and permanently **NO-GO** under v10's own rule.
