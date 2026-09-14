# MorphServe release-side runtime v10

Status: **complete — repeatable useful release is demonstrated, but the frozen overall RELEASE-SIDE SYSTEMS DECISION is NO-GO because the preregistered 0.98 throughput non-inferiority margin narrowly fails**.

V9 remains a frozen systems and quality–latency NO-GO. V10 changed only AWQ exit intent. It retained the exact v7/v9 entry rule, AWQ-Marlin W4-16 layers `[0,16)`, 1,759/4,170 physical capacities, 1,768 policy reference, segmented KV, allocator/shrink implementation, FCFS ordering, checkpoint, request shape, and all v2–v9 artifacts.

The release mechanism itself is a strong positive result. Every one of four held-out alternating Dynamic runs restored usefully; both two-cycle repeats produced `FP16 -> AWQ -> FP16 -> AWQ -> FP16`; five of six restores latched above the FP16 base and exercised a real admission hold/drain; all six completed while 51 later low-phase requests remained, and every one of those requests prefetched and executed FP16. No chatter, loss, re-prefill, KV corruption, capacity error, or OOM occurred. High-phase P95 TTFT retained an 8.017–10.364-second advantage over matched FP16.

The formal decision nevertheless remains NO-GO. The frozen 0.98 throughput margin failed in low-only repeat 0 (`0.97778`) and one-cycle repeat 1 (`0.97774`); the one-cycle two-repeat ratio was `0.97944`, only 0.00056 below the bound. The low-only miss occurred with zero transitions, which makes it likely run-duration noise rather than a release defect, but a preregistered gate is not waived after results.

## Preregistration and analysis correction

- Protocol commit before all 16 runs: `8e7675f` (`research(protocol): preregister release-side runtime v10`).
- Every raw metadata file records full commit `8e7675f6...`, a clean tree, the same manifest hash `41d3853...`, and the complete preregistered source-hash map.
- All workloads, conditions, thresholds, margins, phase boundaries, and run order were fixed in that commit.
- All 16 planned attempt-0 processes completed; there were no replacement runs.

A post-result **analysis-only correction** is disclosed in `benchmark-results/release-side-runtime-v10/analysis_correction.json`. The preregistered analyzer passed `0.50/0.95/0.99` to a percentile helper whose API expects `50/95/99`. Raw runs, controller, workloads, gates, and selection were unchanged; the corrected phase P95 values are used below. The preliminary and corrected overall decisions are both NO-GO, though the corrected high-phase preservation gate passes. The audit allows only the declared analyzer/audit hash changes and verifies every actuation source against the run commit.

## V9 release-delay decomposition

The full development replay is `docs/release-side-runtime-v10/release-delay-diagnosis.md` and `benchmark-results/release-side-runtime-v10/development/`.

| v9 Dynamic run | AWQ stable s | wait/preempt ready s | first physical feasibility s | first `used/1768<=.70` s | release-producing/terminal <=.70 episode s | old rule s | PRIMARY replay s |
|---|---:|---:|---:|---:|---:|---:|---:|
| high rep0 | 69.530 | 81.251 | 69.530 | 89.751 | 89.751 | 98.751 | 86.001 |
| high rep1 | 48.226 | 58.501 | 48.226 | 89.751 | 89.751 | 98.751 | 85.000 |
| phased rep0 | 134.018 | 143.001 | 134.018 | 200.501 | 286.251 | censored at 87.50% dwell | 199.001 |
| phased rep1 | 110.932 | 120.001 | 110.932 | 201.001 | 286.500 | 295.751 | 199.751 |

Barrier interpretation:

1. The ten-second residence barrier consumed about 10.24 seconds but added **0 seconds incrementally** at eventual old release because later conditions dominated.
2. First 0.70 crossings lagged exact physical feasibility by 20.221/41.524/66.484/90.069 seconds.
3. In phased runs, the release-producing below-0.70 episode did not begin until 152.233/175.568 seconds after first feasibility.
4. The 90% dwell then added about nine seconds.
5. Waiting/preemption became jointly ready only 8.983–11.721 seconds after AWQ stabilized and was nonbinding later.
6. Actual v9 restore request-to-hot-start intervals were only 0.0419/0.0073/0.0050 seconds; no observed v9 drain explains the late request.

These components overlap and are not presented as an additive causal mediation. The decisive observation is that the old controller waited for policy pressure far below physical safety.

## Frozen release-intent contract

The sole v10 release candidate is `quiet_pressure_drop_5v10`. After the first stable-AWQ controller sample and 15 seconds of AWQ-only history, intent requires:

- current waiting `==0`;
- trailing-five-second time-weighted waiting mean `<=0.5`;
- zero new preemptions in five seconds;
- current scheduler-used/1,768 `<=1.25`; and
- mean scheduler-used/1,768 in the recent five seconds `<=0.80` times its mean in the preceding ten seconds.

The rule uses no physical allocation, phase/workload label, scale/RPS, future arrival, result, TTFT, quality, or token. Physical safety remains solely in v8: pause prefills/swap-ins, keep active decode running, wait for allocated `<=1,759`, shrink/remap, restore FP16, publish capacity, resume.

## Held-out workloads and evidence volume

Content source rows 0–63 were mapped onto the previously validated scale-5.75/4.25 arrival templates. Relative to v9 rows 42–105, 42 questions are new and 22 overlap; a disjoint 64-row slice is impossible from 106 frozen questions.

- low-only: 64 requests;
- one cycle: low→high→low, 192 requests;
- two cycle: low→high→low→high→low, 320 requests.

There are no artificial gaps. Each low phase retains 64 continuing arrivals. Across 16 fresh processes:

- 3,328/3,328 requests completed;
- 1,703,936/1,703,936 output steps completed;
- 94,802 forward observations and 20,329 pressure samples were recorded;
- 7,193 controller samples and 5,087 NVML samples were recorded;
- maximum absolute arrival jitter was 0.0814 seconds, below 0.25 seconds;
- six entries and six restores completed successfully.

The raw matrix is under `benchmark-results/release-side-runtime-v10/runs/`; the immutable 16-cell order and attempt lineage are `run-plan.json` and `execution_status.json`.

## Low-only and high-pressure preservation

Low-only Dynamic issued zero transitions in both repeats. P95 TTFT was 2.199/2.129 seconds versus FP16 2.171/2.158, within the frozen 0.111-second epsilon in both.

Every alternating high phase entered AWQ exactly once and never released inside high. Entry requests occurred about 9.1–9.4 seconds after each high phase began.

| workload / high phase | repeat | FP16 P95 s | Dynamic P95 s | benefit s | epsilon s | waiting area Dynamic / FP16 | preemptions Dynamic / FP16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| one-cycle high 1 | 0 | 14.996 | 5.416 | **9.580** | 1.226 | 42.25 / 245.24 | 3 / 24 |
| one-cycle high 1 | 1 | 13.770 | 5.753 | **8.017** | 1.226 | 44.75 / 185.25 | 3 / 20 |
| two-cycle high 1 | 0 | 15.776 | 5.595 | **10.181** | 0.790 | 42.50 / 227.25 | 3 / 23 |
| two-cycle high 1 | 1 | 15.821 | 5.648 | **10.173** | 0.790 | 43.00 / 235.74 | 3 / 26 |
| two-cycle high 3 | 0 | 16.887 | 6.762 | **10.126** | 0.840 | 55.75 / 269.99 | 3 / 20 |
| two-cycle high 3 | 1 | 16.697 | 6.333 | **10.364** | 0.840 | 54.25 / 270.48 | 3 / 20 |

Thus the held-out cycles reproduce the confirmed mechanism: Dynamic charges entry but retains large queue/preemption/P95 relief versus runtime-static FP16.

## Per-transition detection, drain, and hot-restore timeline

All times/durations are seconds. `blocks` is physical allocation minus the 1,759 base at release intent.

| workload | rep | exit | intent s | blocks | drain? / drain s | decode forwards / requests during drain | hot restore s | pause→resume s | catch-up s | later low arrivals executing FP16 | next entry after resume s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| one-cycle | 0 | 1 | 199.001 | +85 | yes / 1.389 | 29 / 22 | 1.665 | 3.071 | 3.000 | 51 / 51 | none |
| one-cycle | 1 | 1 | 199.500 | +88 | yes / 1.053 | 20 / 22 | 1.653 | 2.722 | 8.219 | 51 / 51 | none |
| two-cycle | 0 | 1 | 199.001 | +85 | yes / 1.187 | 24 / 22 | 1.678 | 2.881 | 3.000 | 51 / 51 | 83.119 |
| two-cycle | 0 | 2 | 376.501 | -558 | no / 0.000 | 0 / 0 | 1.685 | 1.710 | 3.000 | 51 / 51 | none |
| two-cycle | 1 | 1 | 199.750 | +88 | yes / 0.947 | 19 / 22 | 1.649 | 2.611 | 7.873 | 51 / 51 | 82.392 |
| two-cycle | 1 | 2 | 375.001 | +284 | yes / 0.844 | 17 / 22 | 1.717 | 2.584 | 8.614 | 51 / 51 | none |

Five restores genuinely exercised drain above base. Active decoders continued in every such drain. Physical legality arrived in 0.844–1.389 seconds; hot restoration took 1.649–1.717 seconds; total admission pause stayed 1.710–3.071 seconds. The hot restore is material but does not dominate recovery enough to violate the five-second pause gate. Catch-up stayed 3.000–8.614 seconds, under ten seconds.

No request arrived during the short admission pauses because the natural fixed-cohort timing placed each pause between cohorts; the recorded count and queue area during pause are zero. This was not created by an idle gap. Fifty-one later requests arrived in every low phase. All 306 prefetched in FP16 and executed FP16 outputs; 272 were fully FP16, while 34 middle-low requests total (17 in each two-cycle repeat) later crossed the legitimate next high entry after first executing 132–149 FP16 output steps.

## Multi-cycle state timeline and chatter

Both multi-cycle repeats followed the exact order:

- rep0: entry 109.760 → restore 199.019 → re-entry 285.021 → restore 376.523;
- rep1: entry 109.758 → restore 199.765 → re-entry 284.770 → restore 375.022.

The second high began at 275.625 seconds. Re-entry therefore occurred only after high began and 82.392/83.119 seconds after prior admission resume—not as restore-induced chatter. Both runs ended stable FP16. The one-cycle repeats also completed exactly one round trip and ended FP16.

## Recovery latency, queue, and static controls

Recovery-low P95 TTFT remained far below matched FP16 in every phase while above always-AWQ, as expected after returning to dense FP16:

| workload / recovery | repeat | FP16 P95 s | Dynamic P95 s | static AWQ P95 s | Dynamic waiting area / FP16 / AWQ |
|---|---:|---:|---:|---:|---:|
| one-cycle low 2 | 0 | 15.527 | 7.292 | 4.797 | 58.00 / 268.01 / 23.99 |
| one-cycle low 2 | 1 | 12.640 | 7.594 | 5.875 | 67.50 / 188.76 / 30.75 |
| two-cycle low 2 | 0 | 15.824 | 7.518 | 6.096 | 57.75 / 333.02 / 33.50 |
| two-cycle low 2 | 1 | 16.698 | 7.706 | 6.045 | 67.00 / 328.73 / 33.75 |
| two-cycle low 4 | 0 | 19.034 | 8.680 | 5.841 | 96.70 / 372.49 / 31.25 |
| two-cycle low 4 | 1 | 18.797 | 8.251 | 6.286 | 73.50 / 361.50 / 36.00 |

Dynamic was never worse than both controls by more than phase epsilon. The drain/catch-up gates therefore do not indicate that simply staying AWQ was obviously preferable for the rest of low service.

## Frozen throughput non-inferiority

All requests completed, but duration-based completed-request throughput did not pass every 0.98 ratio:

| workload | Dynamic/FP16 rep0 | rep1 | two-repeat ratio | gate |
|---|---:|---:|---:|---|
| low-only | 0.97778 | 0.99005 | 0.98391 | **FAIL** (rep0) |
| one-cycle | 0.98114 | 0.97774 | 0.97944 | **FAIL** (rep1 and aggregate) |
| two-cycle | 0.98376 | 0.98800 | 0.98588 | PASS |

The one-cycle aggregate misses by 0.00056 (0.056 percentage point); low-only rep0 misses by 0.00222 despite zero transitions. This supports a run-noise interpretation but cannot override the frozen margin. Unlike v9, no “numerically no lower” requirement is used.

## Correctness, capacity, HBM, and environment

- All 3,328 requests had one prefill batch, one stable Engine request ID, 512 valid tokens, 512 precision labels/timestamps, and contiguous positions 1,023–1,534.
- All 12 transitions reported success; every active-request KV digest matched before/after resize.
- Every restore first became legal at allocation `<=1,759`, returned physical and scheduler capacity to 1,759, and resumed only after capacity publication.
- There was no request loss, duplicate prefill, remap error, OOM, invalid capacity, or oscillation.
- Model-reported post-restore allocated bytes were identical (`20,070,251,520`) across all six restores. Driver-used HBM repeat spreads were 0 MiB for one-cycle exit 1, 80 MiB for two-cycle exit 1, and 302 MiB for two-cycle exit 2. They were not monotonic within process and did not coincide with capacity/KV failure; they remain allocator/environmental variation, not proof of a leak.

NVML succeeded for all 5,087 samples. Static-AWQ whole-run P95 was 2.984/3.264 seconds (one cycle) and 3.938/4.231 (two cycle), much less variable than v9's 21.395/2.253 split. Static-AWQ mean SM clocks were 1,663–1,675 MHz, mean temperatures 80.5–82.1°C, and mean power 256.3–258.3 W; loaded samples mainly reported software power-cap mask 4, not a distinct thermal excursion. This records the environment but does not retrospectively explain v9.

## Descriptive answer retention

Generated answers are present for every request. Descriptive, non-independent macro F1 is retained in `analysis/quality_descriptive.csv`. Dynamic was 15.732/15.289% in one-cycle and 15.319/15.865% in two-cycle, versus static AWQ 14.500/14.076% and 13.968/14.472%. These repeated-phase values are not a semantic significance analysis and do not enter the release decision.

## Reproduction

Raw execution and regeneration commands are in `benchmark-results/release-side-runtime-v10/execution_commands.json`. To reproduce raw runs exactly, check out preregistration commit `8e7675f`, use the recorded environment, and run the immutable plan. On the result commit, regenerate corrected tables and audit with:

```bash
benchmark-results/release-side-runtime-v10/regenerate.sh
```

## CONFIRMED FINDINGS

- The v9 restoration delay was policy-side: its 0.70 threshold plus 90% dwell remained blocking long after queue/preemption clearance and physical feasibility; observed v9 request-to-hot-start latency was milliseconds.
- The single frozen causal pressure-drop policy restored in all four held-out alternating repeats and all six expected exits.
- Five exits latched 85–284 blocks above base and safely exercised the existing v8 drain without controller-side allocator duplication.
- Every exit left 51 subsequent low-phase arrivals; all 306 prefetched and executed FP16, and 272 remained FP16 for all 512 output steps.
- Both multi-cycle repeats completed `FP16 -> AWQ -> FP16 -> AWQ -> FP16` without labels, intervention, chatter, or corruption.
- Every held-out high phase retained 8.017–10.364 seconds of P95 benefit versus matched FP16, with lower waiting area and no more preemptions.
- Drain, hot restore, and catch-up all met their frozen bounds; staying AWQ was not obviously preferable under the preregistered comparison.

## SUPPORTED BUT UNCERTAIN FINDINGS

- The release-side mechanism appears repaired: every direct release/usefulness criterion passed. Generalization is still limited to one trace family, request shape, checkpoint, GPU class, and two repeats.
- The 0.98 throughput misses are tiny and include a no-transition low-only miss, so run noise is a plausible explanation. Two repeats cannot establish that explanation, and the gate remains failed.
- Driver HBM spread reached 302 MiB at the second two-cycle exit although model allocated bytes and physical capacity were identical. This does not show instability, but more equal-state cycles would be needed to characterize caching variance.
- V10 static-AWQ variance was modest under measured clocks/temperature/power, but those observations cannot identify the cause of v9's outlier.

## BLOCKED QUESTIONS

- Whether the 0.98 throughput misses disappear with more independently preregistered runs is not answered; selective repeats are prohibited.
- Whether `quiet_pressure_drop_5v10` generalizes beyond scale-5.75/4.25 timing and source rows 0–63 is unmeasured.
- Whether reduced AWQ exposure yields statistically significant DuReader F1 remains a separate power problem and was deliberately not tested here.
- The exact source of driver-level HBM spread and v9 static-AWQ variance is not causally identified.

## REMAINING UNCERTAINTY

- Two fresh processes per cell provide limited run-level variance estimation.
- Twenty-two of 64 held-out contents overlap v9 because the frozen corpus has only 106 rows, although all timestamps and content assignment were committed before results.
- The cohort schedule happened to place zero arrivals inside admission pauses. Drain correctness is directly exercised with 22 active decoders, but queue accumulation during an arrival-overlapping drain remains unobserved.
- The post-result percentile correction is bounded and disclosed, but it reduces the evidentiary ideal relative to a bug-free preregistered analyzer.

## RELEASE-SIDE SYSTEMS DECISION: NO-GO

The principled earlier release intent **does repair v9's direct release failure**: useful FP16 restoration is repeated on held-out one- and two-cycle workloads, real drain occurs safely, later low arrivals execute FP16, high-pressure benefit persists, and there is no chatter or state corruption.

However, the frozen overall GO contract also required every repeat and two-repeat workload aggregate to meet the 0.98 throughput non-inferiority margin. Low-only rep0 and one-cycle rep1 fail, and the one-cycle aggregate is 0.97944. Because that preregistered failure cannot be waived after observing results, the formal release-side systems decision is **NO-GO**, with the release mechanism itself confirmed and the remaining blocker narrowed to small run-level throughput non-inferiority uncertainty rather than detection, drain, hot restore, reversibility, or correctness.
