# V9 AWQ→FP16 release-delay diagnosis

Status: **development evidence only**. This replay selects a v10 release-intent rule; it is not final validation because an earlier admission hold and restore would change all later execution.

## Evidence and method

`benchmark.analyze_release_latency` replays the four frozen v9 Dynamic high/phased `controller.jsonl` traces with the exact left-continuous time integration helpers from v7. It emits:

- `benchmark-results/release-side-runtime-v10/development/v9_release_trajectory.csv` — every post-entry queue, preemption, scheduler-used/physical-used KV, `/1768` utilization, `1759-used` margin, old-rule component, and candidate component;
- `development/v9_release_delay_decomposition.{csv,json}` — first-passage times and barrier lags;
- `development/v7_operating_regimes.csv` — the preserved scale-5.75 low/default and scale-4.25 AWQ-preferred measurements used to bound the held-out regimes.

The five requested causes are interacting, not independently additive. In particular, a 90%-of-ten-second dwell cannot be separated from the threshold whose truth it integrates. The tables therefore report exact first-passage/barrier times and an explicitly ordered threshold→dwell description rather than pretending to have a randomized causal mediation decomposition.

## Reconstructed first-passage timeline

All values are measurement elapsed seconds. `AWQ stable` is the measured hot-entry end. Exact Engine transition context shows allocation was at or below the 1,759-block base at that boundary in all four runs; later admissions raised it again.

| v9 Dynamic run | AWQ stable | old 10-s residence ready | old wait/preempt both ready | first physical shrink feasible | first `used/1768<=.70` | start of release-producing/terminal <=.70 episode | old 90% dwell / complete rule | PRIMARY intent replay | blocks vs 1,759 at intent |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| high rep0 | 69.530 | 79.001 | 81.251 | 69.530 | 89.751 | 89.751 | 98.751 | **86.001** | -326 |
| high rep1 | 48.226 | 57.500 | 58.501 | 48.226 | 89.751 | 89.751 | 98.751 | **85.000** | -306 |
| phased rep0 | 134.018 | 143.500 | 143.001 | 134.018 | 200.501 | 286.251 | censored at 295.000; dwell 87.50% | **199.001** | **+85** |
| phased rep1 | 110.932 | 120.250 | 120.001 | 110.932 | 201.001 | 286.500 | 295.751 | **199.751** | **+88** |

The phased recovery boundary was 175.0 s. The PRIMARY remained false throughout the recorded high interval and first became true 24.001/24.751 s into recovery. On the unchanged trace, physical usage next fell to the 1,759-block base 1.249/1.251 s later. That last number is only development context: a real admission hold changes the drain trajectory and must be measured prospectively.

The high-only last arrival was 66.938 s. PRIMARY first truth at 86.001/85.000 s therefore occurs only after the finite high input has stopped and the measured pressure has dropped, not during the demonstrated sustained-arrival interval.

## Delay attribution

### 1. Ten-second persistence window

The standalone residence barrier consumed 10.24 s after the entry request in every trace. It was **not incrementally binding at the old release time**: by the time every other old condition was true, the residence barrier had expired. Its ordered incremental contribution to the observed old-rule release was therefore 0.000 s in each run that released.

### 2. The 0.70 FP16-reference threshold

This was the dominant pre-dwell barrier. Relative to the first exact physical-feasibility instant, the first low-threshold crossing lagged by:

- 20.221 s (high rep0);
- 41.524 s (high rep1);
- 66.484 s (phased rep0);
- 90.069 s (phased rep1).

Transient crossings did not suffice. The release-producing/terminal below-0.70 episode did not begin until 286.251/286.500 s in the phased runs, **152.233/175.568 s after physical shrink had first been legal**. At 0.70, the policy waited for `used <=1237.6` blocks—over 521 blocks below the actual 1,759-block safety limit.

### 3. The 90% dwell fraction

Once the release-producing below-0.70 episode began, the 90%-of-ten-second rule added 9.001/9.000/at-least-8.873/9.251 s in high rep0, high rep1, phased rep0, and phased rep1. Phased rep0 ended at 87.50% dwell and was censored rather than extended.

### 4. Waiting and preemption conditions

The old ten-second waiting/preemption pair became jointly ready 11.721/10.274/8.983/9.069 s after AWQ became stable. It was already true long before every eventual release. Post-AWQ time-weighted waiting areas were 8.709, 26.642, 83.678, and 89.277 request-seconds respectively, retained in the raw decomposition.

Queue-free or physically-feasible current state alone is unsafe: both phased traces became queue-free and briefly feasible during the still-active high phase. This rules out a naive queue-clear/base-feasible release.

### 5. Actual physical drain/shrink feasibility

Exact entry-completion contexts were already at/below 1,759 in all four traces, so physical feasibility contributed **zero to the observed v9 pre-request delay**. After later admissions, the PRIMARY would request at +85/+88 blocks in the phased traces; unchanged replay reached safety about 1.25 s later. Only held-out actuation can establish the real drain duration.

For the three v9 restores that actually occurred, request-to-hot-start pending intervals were 0.0419, 0.0073, and 0.0050 s. They contain no material drain episode. V9 therefore measured policy detection delay, not a costly v8 drain.

## Frozen candidate selection

Exactly one new release-intent candidate is retained: `quiet_pressure_drop_5v10`.

While stable in AWQ, after 15 seconds of complete AWQ history, intent is true iff:

1. current waiting depth is zero;
2. the exact time-weighted waiting mean over `[t-5,t]` is `<=0.5`;
3. no new preemption occurred over `[t-5,t]`;
4. current scheduler-used KV divided by 1,768 is `<=1.25`; and
5. mean FP16-reference utilization over `[t-5,t]` is `<=0.80` times its mean over `[t-15,t-5]`.

This is a causal pressure-drop rule, not a physical safety test. It does not inspect physical allocation, workload label, phase, scale, nominal/known RPS, future arrivals, completion result, TTFT, or generated content. The existing Engine remains the sole authority for the 1,759-block shrink.

The candidate fires 12.751/13.750 s earlier than the old rule in high rep0/1 and 96.000 s earlier in phased rep1. In phased rep0 it fires 96.12 s before trace end while the old rule remains false. Most importantly, the development replay requests with allocation above base in both phased repeats, directly motivating prospective validation of the v8 drain path.

No sensitivity policy is actuated or preregistered. This avoids selecting among several post-result release variants on only four development trajectories.
