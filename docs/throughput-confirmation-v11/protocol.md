# Throughput confirmation v11 protocol

Status: **Phase A preregistration; temporal proof is the clean commit recorded by every v11 serving run**.

## Scope and preserved decisions

V10 remains the frozen **RELEASE-SIDE SYSTEMS DECISION: NO-GO** under its own preregistered every-repeat and two-repeat 0.98 rule. V11 neither adds observations to v10 nor changes any v10 artifact, result, threshold, workload, or decision. V11 asks only whether the approximately 0–2.2% v10 completed-throughput deficit is a repeatable cost of the closed-loop control/adaptation system or finite-run noise that two processes per cell could not resolve.

The following are unchanged from v10: the sustained-compound FP16→AWQ entry rule; `quiet_pressure_drop_5v10`; every threshold/window and the 0.25-second policy grid; AWQ-Marlin and decoder layers `[0,16)`; 1,759/4,170 physical capacities; the separate 1,768 policy reference; strict-FCFS ordering; segmented KV resize/remap/shrink; admission hold/drain; hot transition implementation; model and quantized checkpoint; 1,024-token prompts; exact 512-token greedy outputs; batch limits; seed; runtime preparation; workload bytes; and physical RTX 3090 GPU 5. No semantic-quality experiment is included.

All v5–v10 result/doc namespaces are read-only and hashed in the manifest. The bounded v10 percentile correction and original raw-run provenance remain exactly as reported.

## Phased design

### Phase A: low-only control-plane attribution

Use the exact v10 `low-only.jsonl` bytes. Twelve independent paired blocks are fixed before results. Each block contains one fresh-process `runtime_static_fp16` run and one fresh-process `closed_loop_dynamic` run, executed contiguously on GPU 5. Condition order alternates exactly `FP16, Dynamic` then `Dynamic, FP16`, giving six blocks in each order. All 24 runs use the same runtime-capable process and prepare both pinned states. Dynamic is expected to issue zero transitions; any transition is a systems guardrail failure.

The immutable order is `benchmark-results/throughput-confirmation-v11/phase-a-run-plan.json`. V10 observations are excluded from every primary v11 statistic and retained only as historical/sample-size context. There is no early stop, selective extra block, outlier deletion, or post-result run-count change.

### Phase B: fixed controller-overhead decision

Instrumentation records only outputs and never supplies a policy input. Before Phase A serving, the instrumented evaluator must match the current v10 evaluator exactly on every decision field for all 36 archived v7 telemetry streams, all six archived v9 Dynamic streams, and all six archived v10 Dynamic streams. The v10 streams must also reproduce every archived entry/release boolean and requested action. Action sample index and the recorded fixed-grid scheduled timestamp must match. Failure rejects the instrumented runner before serving.

After Phase A:

1. Non-inferiority is supported if the preregistered workload lower confidence bound is strictly above 0.98.
2. A repeatable nonzero deficit is declared only if the upper endpoint of a two-sided 95% paired-log confidence interval is below 1.0.
3. Measured synchronous controller work materially accounts for a deficit only if the median across positive-duration-excess pairs of `(sum evaluate wall + measured decision/trace bookkeeping wall) / (Dynamic duration - FP16 duration)` is at least 25%.
4. Exactly one semantics-preserving evaluator optimization is eligible only if both items 2 and 3 hold. Otherwise the controller remains unchanged.
5. An eligible optimization is accepted only after exact parity on every archived v7/v9/v10 stream for entry/release booleans, action sequence, sample index, and fixed-grid timestamp. Otherwise it is rejected.

The measured trace-bookkeeping boundary includes policy-history insertion, decision record construction, and `trace_fields`; the unchanged outer v10 runner's final dictionary copy/list append cannot be separated without duplicating or editing the frozen runner and is reported as an explicit lower bound. Process CPU time is supplemental because it is process-wide; synchronous wall time is the attribution quantity.

After the Phase B choice, the unchanged or parity-proven implementation will be frozen in a new commit before Phase C. Phase C receives its own immutable plan and manifest before any Phase C serving result.

### Phase C: independent adaptation confirmation

The fixed design is eight matched fresh-process pairs on the unchanged held-out one-cycle workload and eight on the unchanged held-out two-cycle workload: 32 runs total. Pair order will be balanced four/four per workload and frozen before results. All eight pairs per workload must be run; there is no early stop. V10 runs remain historical only. No always-AWQ condition is needed because the sole endpoint is Dynamic versus matched FP16 and v10 already established AWQ interpretation.

Each Dynamic run must preserve: no false entry in low pressure; one entry per true high phase; useful low-phase release; a real admission drain whenever intent allocation exceeds 1,759; later post-resume low requests prefill and execute FP16; exact one-/two-cycle transition order; no chatter; no loss, duplicate prefill, ID/position error, KV digest mismatch, illegal capacity, remap error, OOM, or failed transition. These are hard guardrails, not alternative throughput endpoints.

## Observation-only timing

The new controller class reproduces v10 arithmetic but adds monotonic wall and process-CPU timing around:

- complete `evaluate()` work;
- inherited 3/10-second and v10 5/10/15-second segment/window construction and integration;
- policy-history/decision bookkeeping; and
- transition trace-field construction.

The unchanged v10 runner already records 0.25-second telemetry lateness as `sampling_jitter_s`, request launch lateness as `arrival_jitter_s`, GPU/HBM samples, and producer/consumer first/final timestamps. V11 additionally runs the same 50-ms event-loop-lag probe in both conditions. Output-consumer scheduling lag is derived where measurable as first stream receipt minus Engine first-output timestamp and stream completion receipt minus Engine completion timestamp; no claim is made about intermediate-token producer lag. All raw timing is buffered and written after measurement. None affects a decision.

## Primary endpoint and inference

The independent unit is one fresh-process paired block, never a request or phase. For condition `c`, workload `w`, block `i`:

`T_cwi = completed_request_count / measurement_duration_s`,

using the unchanged v10 runner/summarizer measurement clock. A normally valid run must complete the exact planned count (64, 192, or 320) and all exact 512 output steps. For each pair:

`Y_wi = log(T_Dynamic,w,i / T_FP16,w,i)`.

The workload estimand is `R_w = exp(E[Y_wi])`, the geometric-mean run-level Dynamic/FP16 completed-throughput ratio under this paired fresh-process procedure. Test:

- `H0_w: E[Y_wi] <= log(0.98)`
- `H1_w: E[Y_wi] > log(0.98)`.

For every workload use a one-sample paired Student t analysis on `Y_wi`, one-sided alpha 0.025 (97.5% one-sided lower confidence bound). If `n` is the fixed valid-pair count, the bound is:

`L_w = exp(mean(Y) - t_(0.975,n-1) * sd(Y)/sqrt(n))`.

Non-inferiority passes only if `L_w > 0.98`; equality fails. Report every ratio/log ratio, `n`, geometric mean, SD, SE, degrees of freedom, t statistic, one-sided p-value, and lower bound. No normality/outlier diagnostic selects a different primary method; no trimming, winsorization, bootstrap, permutation, sign, Wilcoxon, or arithmetic ratio-of-means substitutes post hoc.

The final confirmatory claim is conjunctive: low-only, one-cycle, and two-cycle must each pass. This is an intersection-union test, so each component uses alpha 0.025 without multiplicity adjustment; a pooled result cannot rescue any workload. No pooled primary result is planned.

Also report, without gates: measurement-duration ratio, generated-token-throughput ratio, P95/P99 TTFT, mean/P95 TPOT, SM/memory clocks, temperature, power, controller costs, telemetry/launch jitter, event-loop lag, and first/final output-consumer lag.

## Validity, attempts, and missingness

Administrative invalidity is limited to independently evidenced failure to execute/record the intended experiment: wrong clean commit, manifest, workload hash, device, config, process freshness, absent/corrupt timestamps/files, external interruption, or arrival-generator malfunction. A Dynamic timeout, transition failure, scheduler stall, OOM, request loss, or zero completion caused by the system is a scientific failure and cannot be discarded.

Every attempt is preserved. A selected block must contain both members from the same block attempt and frozen order; conditions cannot be spliced across attempts. At most two full-block administrative replacements are allowed. The chronologically first complete administratively valid block attempt is selected. If the fixed valid-pair count is not reached, that workload is inconclusive and does not pass. No block is replaced because its ratio, latency, tail, transition, or environment is unfavorable.

## Finite-horizon tail decomposition

For every run report the final planned arrival offset/timestamp, final actual arrival, last request stream-completion timestamp, post-last-planned-arrival completion drain, and the runner interval after last completion. Let `H` be the common planned horizon, `J` final actual-arrival elapsed minus `H`, `Q` last completion minus final actual arrival, and `R` measurement end minus last completion. Then duration is exactly `H+J+Q+R`. The paired log throughput ratio is decomposed sequentially into arrival-timing, completion-tail, and post-completion-runner log terms whose sum must equal the primary log ratio. The completion-tail term divided by total log ratio is the signed fraction attributable to tail duration and may exceed `[0,1]` when components offset. This is diagnostic only and never changes the primary endpoint.

## HBM and environment

All telemetry retains driver-used/free HBM, physical/scheduler KV capacities, allocator transition memory, NVML clocks, temperature, power, utilization, P-state, fan, and throttle mask. Equal-state HBM values are ordered by transition and fresh-process ordinal. Monotonic persistent growth together with allocator/state evidence is a defect; non-monotonic driver-cache variation alone is not labeled a leak. Identical 1,759/4,170 capacities and correctness remain hard requirements.

## Decisions

**THROUGHPUT CONFIRMATION GO** requires all three workload-specific 97.5% one-sided lower bounds to exceed 0.98, every systems/reversibility/correctness guardrail to pass, and no new persistent controller-overhead, HBM, or stability defect.

**THROUGHPUT CONFIRMATION NO-GO** results if any workload remains credibly below/inconclusive against the 0.98 margin, a material controller cost cannot be removed without semantic change, or repeated runs reveal instability. A failed workload cannot be replaced by pooling, and the 0.98 margin is never tuned.
