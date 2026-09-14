# Throughput confirmation v11 — Phase A/B result

Status: **Phase A complete; Phase B freezes the controller unchanged for Phase C**.

V10 remains unchanged and remains **RELEASE-SIDE SYSTEMS DECISION: NO-GO**. These 24 new processes are a separate v11 experiment and are not added to v10.

## Preregistration and parity

- Phase A protocol commit: `447c049`; command-only working-directory correction before any model/run initialization: `9dc8d54`.
- Every selected raw run records clean full commit `9dc8d5417a62f33cdbab990de156ba380864ed53`, the same manifest hash `594f83dd...`, GPU 5, and the byte-identical v10 low-only workload hash `f8cedda3...`.
- The first invocation failed path validation before creating a run directory; it is retained in the execution ledger/log and diagnosed in `doc/debug-report.md`. It is not a serving observation.
- `archived_policy_parity.json` passes 27,217 sample decisions across all 36 v7 streams, six v9 Dynamic streams, and six v10 Dynamic streams. Instrumented and current v10 decisions match exactly; archived v10 booleans/actions also match at every sample index/fixed-grid timestamp.
- A bounded post-result audit-only correction is disclosed in `analysis_correction.json`: the first audit conflated a real scientific guardrail failure with administrative raw invalidity. No raw run, selection, controller, analyzer, inference, threshold, or scientific result changed. The corrected artifact audit passes 12/12 checks.

## Paired throughput result

Twelve contiguous fresh-process blocks were run in the frozen alternating order (six FP16→Dynamic, six Dynamic→FP16). No pair was omitted, replaced, or added.

| pair | order | Dynamic / FP16 throughput | duration ratio D/F | Dynamic transitions |
|---|---|---:|---:|---:|
| low-00 | F→D | 0.99408 | 1.00596 | 0 |
| low-01 | D→F | 0.99633 | 1.00368 | 0 |
| low-02 | F→D | 0.99944 | 1.00056 | 0 |
| low-03 | D→F | 0.99915 | 1.00085 | 0 |
| low-04 | F→D | 0.94844 | 1.05437 | 2 |
| low-05 | D→F | 0.96776 | 1.03332 | 0 |
| low-06 | F→D | 0.98065 | 1.01973 | 0 |
| low-07 | D→F | 0.99000 | 1.01010 | 0 |
| low-08 | F→D | 1.00495 | 0.99507 | 0 |
| low-09 | D→F | 1.01962 | 0.98076 | 0 |
| low-10 | F→D | 1.00278 | 0.99723 | 0 |
| low-11 | D→F | 0.98790 | 1.01225 | 0 |

The preregistered paired-log result is:

- geometric-mean ratio: **0.990763**;
- run-level log-ratio SD: **0.018911**; SE: **0.005459**; df: 11;
- t statistic versus `log(0.98)`: **2.0008**;
- one-sided p: **0.03535** at preregistered alpha 0.025;
- 97.5% one-sided lower bound: **0.978930**.

Therefore low-only non-inferiority is **not established** because the lower bound is 0.001070 below 0.98. Conversely, the two-sided 95% upper ratio is **1.002739**, so the preregistered criterion for a repeatable nonzero deficit is also false. The evidence is genuinely inconclusive around the strict margin: the center is a 0.92% deficit, but the observed run-level tail variance remains wide enough to include no deficit.

Generated-token throughput ratios equal completed-throughput ratios because every run completed the same 64×512 outputs. The mean duration ratio was 1.00949 and median 1.00482.

## Controller-overhead attribution

Across the 12 Dynamic runs:

- complete synchronous evaluator plus measured trace work consumed **1.554–2.409 s** per run (mean **1.964 s**);
- this was **1.32–1.87%** of measurement duration;
- window construction/integration consumed **1.469–2.159 s**, the dominant measured component;
- the median measured-controller fraction of positive matched duration excess was **1.736** (173.6%), although this ratio is not causal and becomes unstable when matched excess is near zero;
- post-result counterbalanced replay calibration on 487 archived samples measured the timing implementation at **1.0335×** the current evaluator, adding about **0.020 s** per full replay—far smaller than the 1.55–2.41 s serving-time observation.

Thus controller work is not negligible: rebuilding/sorting and rescanning full history at every 0.25-second tick is measurable and large enough in magnitude to explain a material share of a roughly 1% average deficit. It does **not**, however, establish that all measured wall time is lost GPU service: model execution uses an executor, and the paired result does not show a repeatable nonzero deficit under the frozen criterion.

The preregistered optimization branch required both a repeatable deficit (two-sided 95% upper ratio below 1.0) and at least 25% measured accounting. Only the second condition holds. Therefore **no controller optimization is permitted**. The current parity-proven instrumented implementation and all v10 semantics are frozen unchanged for Phase C.

## Unexpected low-only transition and scheduling variability

Eleven Dynamic runs made zero transitions. `low-04` entered AWQ at fixed-grid sample 370 (92.5 s), after the last planned arrival at 90.5625 s but while the finite-run tail still had waiting depth 10 and scheduler-used blocks 1,701/1,768. Its exact three-second fractions were 1.0 for KV≥0.95 and 0.5834 for waiting≥4, so the unchanged entry rule correctly fired on the observed state. It restored at sample 440 (110.0 s). Both transitions succeeded and KV/capacity/request integrity remained valid.

This is a scientific guardrail failure, not an administrative exclusion. It means the intended all-zero-transition control was not realized in one of 12 fresh processes. That run also had the worst ratio (0.94844), Dynamic P95 TTFT 6.632 s, and much larger loop/telemetry lateness. Across all Dynamic runs, P99 event-loop lag ranged from 1.31 ms to 545 ms and P99 telemetry lateness from 1.38 ms to 634 ms; the corresponding FP16 maxima were 13.4 ms and 50.9 ms. This supports run/tail scheduling variability, but it also reveals low-only instability that two v10 processes did not expose.

## Finite-horizon tail attribution

Every pair's exact log decomposition closes numerically. The completion-tail term accounts for approximately all signed log-throughput difference (median signed fraction **1.0011**, range 0.9880–1.0303); arrival timing and post-completion runner terms are tiny and partially offset. The final planned arrival is 90.5625 s in every run. Post-last-planned-arrival completion drain ranges from 26.43–32.02 s for FP16 and 26.97–38.40 s for Dynamic. The two largest deficits (`low-04`, `low-05`) correspond to Dynamic drains of 38.40 and 36.10 s.

This confirms that the fixed-workload throughput variation is overwhelmingly a completion-tail/makespan phenomenon rather than arrival-count or launch-timing difference. It is diagnostic only; the primary metric remains unchanged.

## HBM and environment

All 24 processes completed all 1,536 requests and 786,432 output steps with valid 1,759 final capacity and FP16 state. NVML was available throughout. Mean SM clocks, temperatures, and power substantially overlapped between conditions; no monotonic thermal drift explains the ratio sequence.

Static FP16 ended at 21,233.2 MiB driver-used HBM in all 12 processes. Dynamic end values were 21,233.2 MiB in ten, 20,283.2 MiB after the `low-04` restore, and 21,793.2 MiB in `low-05`; they then returned to 21,233.2 MiB. The variation is non-monotonic, capacities remained identical, and no KV/allocator failure occurred, so it is not labeled a leak.

## Phase B decision

**KEEP CONTROLLER UNCHANGED.** Low-only NI narrowly misses the preregistered 97.5% lower-bound rule, but a repeatable nonzero deficit is not established. Controller evaluation is material in magnitude, yet the predeclared conjunction authorizing one optimization is not met. Phase C will use this unchanged parity-proven implementation and the same frozen policy/runtime.
