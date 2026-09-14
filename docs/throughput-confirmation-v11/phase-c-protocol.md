# Throughput confirmation v11 — Phase C frozen supplement

Status: **preregistered after Phase A/B and before every Phase C serving result**.

This supplement incorporates `protocol.md` without changing it. V10 remains frozen **NO-GO**. Phase A's 12-pair geometric-mean low-only ratio was 0.990763 with a 97.5% one-sided lower bound of 0.978930; non-inferiority was not established, but a repeatable nonzero deficit was also not established because the two-sided 95% upper ratio was 1.002739. Measured controller work was material in magnitude, but the preregistered two-condition optimization trigger was not met. Therefore Phase B makes **no optimization**: Phase C freezes the same observation-only instrumented evaluator and exact v10 policy/runtime at result commit `6f14431` plus the audit-coverage-only change included in this preregistration commit.

## Immutable Phase C matrix

`benchmark-results/throughput-confirmation-v11/phase-c-run-plan.json` fixes 16 contiguous paired blocks and 32 fresh processes:

- eight pairs on byte-identical v10 `heldout-one-cycle.jsonl`;
- eight pairs on byte-identical v10 `heldout-two-cycle.jsonl`;
- each pair contains one runtime-static FP16 and one closed-loop Dynamic process;
- one- and two-cycle pairs are interleaved by round;
- condition order is balanced four/four within each workload and opposite between workloads in each round.

All planned pairs run to completion; no early stop, result-dependent extension, selective rerun, or v10 observation enters the primary inference. Attempt/missingness rules remain those in the main protocol.

## Frozen analysis and decision

For each workload separately, use the already-preregistered paired Student t analysis of run-level log Dynamic/FP16 completed-throughput ratios with margin 0.98 and one-sided alpha 0.025. Eight is the fixed independent pair count. Non-inferiority requires the 97.5% one-sided lower bound to be strictly greater than 0.98. One-cycle cannot rescue two-cycle or vice versa; neither can rescue Phase A low-only. No pooled primary inference is used.

The final v11 claim is the intersection of all three independently preregistered workload components:

1. Phase A low-only lower bound >0.98;
2. Phase C one-cycle lower bound >0.98;
3. Phase C two-cycle lower bound >0.98.

All systems/correctness/HBM requirements in the main protocol remain hard. In every Dynamic run, offline audit must confirm no low-phase false entry, exactly one entry per high phase, release only in the following recovery low, real drain whenever intent is above 1,759, post-resume low requests prefill and execute FP16, exact state order, no chatter, one prefill per request, all positions/tokens/IDs intact, KV digest preservation, legal 1,759/4,170 capacity, successful transitions, no OOM, and final FP16/1,759 state.

Even if both Phase C workloads pass, the final v11 decision is NO-GO if low-only remains failed or any hard guardrail fails. This does not revise v10.
