# Release-side runtime v10 protocol

Status: **pre-registered before every v10 serving result; the clean run commit is recorded in raw metadata**.

## Narrow scientific question

V9 remains a frozen systems and quality–latency **NO-GO**. This follow-up asks only whether v9 failed because AWQ release intent was unnecessarily late, and whether a causal pressure-drop intent can safely exercise the existing v8 admission-hold/drain/physical-shrink path early enough to provide useful FP16 recovery on held-out alternating workloads.

The confirmed FP16→AWQ mechanism is not retuned. The state pair remains `{FP16, AWQ-Marlin W4-16}`; AWQ still covers decoder layers `[0,16)`. The checkpoint, quantized artifact, strict-FCFS scheduler ordering, v8 segmented KV allocator/remap/shrink implementation, 1,759-block physical FP16 base, 4,170-block safe runtime AWQ capacity, 1,768-block policy reference, model, 1,024-token prompts, exact 512-token outputs, backend, batch limits, and transition implementation remain fixed. Existing v2–v9 evidence is read-only development/history.

No workload label, phase, scale, nominal RPS, future arrival, completion outcome, TTFT, answer, token, or offline classification is available to the online controller.

## Development diagnosis and candidate bound

`docs/release-side-runtime-v10/release-delay-diagnosis.md` and `benchmark-results/release-side-runtime-v10/development/` replay every v9 Dynamic high/phased trace. Replay is development evidence only. It shows:

- v9 physical shrink was already feasible at AWQ-entry completion, but later admissions raised allocation again;
- the old waiting/preemption pair cleared roughly 9–12 seconds after AWQ stabilized;
- the 0.70 threshold delayed the release-producing episode by tens to more than 150 seconds after first feasibility;
- 90%-of-ten-second dwell then added approximately nine seconds;
- actual successful v9 restore request-to-hot-start intervals were 0.005–0.042 seconds;
- queue-free/base-feasible snapshots occurred inside the high phase and are therefore insufficient; and
- one causal pressure-drop candidate first fires at 199.001/199.751 seconds in the two phased traces, 24.001/24.751 seconds after recovery starts, with allocation 85/88 blocks above base.

Exactly **one** new release-intent candidate is selected. There is no actuating or reported sensitivity candidate to substitute after results.

## Frozen controller

### Entry — unchanged v7/v9 policy

At each causal 0.25-second tick while stable FP16, request FP16→AWQ when the full trailing three seconds have:

- scheduler-used blocks divided by 1,768 `>=0.95` for at least 80% of elapsed time;
- waiting depth `>=4` for at least 50% of elapsed time; and
- current waiting depth `>=4`.

The implementation subclasses the v9 controller and calls its exact `evaluate` path for entry.

### PRIMARY release intent — `quiet_pressure_drop_5v10`

After an entry completes, reset the release-only history; its origin is the first controller sample that observes stable AWQ. While stably AWQ, require at least 15.0 seconds of this complete AWQ-only history. At tick `t`, latch AWQ→FP16 release intent iff all are true:

1. current waiting depth is exactly `0`;
2. exact left-continuous, time-weighted mean waiting depth on `[t-5,t]` is `<=0.5`;
3. preemption-count increase on `[t-5,t]` is exactly `0`;
4. current scheduler-used blocks divided by 1,768 is `<=1.25`; and
5. exact mean scheduler-used/1,768 utilization on `[t-5,t]` is `<=0.80` times its mean on `[t-15,t-5]`.

The v7 `telemetry_segments`, `integrate`, and `counter_value_at` semantics are authoritative. The rule is a queue-clear plus pressure-trend hysteresis. The 1.25 ceiling is intentionally above the 1,759-block physical base (`1.25*1768=2210`), so it is not allocator safety duplicated in the controller.

### Physical restore safety — unchanged v8 authority

Release intent may latch above the FP16 base. Once the existing Engine receives the restore request it:

1. pauses new prefills and swap-ins without changing FCFS order;
2. continues decoding active requests;
3. waits until the physical allocator reports allocation `<=1,759`;
4. calls the unchanged v8 KV compact/remap/shrink and dense-FP16 restoration;
5. publishes the 1,759 scheduler capacity only after physical resize succeeds; and
6. resumes admissions.

The controller never authorizes shrink and does not reproduce allocator logic. Scheduler, allocator, model transition, kernel, and remap behavior are unchanged; Engine changes are observation-only lifecycle timestamps plus an explicit immediate restore hold at the already-existing forward boundary.

After completed restoration, the unchanged entry rule is active again. No workload-aware cooldown is added.

## Frozen held-out workloads

The input bytes and hashes are in `benchmark-results/release-side-runtime-v10/input/workload_metadata.json`; run order is in `run-plan.json`.

Arrival offsets are exactly the validated v7 scale-5.75 low/default and scale-4.25 AWQ-preferred templates. Request content is deterministically remapped to frozen source rows 0–63 instead of v9 rows 42–105. This yields 42 new and 22 overlapping questions; a fully disjoint 64-row slice is impossible from the 106-row frozen source. Content inference is not a release-side gate.

- `low-only.jsonl`: 64 scale-5.75 requests.
- `heldout-one-cycle.jsonl`: low→high→low, 192 requests.
- `heldout-two-cycle.jsonl`: low→high→low→high→low, 320 requests.

Each phase begins one natural existing cohort interval after the preceding phase's last arrival. There is no artificial idle gap. Every low phase has all 64 continuing arrivals; each final low's last arrival is 90.5625 seconds after phase start.

## Frozen run matrix

Every run is a fresh process on physical RTX 3090 GPU 5. The 16 counterbalanced cells are:

- low-only: runtime-static FP16 and Dynamic, two repeats each (4);
- one-cycle: runtime-static FP16, runtime-static AWQ-W4-16, and Dynamic, two repeats each (6);
- two-cycle: the same three conditions, two repeats each (6).

All conditions use the same runtime-capable process and prepare both pinned weight states before KV profiling. Runtime-static AWQ morphs before warmup/measurement. Every Dynamic transition after the measurement clock is charged. Static AWQ is interpretive and cannot determine the primary release decision.

An incomplete attempt is preserved. Resume creates `-attemptN`; `execution_status.json` selects exactly one valid attempt per planned cell. No selective third successful repeat is allowed.

## Raw telemetry contract

Each run records metadata, request, periodic telemetry, batches, controller samples, transitions, initialization transition, runtime preparation, GPU environment, and summary.

Every restore records separately:

- release-intent condition first true and controller restore request;
- Engine restore receipt and admission-pause/drain start;
- allocated and scheduler-used blocks at intent plus blocks above/below 1,759;
- allocated blocks at pause and first physical-shrink legality;
- drain end;
- hot restore start/end;
- capacity publication and admission resume;
- arrivals during the pause;
- decode forwards/request IDs during drain;
- queue area/peak during drain and pause;
- catch-up end and queue area after resume;
- post-restore arrivals and FP16 execution; and
- next entry request, if any.

Request IDs, one-prefill evidence, all 512 token positions/precision labels/receipt times, KV digests, physical/scheduler capacities, transition memory, allocated HBM, request loss, and errors remain recorded.

`gpu_environment.jsonl` samples NVML at approximately one second and includes SM/memory clocks, temperature, power draw/limit, utilization, P-state, fan, and throttle-reason mask. Unsupported telemetry is explicitly marked unavailable; it never changes controller behavior.

## Frozen metrics, margins, and gates

Fresh serving run is the independent systems unit. Request-level samples do not create extra repeats.

### Noise and throughput

For each workload phase:

`epsilon = max(0.111 s, FP16 two-repeat P95 range, Dynamic two-repeat P95 range, 0.05 * FP16 P95 mean)`.

Completed-request throughput is non-inferior iff every matched Dynamic/FP16 repeat ratio is `>=0.98` and the ratio of the two-repeat mean-equivalent sums is `>=0.98`. V9's brittle “Dynamic mean numerically no lower” clause is retired. Strict `TTFT>2 s` remains reported as a paper-reference metric, not a sole decision gate.

### Low-only and high-phase preservation

- Both low-only Dynamic repeats must make zero AWQ entries and remain within epsilon of matched FP16 P95 TTFT.
- Every high phase in both alternating repeats must request exactly one entry, make no release request, improve matched FP16 P95 TTFT by more than phase epsilon, have lower time-weighted waiting area, and have no more preemptions.
- All requests/transitions/KV/capacities must remain valid.

### Useful recovery — every expected restore

A restore passes only if all are true:

1. release intent is in the low phase following a high phase, scored offline only;
2. lifecycle timestamps are complete and ordered;
3. sampled logical KV digest is preserved and restoration returns to 1,759 blocks/FP16;
4. at least **8** requests from that low phase arrive after admission resume, and at least 8 prefill and execute one or more output steps in FP16;
5. at least **20.0 seconds** remain until that low phase's last planned arrival at resume;
6. no request is lost/re-prefilled, no ID/position is corrupted, and no OOM/remap error occurs;
7. pause-to-resume is `<=5.0 seconds`;
8. catch-up is `<=10.0 seconds`, where catch-up ends at the first time after resume that waiting depth has remained continuously zero for 3.0 seconds;
9. Dynamic's recovery-low P95 is not worse than **both** matched static FP16 and static AWQ by more than phase epsilon; and
10. no entry occurs during the first 15.0 seconds after resume or before the next actual high phase. A later entry is valid only after the offline phase boundary and the unchanged causal entry signal.

The 5-second pause bound allows more than twice the v9 1.64–1.95-second hot restore while still rejecting a drain/restoration stall that consumes useful low service. The 10-second catch-up bound directly limits queue cost rather than inferring it from aggregate throughput.

For the two-cycle workload, each repeat must produce exactly:

`FP16 -> AWQ -> FP16 -> AWQ -> FP16`

without manual intervention or online phase labels. One-cycle repeats require exactly one ordered round trip. Extra transitions, high-phase release, immediate re-entry, or missing final FP16 are failures.

### Release-side decision

**RELEASE-SIDE SYSTEMS GO** requires all 16 raw runs valid, both low-only gates, every high-phase preservation gate, every throughput margin, all six expected useful restores, both repeated exact one-/two-cycle state sequences, no chatter, and no corruption. Otherwise the decision is **NO-GO**.

GPU environment availability is a mandatory evidence/completion check where NVML is available, but an explicitly recorded sensor failure does not silently turn a scientific result into GO or NO-GO. HBM and restored-state spread are reported; physical/correctness instability is still a hard failure.

Generated answers and descriptive DuReader F1 are retained, but no semantic significance or CI is a release-side completion gate.

## Failure interpretation

- No timely intent: release detection failure.
- Intent above base followed by long legality delay: drain/admission-cost failure.
- Feasible quickly but hot restore dominates: transition-cost failure.
- Restore succeeds but catch-up misses bound: admission-pause queue failure.
- Release inside high or re-entry before the next high: hysteresis/chatter failure.
- OOM, remap, capacity, request, position, ID, or digest failure: substrate regression.
- Development replay works but held-out cycles fail: retrospective overfit.
- Static-AWQ variance remains despite GPU telemetry: interpretive uncertainty, never permission to alter the primary policy.

No threshold, workload, run count, backend, controller, gate, or implementation may change after the first v10 serving result.
