# MorphServe closed-loop runtime adaptation v9

Status: **complete — the entry/pressure-relief mechanism is confirmed, but the pre-registered end-to-end decisions are NO-GO**.

This final reproduction experiment combined the v7 causal crossover signal with the v8 one-process state-preserving runtime substrate for the exact pair `{FP16, AWQ-Marlin W4-16}`. It used contemporaneous same-runtime-envelope controls, two fresh-process repeats for each of three frozen workloads and three conditions, and charged every post-clock transition. No threshold, workload, backend, scheduler, transition implementation, or success gate was changed after results.

The central result is mixed but decisive. Dynamic avoided AWQ at low pressure and cut high-pressure P95 TTFT from 13.261/12.485 s to 5.068/5.222 s after charging 0.760/0.941-s entry transitions. Every request and sampled KV state survived. However, the frozen 10-s release hysteresis restored too late to provide the predeclared useful recovery in either phased repeat, and two secondary high-pressure mechanism gates narrowly/partly failed. Dynamic's DuReader F1 was 0.939 percentage points above static AWQ in the unique-question aggregate, but the paired 95% CI `[-1.582,+3.819]` includes zero. The full closed-loop quality–latency claim is therefore not confirmed.

## Frozen protocol and evidence volume

- Protocol commit: `168a27d` (`research(protocol): preregister closed-loop runtime adaptation v9`), before all v9 serving runs.
- Conditions: `RUNTIME-STATIC-FP16`, `RUNTIME-STATIC-AWQ-W4-16`, and `CLOSED-LOOP-DYNAMIC`.
- Workloads: v7 scale 5.75 low control, v7 scale 4.25 headline, and deterministic low→high→low concatenation.
- Matrix: 18/18 fresh-process runs, no replacement attempts; all returned success.
- Requests: 1,920/1,920 completed; 983,040/983,040 output tokens; every request had 1,024 prompt tokens, exactly 512 valid output steps, one stable engine request ID, contiguous positions 1,023–1,534, and 512 per-step precision/timestamp records.
- Raw observations: 12,486 fixed-cadence pressure samples and 59,353 forward-batch events.
- Arrival integrity: maximum absolute launch jitter was 0.0182 s, below the frozen 0.25-s gate. No arrival happened during a hot transition; this follows the trace's cohort timing, not a paused source.
- Runtime preparation: all conditions prepared both host variants before KV profiling; measured preparation ranged 10.281–13.345 s (median 12.533 s) and remained outside the measurement clock.
- Capacity: every process initialized at the v8 runtime base of 1,759 physical blocks. Static AWQ initialization and dynamic entry expanded to the same safe 4,170-block total. Controller policy pressure remained `logical used / 1768`; physical safety and policy reference were independently traced.
- Environment: one NVIDIA RTX 3090 exposed as physical device 5, torch 2.9.0+cu128, vLLM 0.11.2, Transformers 4.51.3, the existing Llama-3.1-8B snapshot, existing AutoAWQ W4/G128/ZP artifact, strict FCFS, and the v8 segmented KV implementation.

Raw evidence is in `benchmark-results/closed-loop-runtime-v9/runs/`. `analysis/serving_runs.csv` is the complete per-run wide table; every value below regenerates from requests, telemetry, batches, controller samples, and transition traces.

## Headline high-pressure Table-5-style comparison

Values before `/` are repeat 0/1. Parenthesized values are two-repeat means. F1 uses the explicitly defined 64-unique-question aggregate; its repeat values are shown separately below.

| condition | DuReader F1 % | P95 TTFT s | >2-s violation | P99 TTFT s | P95 queue s | preemptions | FP16/AWQ wall occupancy | AWQ output-token fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RUNTIME-STATIC-FP16 | 15.255 | 13.261 / 12.485 (12.873) | 34.4% / 32.8% (33.6%) | 16.393 / 16.428 (16.411) | 11.323 / 10.554 (10.939) | 12 / 12 | 100.0% / 0.0% | 0.0% |
| RUNTIME-STATIC-AWQ-W4-16 | 14.494 | 21.395 / 2.253 (11.824) | 78.1% / 54.7% (66.4%) | 24.366 / 5.489 (14.928) | 19.580 / 0.049 (9.814) | 0 / 0 | 0.0% / 100.0% | 100.0% |
| CLOSED-LOOP-DYNAMIC | 15.433 | **5.068 / 5.222 (5.145)** | 40.6% / 20.3% (30.5%) | **10.521 / 7.875 (9.198)** | **2.795 / 3.687 (3.241)** | **5 / 2** | 59.0% / 38.9% mean | **30.7% / 58.8% (44.7%)** |

The strict 2-s count remains a paper-reference metric, not the sole local criterion: local synchronized prefill service is near two seconds. Dynamic repeat 0, for example, has a higher strict-violation rate than FP16 (40.6% versus 34.4%) while reducing P95 by 8.193 s and P99 by 5.872 s.

Static AWQ repeat 0 is a genuine same-envelope raw result, not discarded: P95 was 21.395 s versus 2.253 s in repeat 1, waiting area 333.24 versus 11.75 request-seconds, and TPOT P95 94.4 versus 60.3 ms despite zero preemptions. Its cause is unresolved because temperature/power clocks were not recorded. This large run-level variance prevents a clean always-AWQ latency ranking; it does not affect the predeclared Dynamic-versus-FP16 high gate, whose noise formula excludes static AWQ.

## Low-pressure negative control

The primary controller made **zero transitions in both** scale-5.75 repeats. All 64 requests in each remained FP16, so Dynamic had 0% AWQ wall/token occupancy. Dynamic P95 TTFT was 2.230/2.144 s versus runtime-static FP16 5.706/2.150 s. The large FP16 repeat-0 tail made the frozen low-load noise threshold 3.556 s; both matched differences were within it. Thus the false-entry and FP16-like latency gates pass without interpreting the favorable Dynamic repeat-0 difference as a causal controller gain.

Static AWQ low P95 was 2.343/2.267 s. This preserves v7's qualitative reason to default to FP16 under low pressure: no quantization was needed, while its local latency was within the conservative run-level envelope.

## High-pressure causal controller result

The controller causally entered AWQ in both headline repeats using only the frozen three-second trailing window:

| repeat | entry condition / request s | hot entry s | restore condition / request s | hot restore s | requests spanning entry/restore |
|---:|---:|---:|---:|---:|---:|
| 0 | 68.751 / 68.756 | 68.769–69.530 (0.760 s) | 98.751 / 98.758 | 98.800–100.444 (1.644 s) | 33 / 1 |
| 1 | 47.250 / 47.262 | 47.286–48.226 (0.941 s) | 98.751 / 98.757 | 98.765–100.490 (1.725 s) | 27 / 1 |

Pending/boundary time before entry was only 0.014/0.024 s; before restore it was 0.042/0.007 s. Entry hot costs are above v8's controlled 0.661-s median because these measurements include enabled KV hashing and loaded-service context. Restore costs likewise exceed v8's 1.510-s median. These stalls are fully included in request TTFT/TPOT. P95 direct transition overlap in the arrival→first-token interval was 0.760/0.941 s, approximately the full entry transition; this is overlap/mechanism evidence, not a causal decomposition of each request's delay.

Against runtime-static FP16:

- P95 advantage was **8.193/7.263 s**, above the frozen 0.776-s noise threshold in both repeats.
- P95 queue fell from 11.323/10.554 s to 2.795/3.687 s.
- Waiting area fell from 195.50/172.00 to 55.75/50.75 request-seconds (71.5%/70.5% reductions).
- Preemptions fell from 12/12 to 5/2.
- Completed throughput was 0.6247/0.6179 versus 0.6232/0.6216 requests/s. Each Dynamic run exceeded 98% of FP16, but its two-repeat mean was lower by 0.00109 requests/s (0.18%), failing the frozen “mean no lower” clause.
- Fixed-reference KV>=0.95 dwell was 31.50/43.29 s versus 43.52/41.53 s. Repeat 0 improved; repeat 1 was 1.75 s worse. This fails the frozen “lower in both” mechanism clause even though queueing and preemption relief aligned strongly.

Dynamic versus always-AWQ P95 gaps were -16.326 s in repeat 0 and +2.969 s in repeat 1. Repeat 1 is the interpretable steady-state trade: Dynamic preserved FP16 until the 47.262-s trigger and paid 0.941 s to enter, so it did not match the 2.253-s always-AWQ tail. Repeat 0's static AWQ instability reverses that ordering and cannot be explained by controller delay. The experiment therefore quantifies but cannot give one stable always-AWQ latency penalty.

## Precision occupancy and fidelity mechanism

Headline Dynamic repeat 0 served 31 requests never exposed to AWQ, 24 partially exposed, and 9 fully exposed; repeat 1 served 13/22/29. AWQ executed 30.7%/58.8% of output steps. This variation follows the causal entry time rather than a workload label.

Paired against runtime-static FP16:

| condition | position-aligned token agreement | mean common prefix | exact-output match |
|---|---:|---:|---:|
| Static AWQ-W4-16 | 31.81% | 154.30 / 512 tokens | 27.34% |
| Dynamic | **81.96%** | **416.63 / 512 tokens** | **78.91%** |

These are fidelity/mechanism measures only. They show that lower AWQ exposure leaves Dynamic much closer to FP16 generation; they do not by themselves establish better semantic answers.

## DuReader semantic quality

The two serving-repeat macro F1 values were:

| condition | repeat 0 | repeat 1 | 64-question aggregate | paired 95% CI |
|---|---:|---:|---:|---:|
| Runtime-static FP16 | 15.255% | 15.255% | 15.255% | [11.466, 19.333] |
| Runtime-static AWQ-W4-16 | 13.316% | 15.672% | 14.494% | [10.736, 18.543] |
| Dynamic | 15.870% | 14.997% | **15.433%** | [11.242, 19.930] |

The aggregate averages the two executions of each question first, then macro-averages/bootstrap-resamples exactly 64 unique questions. Dynamic-minus-static-AWQ is **+0.939 percentage points**, but its paired 95% CI is **[-1.582,+3.819]**. Dynamic-minus-FP16 is +0.178 pp with CI [-0.959,+1.617]. Semantic improvement over static AWQ is therefore unresolved. The large fidelity separation is supportive mechanism evidence, not permission to force an F1 claim.

## Low→high→low reversibility

The deterministic phase boundaries were: initial low `[0,100.625)`, high starting 100.625 s, recovery starting 175.0 s, and final recovery arrival 265.5625 s. No artificial idle gap was inserted.

Both Dynamic repeats correctly stayed FP16 throughout the initial low phase and entered AWQ only after high pressure:

- Repeat 0 requested entry at 133.260 s and completed at 134.018 s. It never satisfied the release rule before measured service ended at 295.12 s. At the last 295.000-s controller sample, waiting mean and preemption conditions passed, but only 87.50% of the trailing window had `used/1768 <=0.70`, below the frozen 90% requirement. The next qualifying tick would have been after request completion and was intentionally not manufactured.
- Repeat 1 requested entry at 110.010 s and completed at 110.932 s. Release first became true at 295.751 s, 30.19 s after the final recovery arrival. Restore completed at 297.721 s while the final recovery request was still active; that request then produced seven contiguous FP16 output steps and completed 0.343 s later without re-prefill. This is real round-trip state preservation, but it fails the predeclared useful-recovery requirement because no later recovery request remained to arrive.

Thus one repeat did not restore, and the other restored too late. Neither completed FP16 restoration before the final recovery arrival. The workload did contain 90.56 s of recovery arrivals, but the 512-token synchronized cohorts kept fixed-reference KV pressure above the release dwell limit almost to the end. No oscillation occurred (one transition in repeat 0, one ordered round trip in repeat 1), but the frozen release policy did not provide useful repeated recovery.

## Correctness, capacity, and stability audit

- All seven measured Dynamic transitions reported success, exact target precision/capacity, and matching sampled logical KV digests around physical resize.
- Entry expansion published 4,170 physical/scheduler blocks; restore returned to 1,759. Stable-state telemetry never violated allocator bounds.
- There was no request loss, CUDA OOM, failed remap, duplicate/invalid decoding position, changed engine request ID, or extra output forward.
- High Dynamic restored with one active request in both repeats; phased repeat 1 restored with one active recovery request and executed seven valid post-restore FP16 steps.
- No extra transition/chatter occurred.
- The predeclared cross-repeat restored-HBM spread check is unavailable for the phased workload because only one repeat restored. It fails as missing evidence rather than being inferred from the high workload or v8 cycles.

`analysis/run_validation.json` passes every raw run. The frozen completion audit hashes the protocol/controller/analyzer/audit, v7 policy helper, unchanged v8 substrate, historical namespaces, and workloads, then directly validates every selected raw run. A separate post-result verifier (`independent_verification.json`, 103/103 checks) hashes every raw file and independently recomputes request arrays, arrivals, capacities, controller windows, serving metrics, performance gates, F1/bootstrap CI, fidelity, phased outcome, headline values, and both decisions.

## Frozen gate outcome

Passed:

- 18/18 raw validity and same-runtime envelope;
- low false-entry and FP16-like noise gates;
- headline causal AWQ entry in both repeats;
- headline P95 improvement beyond run noise in both repeats;
- headline queue/area relief, preemption alignment, and per-repeat 98%-throughput floor;
- transition success/KV/request preservation;
- no oscillation.

Failed:

- headline fixed-reference KV>=0.95 dwell was not lower in repeat 1;
- headline two-repeat mean throughput was 0.18% below FP16;
- useful phased restoration failed in both repeats;
- two-repeat restored-HBM evidence was consequently unavailable;
- Dynamic-minus-AWQ semantic F1 CI did not exclude zero.

These failures are retained. No post-result threshold, longer recovery, extra repeat, state, or success-rule change was made.

## Reproduction

From the repository root:

```bash
AWQ_ENV=/nfs/home/s314511048/.venvs/morphserve-vllm0112
AWQ_EXT=/nfs/home/s314511048/.cache/morphserve/swiftllm-c-torch29/lib
MODEL=/nfs/home/s314511048/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
AWQ_MODEL=/nfs/home/s314511048/.cache/morphserve/llama31-8b-autoawq-w4-g128-zp

# Raw matrix (resume verifies successes and preserves failed attempts).
CUDA_VISIBLE_DEVICES=5 PYTHONPATH="$AWQ_EXT:$PWD/swiftLLM" \
  "$AWQ_ENV/bin/python" -u -m benchmark.run_closed_loop_plan \
  --plan benchmark-results/closed-loop-runtime-v9/run-plan.json \
  --manifest benchmark-results/closed-loop-runtime-v9/protocol_manifest.json \
  --model-path "$MODEL" --quantized-model-path "$AWQ_MODEL" \
  --output-dir benchmark-results/closed-loop-runtime-v9/runs \
  --log-dir benchmark-results/closed-loop-runtime-v9/logs \
  --status benchmark-results/closed-loop-runtime-v9/execution_status.json --resume

# Regenerate tests, all tables/figures/decisions, and independent audit.
benchmark-results/closed-loop-runtime-v9/regenerate.sh
```

Exact commands are in `benchmark-results/closed-loop-runtime-v9/execution_commands.json`. Timelines are `analysis/headline_latency_queue_kv_state_timeline.png` and `analysis/phased_controller_timeline.png`; independent recomputation is `independent_verification.json`.

## CONFIRMED FINDINGS

- The frozen controller avoided AWQ in both low-pressure repeats and causally entered AWQ in both high-pressure repeats without workload/RPS/future/outcome/quality inputs.
- After charging 0.760/0.941-s entry stalls, Dynamic repeatedly reduced headline P95 TTFT by 8.193/7.263 s versus same-envelope runtime-static FP16, far above the 0.776-s run-noise threshold.
- Queue P95, waiting area, and preemptions all fell materially in both high repeats; the high-pressure effect is pressure relief, not a faster AWQ prefill kernel.
- All 1,920 requests, 983,040 output steps, engine IDs, positions, physical capacities, and sampled logical KV states remained valid. One phased active request continued for seven FP16 steps after real restoration.
- Dynamic limited headline AWQ output-token exposure to 30.7%/58.8% and was much closer to FP16 output than always-AWQ by token agreement, common prefix, and exact match.
- The exact release hysteresis did not restore usefully in either phased repeat. This is a measured policy/workload interaction, not a transition-substrate failure.

## SUPPORTED BUT UNCERTAIN FINDINGS

- Dynamic aggregate DuReader F1 was 0.939 pp above static AWQ, consistent with reduced AWQ exposure, but the paired 95% CI [-1.582,+3.819] does not establish semantic superiority.
- Dynamic's headline mean throughput was only 0.18% below FP16 and each repeat passed the 98% floor, but the frozen no-lower-mean clause still fails.
- Fixed-reference KV pressure broadly fell after entry, but repeat 1's >=0.95 dwell was 1.75 s longer than FP16, so the required repeat-wise KV mechanism alignment is not confirmed.
- Static AWQ's 21.395/2.253-s P95 split indicates severe same-envelope variability. The raw result is valid, but its thermal/runtime cause is unmeasured.

## BLOCKED QUESTIONS

- Whether a longer pre-registered recovery phase would let the unchanged ten-second release rule restore while later arrivals remain is unanswered; extending this final workload after results would be post hoc.
- Whether another independently pre-registered release policy could preserve hysteresis while restoring earlier is outside this reproduction stage.
- The cause of static AWQ repeat-0 slowdown is blocked by absent power/clock/temperature telemetry and the prohibition on selective extra repeats.
- Generalization beyond this trace segment, 1,024/512 shape, RTX 3090, checkpoint, AWQ artifact, and strict-FCFS single-GPU process remains unmeasured.

## REMAINING UNCERTAINTY

- Two fresh runs are the independent unit and expose substantial synchronized-cohort variance, especially for static AWQ and low FP16.
- The 64-question paired quality interval remains wide; repeated serving executions reduce run noise but do not create more independent questions.
- Transition overlap is directly measured, but trigger-delay versus hot-stall versus downstream batching is not a randomized causal mediation decomposition.
- Only one phased repeat returned to FP16, so v9 cannot independently repeat the restored-HBM equal-state check even though v8's three-cycle test passed.

## SYSTEMS ADAPTATION DECISION

**NO-GO under the frozen v9 rule.** The causal entry and high-pressure latency adaptation mechanism is strongly confirmed, but the complete low→high→low controller does not meet the required useful restoration in either phased repeat. The repeat-1 fixed-reference KV-dwell and two-repeat mean-throughput clauses also fail. The repository therefore answers that v7's static crossover and v8's manual morphing combine into a real one-way pressure-relief mechanism, but not yet into the predeclared complete closed-loop runtime-adaptation claim.

## QUALITY–LATENCY TRADE-OFF DECISION

**NO-GO.** Dynamic retains the repeated latency advantage over FP16 and is measurably less exposed/divergent than static AWQ, but semantic DuReader F1 is **SUPPORTED / UNCERTAIN**, not separated from static AWQ. Because the systems decision is NO-GO and the Dynamic-minus-AWQ paired F1 interval includes zero, the stronger quality–latency claim is not established.
