# Completion audit: FP16/AWQ-Marlin crossover v7

Audit basis: the user objective; committed pre-registration; immutable workload
hashes; Git history/diffs; unchanged v2/v4/v5/v6 namespaces; exact local
checkpoint/AWQ artifact; two-state code paths; 22 fixed/supplemental multi-M
state measurements; 36 fresh serving processes; 2,304 request rows; 15,847
periodic telemetry rows; 83,041 forward-batch rows; regenerated CSV/JSON/PNG
analysis; final report; CPU/ABI-environment tests; and an independent machine
audit. A manifest, green test, or aggregate decision is not accepted without
checking that its raw coverage matches every explicit requirement.

## Objective restated as concrete deliverables

1. Preserve every v2/v4/v5/v6 artifact and all validated FP16/AWQ execution,
   checkpoint, quality-data, scheduler, decoding, and KV semantics; create only
   a separate crossover namespace and observational instrumentation.
2. Before new intermediate serving results, freeze enough deterministic loads
   between existing scale 6 and scale 4, exact request contents, two states,
   two independent repeats/state/load, execution limits, run order, noise
   definition, crossover rule, and causal signal candidates.
3. Record every forward batch without changing admission: timestamp, precision,
   prefill/decode counts, tokens/effective M, duration, pre-schedule queue/
   running pressure, safe/used KV blocks, and swap/preemption counters.
4. Before the full matrix, diagnose FP16 versus AWQ-W4-16 prefill compute over
   representative synchronized and queue-coalesced M from about 1K to tens of
   thousands, without optimizing the kernel.
5. Complete exact 64 × 1024-prompt/512-output runs at all frozen points and
   regenerate every required latency, queue, first-prefill, TPOT, throughput,
   queue/KV/block/event, and batch distribution from raw evidence.
6. Apply the frozen repeat/noise/mechanism crossover rule rather than the strict
   two-second reference metric alone, leaving ambiguous points ambiguous.
7. Determine whether a small set of causal queue/KV/preemption signals with
   persistence and release hysteresis separates the regimes without future
   arrivals or final outcomes.
8. Explain whether evidence supports fixed-M kernel regression, capacity-driven
   batch admission, queue/preemption relief, or their combination.
9. End with the required epistemic sections and `CROSSOVER DECISION: GO` or
   `NO-GO`; if GO, name `{FP16, AWQ-W4-16}`, the region, signals, and next
   hysteresis inputs, but do not implement a controller.

## Prompt-to-artifact checklist

| Explicit requirement | Concrete evidence inspected | Result |
|---|---|---|
| Continue after packed-int4-backend-v6 | Git parent `d29cb79`; v7 protocol references v6 findings without replacing them | PASS |
| Preserve v2/v4/v5/v6 | `git diff --quiet d29cb79 --` all four prior benchmark namespaces and v6 docs; machine audit | PASS |
| Separate namespace | Only `benchmark-results/fp16-awq-crossover-v7/` and `docs/fp16-awq-crossover-v7/` contain new experiment artifacts | PASS |
| No runtime adaptation/controller | `crossover_decision.json: controller_implemented=false`; source diff contains no controller/state-switch implementation | PASS |
| No backend/scheduler/KV/model-path change | Scheduler, block manager, model, weight, kernel, and layer diffs from `d29cb79` are empty; only opt-in engine observation and benchmark code changed | PASS |
| Same checkpoint | All 36 metadata rows use path-hash `c5c17a...f318a3`; manifest retains snapshot `d04e592...`; checkpoint component hashes match v6 | PASS |
| Same quality data/request contents | Common content hash `2219f1...e7d038c`; source workload hash `5811f7...442337`; no quality input modified | PASS |
| Existing BurstGPT/DuReader segment | Fixed source sequences 42--105 from the existing workload; endpoint files byte-identical to v5 | PASS |
| Treat v6 scale-4 as exploratory | Protocol and report explicitly do so; no v6 run is counted among 36 v7 repeats | PASS |
| Pre-register fixed grid | Commit `72da2d5` precedes result commit `83996c6`; protocol/manifest freeze nine scales and hashes | PASS |
| Include scale 6/4 endpoints | V7 hashes `4fbf8b...3499e` and `8eb239...b74` equal v5 endpoints byte-for-byte | PASS |
| Enough non-adaptive intermediate points | Seven fixed intermediate scales at 0.25-scale spacing; no points added/removed after results | PASS |
| Exact same trace/request contents across loads | All nine workload files have 64 rows and one content hash; only arrival/scale metadata differs | PASS |
| Two states only | Run plan and all metadata contain only `fp16_0` and `awq_w4_16` | PASS |
| 64 fixed 1024/512 requests | 36 × 64 = 2,304 completed rows; every observed prompt/output length is exact | PASS |
| At least two independent repeats/final point | Exactly two counterbalanced fresh-process repeats per state across nine points | PASS |
| Identical engine settings | All metadata: max batch 32, max tokens 49,152, block 16, CPU blocks 4,096, GPU utilization .99, seed 2025, one warmup | PASS |
| Same RTX 3090 | All 36 rows report NVIDIA GeForce RTX 3090 and requested physical device 5 | PASS |
| Batch instrumentation before experiment | Instrumentation is in protocol commit; first target run validates it; `instrumentation-validation.json` PASS | PASS |
| Instrumentation does not change scheduling | No scheduler edits/callbacks; engine takes read-only snapshots immediately before the unchanged call and appends in memory after forward | PASS |
| Every forward recorded | 83,041 contiguous event indices across runs; metadata event counts equal files | PASS |
| Timestamp and precision | Required in all 83,041 batch events | PASS |
| Prefill sequence/token/effective M | Required fields in every event; identity `M=prefill_tokens+decode_rows` passes every row | PASS |
| Decode sequence count | Required and present for every event; no mixed batches observed, consistent with unchanged scheduler | PASS |
| Prefill duration | Non-null for all 421 prefill events and null for decode-only events | PASS |
| Waiting before schedule / running count | Required pre-schedule fields present in every event | PASS |
| Logical KV and safe/used blocks | Scheduler logical and actual allocator fields present; safe blocks exactly 1,768/4,286 | PASS |
| Preemption/swap counters | Per-step and cumulative swap-in/out/preemption fields present; periodic streams independently retain counters | PASS |
| Multi-M benchmark before matrix | Original paired benchmark command completed before matrix; raw seven-sample files cover M=1,024--32,768 | PASS |
| Actual/expected synchronized M | Original grid includes synchronized burst sizes 1/3/4/6/7/8/9 plus 16/24/32; observed range is inside; bounded M=2,048 supplement is labeled | PASS |
| No kernel optimization | No kernel/worker path changed after diagnosis; report uses measurements only | PASS |
| P50/P95/P99 TTFT every run | 36 rows and required columns in `analysis/serving_runs.csv`; raw request timestamps are sources | PASS |
| Strict TTFT >2 s retained | Count/rate in all 36 rows and full report; not used alone in classification | PASS |
| P50/P95/P99 queueing | 36 complete rows from eligible-to-first-prefill timestamps | PASS |
| First-prefill-to-output distribution | P50/P95/P99 in all 36 rows | PASS |
| TPOT | P50/P95/P99 in all 36 rows | PASS |
| Completed throughput | Request and token throughput in all 36 rows | PASS |
| Waiting/running/swapped queues | P50/P95/P99/max in all rows; raw periodic and pre-schedule streams retained | PASS |
| KV utilization | P50/P95/P99/max/dwell plus safe/used distributions in all rows | PASS |
| Preemption/swap events | Counts per run, batch event times, and periodic cumulative counts retained | PASS |
| Batch/prefill-token distributions | P50/P95/P99, per-M table, and exact histograms for every run | PASS |
| Run-level noise, not correlated requests | Frozen epsilon uses both repeat ranges, 5% FP16 mean, and .111-s floor; request bootstrap is not substituted | PASS |
| Repeated high-pressure P95 advantage | Scale 5 and scales 4.5/4.25/4 pass both-repeat beyond-noise gate | PASS |
| Mechanism-aligned advantage | At every AWQ-preferred point both repeats pass queue P95/area, KV dwell, preemption, and throughput gates | PASS |
| FP16 below crossover equal/better | Scales 6/5.75 are FP16-faster both repeats; scale 5.5 is non-inferior within conservative run noise | PASS |
| Do not move ambiguous points/rule | Scales 5.25 and 4.75 remain `ambiguous`; no post-hoc loads/repeats/classification changes | PASS |
| Strict SLO not crossover criterion | Scale 5 demonstrates lower W4 P95 with worse >2-s rate; report explains boundary effect | PASS |
| Distinguish fixed-M compute | Micro and serving matched-M tables both show W4 prefill regression | PASS |
| Distinguish capacity/admission | Per-load batch count/size/M distributions show larger, less-fragmented W4 batches at scales 6, 5, and 4.5 | PASS |
| Distinguish queue/preemption relief | W4 waiting area 6--21% of FP16 at preferred points; zero versus 7--16 preemptions/run | PASS |
| Combination analysis | Final report separates all three effects and identifies pressure relief as dominant after saturation | PASS |
| Causal observable signals only | Trigger code uses historical periodic queue/KV/preemption/used-block fields; no workload label, future arrival, completion, or latency input | PASS |
| Persistence/hysteresis | Fixed 2--5-second entry windows and ten-second FP16-equivalent release window evaluated | PASS |
| Small candidate set | Three predeclared candidates; two qualify, preemption-only candidate is rejected for four low-regime false fires | PASS |
| Signal separation | Compound and capacity/queue triggers fire 8/8 AWQ-preferred FP16 runs and 0/6 FP16-default runs | PASS |
| No denominator-driven W4 release | Release uses used blocks / 1,768; aligned W4 release delay is 27--51 s after matched entry | PASS |
| Exact state pair/region/signals if GO | Final report names `{FP16, AWQ-W4-16}`, 0.727--0.800 RPS bracket, robust 0.889--1.000 high region, and both trigger/hysteresis inputs | PASS |
| Required final sections | Final report ends with Confirmed, Supported-but-uncertain, Blocked, Remaining uncertainty, and Crossover Decision | PASS |
| Raw-derived reproducibility | `regenerate.sh`, execution commands, all raw streams, analysis script, and audit script retained | PASS |

## Important qualification of the GO

GO means the static conditional ordering and causal entry signals are strong
enough to justify a **separate future controller experiment**. It does not mean
a runtime controller is already safe or implemented. Two intermediate points
remain ambiguous, transition costs are unmeasured, and aligned static W4 traces
are only an observational release/hysteresis check.

The measured bracket is 0.727--0.800 requests/s. The strongest contiguous
high-pressure evidence is 0.889--1.000 requests/s, where every point wins both
repeats by far beyond run noise. The predeclared rule permits ambiguous points
as uncertainty provided there is no confirmed FP16 reversal after the first W4
win and the two highest-load adjacent points pass; both conditions hold.

## Validation commands and observed results

```bash
PYTHONPATH="$PWD/swiftLLM:$PWD/swiftLLM/csrc" \
  /nfs/home/s314511048/.venv/bin/python -m unittest \
  benchmark.test_benchmark benchmark.test_inference_substrate \
  benchmark.test_crossover -v

PYTHONPATH="/nfs/home/s314511048/.cache/morphserve/swiftllm-c-torch29/lib:$PWD/swiftLLM" \
  /nfs/home/s314511048/.venvs/morphserve-vllm0112/bin/python -m unittest \
  benchmark.test_benchmark benchmark.test_inference_substrate \
  benchmark.test_crossover -v

/nfs/home/s314511048/.venvs/morphserve-vllm0112/bin/python -m py_compile \
  swiftLLM/benchmark/*.py swiftLLM/swiftllm/*.py \
  swiftLLM/swiftllm/server/*.py swiftLLM/swiftllm/worker/*.py

benchmark-results/fp16-awq-crossover-v7/regenerate.sh
git diff --check
```

Observed in the final validation pass: legacy CPU-safe suite 18/18 passed;
torch-2.9/vLLM ABI-environment suite 18/18 passed; Python compilation passed;
`git diff --check` passed; old-namespace/scheduler/KV/model/backend diffs were
empty; machine audit **PASS**, 28 checks, 0 failures, decision **GO**.

## Audit conclusion

The objective is covered by direct raw evidence rather than proxy green status.
The result is **CROSSOVER DECISION: GO** for a later controller experiment with
`{FP16, AWQ-W4-16}`. No controller exists in this change.
