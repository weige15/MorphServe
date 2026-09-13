# FP16/AWQ-Marlin crossover characterization v7 protocol

Status: **pre-registered before any new intermediate-load serving result** on
2026-09-13 UTC.

## Question and fixed boundaries

This experiment tests the v6 hypothesis that the validated FP16 state is
preferable below sustained resource pressure, while the validated AWQ-Marlin
W4-16 state becomes preferable only after FP16 reaches genuine KV/queue
saturation. The v6 scale-4 result is motivation, not confirmation.

Only these static states are permitted:

- `fp16_0`: the unchanged SwiftLLM FP16 path;
- `awq_w4_16`: decoder layers `[0,16)` using the already-validated AutoAWQ
  W4/G128/asymmetric-ZP checkpoint and vLLM 0.11.2 Marlin path, with all other
  layers/components unchanged FP16.

This protocol does not implement a controller, runtime adaptation, runtime
layer swapping, a new quantization backend, scheduler changes, KV changes,
attention changes, checkpoint changes, quality-data changes, or kernel
optimization. All prior v2/v4/v5/v6 files remain untouched. New files live
only under `benchmark-results/fp16-awq-crossover-v7/`,
`docs/fp16-awq-crossover-v7/`, and the minimum observational benchmark code.

## Frozen workload grid

Before either state is run at a new intermediate point, freeze the nine-point
scale grid below. Scales multiply the already-frozen 1.75x BurstGPT offsets;
smaller scale means higher offered load.

| scale | nominal requests/s | role |
|---:|---:|---|
| 6.00 | 0.666667 | existing v5/v6 lower-pressure endpoint |
| 5.75 | 0.695652 | predeclared intermediate |
| 5.50 | 0.727273 | predeclared intermediate |
| 5.25 | 0.761905 | predeclared intermediate |
| 5.00 | 0.800000 | predeclared intermediate |
| 4.75 | 0.842105 | predeclared intermediate |
| 4.50 | 0.888889 | predeclared intermediate |
| 4.25 | 0.941176 | predeclared intermediate |
| 4.00 | 1.000000 | existing v5/v6 high-pressure endpoint |

Every file contains the same frozen source sequences 42--105, the same 64
request IDs/prompts/DuReader references, exactly 1024 prompt tokens and 512
greedy output steps. Only arrival offsets and derived load metadata differ.
The scale-6 and scale-4 files in the v7 namespace must be byte-identical to the
v5 endpoints. No point may be added, removed, or moved after results.

Run two fresh-process repeats per state at every point (36 runs total) on
physical RTX 3090 GPU 5. Use the two counterbalanced passes and explicit order
in `run-plan.json`: pass B reverses pass A's load/state ordering so neither
state nor load is systematically favored by run time. Repeats use identical
request contents and arrivals because generation is deterministic;
independence is fresh process, fresh model load/profile/KV allocation, and a
separately measured execution.

Fixed engine and measurement settings are: block size 16, GPU memory
utilization 0.99, 4096 CPU blocks, max batch size 32, max tokens per batch
49,152, max 128 sequence IDs, max 3,072 blocks/sequence, one 8-token warmup,
0.25-second periodic telemetry, seed 2025, strict FCFS scheduler, exact 512-step
greedy decoding, and a 7,200-second timeout. No run may be silently replaced;
failed attempts remain labeled and a replacement keeps the same state/load/
repeat identity with the failure documented.

## Observational batch instrumentation

Batch observation is opt-in and begins after warmup. Immediately before each
unchanged `Scheduler.get_next_batch` call, read queue, running, swapped, KV
block, and cumulative swap counters without mutating them. For every ensuing
model forward, append one event only after the forward returns. Each event
records at least:

- monotonic scheduling and forward timestamps;
- precision state/backend/layer count;
- prefill sequence count and total prefill tokens;
- effective linear/GEMM `M` (prefill tokens plus one row per decoding sequence);
- decoding sequence count;
- full forward duration and prefill duration for prefill-containing batches;
- waiting/running/swapped counts immediately before scheduling and after the
  scheduler decision;
- profile-derived safe KV blocks, logical used blocks, and logical KV
  utilization before/after scheduling;
- per-step and cumulative swap-in, swap-out, and preemption counts;
- stable benchmark request IDs in the batch.

Observation must not influence scheduling, admission, model inputs, or KV
semantics. Raw events are saved as `batches.jsonl`; periodic `telemetry.jsonl`
and per-request timestamps remain unchanged.

## Pre-serving prefill diagnosis

Before the 36 serving runs, benchmark both exact states on multi-sequence
1024-token prefills with sequence counts `{1,3,4,6,7,8,9,16,24,32}`, corresponding to
GEMM `M={1024,3072,4096,6144,7168,8192,9216,16384,24576,32768}`. The trace's synchronized
bursts contain `{1,3,4,6,7,8,9}` requests; the larger fixed counts diagnose
queue-coalesced batches up to the unchanged max batch size. Use two warmups and
seven synchronized samples per `M`.

The microbenchmark calls the identical model forward with
`ignore_kvcache=True` so it isolates compute/kernel behavior and does not
confound it with each state's different KV capacity. It is diagnosis only: no
kernel, flag, fusion, layout, scheduler, or state change may follow from these
measurements inside v7.

## Required raw-derived reporting

For every serving run, regenerate from raw evidence:

- TTFT P50/P95/P99 and the unchanged strict `TTFT > 2 s` reference violation
  count/rate;
- queueing-delay P50/P95/P99;
- first-prefill-to-first-output P50/P95/P99;
- TPOT P50/P95/P99;
- completed request and token throughput;
- waiting/running/swapped queue distributions and maxima;
- logical KV distributions/maxima plus safe/used block distributions;
- preemption/swap events;
- forward-batch size, prefill sequence, prefill-token/effective-M, decoding
  sequence, and prefill-duration distributions.

All latency percentiles use linear interpolation as in the existing benchmark.
The strict 2-second metric is reported but never used alone to choose a state,
because unloaded synchronized multi-request service is already near two
seconds.

## Frozen crossover decision rule

For each load, pair repeats by repeat index and calculate
`d_r = FP16_P95_r - W4_P95_r`. Define the load-specific P95 noise floor before
classification as:

`epsilon = max(0.111 s, FP16 repeat range, W4 repeat range, 0.05 * FP16 repeat mean)`.

The 0.111-second floor is inherited from v6 before new results. Request-level
bootstrap samples are correlated through shared batches/queue episodes and do
not replace the two independent run units.

A load is **AWQ-preferred** only when all conditions hold:

1. both `d_0` and `d_1` are strictly greater than `epsilon`;
2. in both repeats W4 has lower P95 queueing delay, lower time-weighted waiting
   queue area, and less time at logical KV utilization at least 0.95;
3. W4 preemptions are no greater in either repeat and strictly lower in at
   least one, proving this is pressure relief rather than a compute-only win;
4. W4 completed throughput is at least 98% of matched FP16 in each repeat and
   its two-repeat mean is no lower; and
5. all four runs are complete and protocol-valid.

A lower load is **FP16-default/non-inferior** when FP16 has lower P95 TTFT in
both repeats or both paired differences are within `[-epsilon,+epsilon]`, with
no single AWQ advantage beyond `epsilon`. Within-noise ties remain FP16 by
default because they do not justify quantization or switching. Any mixed-sign,
one-repeat-only, invalid, or mechanism-misaligned result is ambiguous.

`CROSSOVER DECISION: GO` requires one unchanged contiguous boundary in the
ordered grid such that:

- scale 6 is FP16-default/non-inferior;
- scale 4 and at least one immediately adjacent lower-load point are
  AWQ-preferred, so one favorable endpoint is insufficient;
- no higher tested load reverses to FP16 and no lower point has a confirmed
  AWQ win;
- the pressure-alignment gates above hold at every AWQ-preferred point;
- at least one predeclared causal online trigger fires in both repeats of every
  AWQ-preferred point and never in either repeat of any FP16-default point; and
- all 36 runs complete 64/64 exact-length requests with valid batch telemetry.

The measured crossover region is the closed bracket between the highest-load
FP16-default point and the next, lowest-load AWQ-preferred point. If no boundary
satisfies every clause, the result is `NO-GO` and no controller is built. The
point/rule cannot be moved after observing results.

## Predeclared online-signal candidates

Signals are evaluated causally at each 0.25-second telemetry sample, using only
current/past runtime state. Future arrivals, final request outcomes, final
latency, and final load labels are never input features. Entry behavior is
assessed on FP16 runs; W4 traces are used only to assess whether an exit rule
would chatter after relief.

Because W4 has a larger safe-block denominator, release checks use the
runtime-observable FP16-equivalent utilization `used_kv_blocks / 1768` rather
than W4's native utilization. Three candidates are fixed from v6 endpoint
behavior:

1. **Sustained compound pressure:** enter when a causal trailing 3-second
   window has FP16 KV utilization at least 0.95 for at least 80% of elapsed
   time, waiting depth at least 4 for at least 50%, and current waiting at
   least 4. Clear only after ten continuous seconds with waiting-depth mean at
   most 0.5, no preemption, and FP16-equivalent utilization at most 0.70 for at
   least 90% of the window.
2. **Capacity margin plus queue integral:** enter after at least one continuous
   second with fewer than 65 free safe blocks (one 1024-token admission margin)
   and a trailing-five-second waiting area of at least 15 request-seconds. Use
   candidate 1's release rule.
3. **Preemption-confirmed pressure:** enter after at least two preemptions in a
   causal trailing five-second window and trailing-two-second mean waiting at
   least 2. Clear only after ten seconds with no new preemption, waiting mean
   below 0.5, and FP16-equivalent utilization below 0.70.

Report per candidate/load/state whether it fired, first-fire time, episode
count/duration, false entry in FP16-default regimes, missed entry in
AWQ-preferred regimes, and potential W4 exit/chatter. No threshold may be tuned
after results. These are candidate trigger/hysteresis inputs for a later
experiment, not a controller implementation.

## Interpretation constraints

Use prefill compute-only curves together with serving batch events to separate:

1. pure AWQ-Marlin large-M prefill regression;
2. larger admitted W4 prefill batches due to extra KV capacity;
3. queue/preemption relief;
4. combinations of the above.

Do not infer causality from lower W4 KV utilization alone: its denominator is
larger by construction. Use actual admitted M/batch distributions, matched-M
microbench ratios, queue delay, preemption, and throughput together. End the
report with exactly the requested finding/uncertainty sections and
`CROSSOVER DECISION: GO` or `NO-GO`. A GO names only `{FP16, AWQ-W4-16}` and
the predeclared bracket/signals; a NO-GO stops without controller work.
