# Static frontier v5 protocol

Status: **pre-registered before v5 GPU runs** on 2026-09-12 UTC.

## Question and boundaries

This experiment asks whether the corrected commit-`9b3814a7` SwiftLLM
Llama-3.1-8B FP16/NF4 substrate contains a stable static quality–latency
frontier that justifies a later dynamic experiment. It does **not** implement
dynamic adaptation, runtime layer swapping, or a controller. It preserves the
v2 and v4 namespaces and writes only under `static-frontier-v5`.

The FP16/Hugging-Face parity gate is inherited from
`benchmark-results/table5-substrate-v4/substrate-evidence/fp16-hf-parity.json`;
it will not be reopened unless new evidence falsifies it. The W4 conditions use
the validated NF4 front-to-back proxy. Decode uses low-bit
`bitsandbytes.gemv_4bit`; multi-row prefill uses the disclosed dequantizing
fallback.

The machine-readable contract is
`benchmark-results/static-frontier-v5/protocol_manifest.json`.

## Frozen inputs and common model protocol

All conditions use the same checkpoint/tokenizer, frozen DuReader prompts and
references, 1024-token prompts, exactly 512 greedy output steps, seed 2025,
front-to-back layer order, scheduler, and engine shape. One 8-token request is
run before the measurement clock to remove tokenization-actor and first-kernel
startup from measured serving tails.

The canonical 106-row source is
`benchmark-results/static-quantization-quality-latency/input/workload.jsonl`
(SHA-256 `5811f789...5442337`). Derived workload hashes and exact IDs are in
`benchmark-results/static-frontier-v5/input/workload_metadata.json`.

## Quality characterization

Each of FP16, W4-8, W4-16, and W4-32 runs all 106 requests sequentially so
queueing cannot affect generation quality. The primary result is macro
DuReader character-overlap F1. Every W4 request is paired to FP16 by stable
request ID. The report will include every paired delta and a fixed-seed,
10,000-resample paired percentile-bootstrap 95% confidence interval.

The predeclared secondary mechanism metric is FP16-vs-W4 output-token
agreement: position-aligned token match rate, exact-sequence rate, and longest
common prefix. It is not a replacement for F1. If F1 improves or is
non-monotonic, no example will be removed or retuned; output agreement will be
used only to determine whether increasing W4 coverage changes model behavior.

## FP16-only serving calibration and freeze rule

Before any v5 W4 run, FP16 alone runs the same dense 64-request frozen trace
(source sequences 42–105) at time-dilation scales 24, 16, 12, and 8 relative
to the already frozen 1.75x offsets. These correspond to nominal average loads
0.1667, 0.25, 0.3333, and 0.5 requests/s. Calibration uses the final 1024/512
protocol, not a shortened proxy.

In ascending load, the knee is the first point meeting any of:

- strict TTFT violation rate at least 20%;
- P95 queueing delay at least 2 s; or
- peak logical KV utilization at least 0.85.

The knee and its adjacent load points are selected; at a boundary, the nearest
three are selected. The selected scales and near-knee point are committed to
the manifest before any v5 W4 serving or quality run.

## Final serving evidence

All four states run exactly the same 64 requests at exactly the same three
frozen loads. Every state gets a second run at the frozen near-knee point. If
the ordering changes materially, a third near-knee repeat is required. Every
run preserves raw metadata, requests, telemetry, and logs and reports TTFT
P50/P95/P99, strict SLO rate, queueing, stage timing where observable,
throughput, queue depths, logical KV occupancy and available blocks,
swap/preemption events, HBM, request counts, and run variability.

## Mechanism and state eligibility

Each state receives two fresh profile probes and a seven-sample model
microbenchmark. The mechanism table reports persistent allocation after load,
peak temporary profile/prefill workspace, safe KV blocks/token slots, 1024-token
prefill, one-token decode, and the serving saturation point.

A W4 state is ELIGIBLE only when:

1. all 64 requests complete and repeated profiles agree within 5%;
2. it has more than 5% additional safe KV blocks than FP16, or repeated serving
   evidence identifies a different pressure-relief mechanism;
3. both near-knee repeats improve both P95 TTFT and strict SLO rate relative to
   matched FP16 repeats;
4. the mean P95 improvement exceeds ordinary run variation (the larger
   within-state P95 range, or 5% of FP16 mean if both ranges are zero); and
5. queue/KV/HBM evidence supports the mechanism.

Lower resident weight memory or a larger quantized-layer count alone never
makes a state eligible. FP16 is the reference state.

## Dynamic decision

**GO** requires at least one eligible W4 state plus either a paired quality
distinction or independent output-agreement distortion. Otherwise the result
is **NO-GO**, with the smallest demonstrated technical blocker named. Exact
paper checkpoint, translated subset, LIS ordering, hidden interval, L4, and
AWQ reproduction remain limitations but are not selection targets.
