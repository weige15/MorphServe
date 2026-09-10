# MorphServe Phase-1 Experimental Reference

## Objective

Before implementing any new scheduler, reproduce the saturation
behavior of the full-precision SwiftLLM baseline under increasing
request arrival rate.

## Target phenomenon

As offered request rate increases:

request load increases
-> GPU/KV-cache capacity pressure increases
-> new prefills or ongoing decoding cannot obtain sufficient capacity
-> requests accumulate in the waiting queue
-> queueing latency increases
-> TTFT increases sharply.

The saturation point is the load level at which GPU memory/KV capacity
becomes insufficient to schedule new requests for prefilling or sustain
ongoing decoding without waiting/preemption.

## Primary experiment

Sweep offered requests per second (RPS) while holding all other serving
and workload parameters fixed.

Primary plot:
- RPS vs P95 TTFT

Supporting plots:
- RPS vs P95 queueing delay
- RPS vs logical KV-block utilization
- RPS vs completed throughput
- optionally RPS vs TPOT

## Required evidence for a memory/KV saturation claim

Do not infer memory saturation from TTFT alone.

A defensible claim should show correlated evidence that:

1. offered RPS increases;
2. logical KV-block utilization approaches capacity and/or the scheduler
   cannot admit new prefills;
3. waiting queue and queueing latency grow;
4. P95 TTFT develops a clear knee or abrupt increase.

Physical HBM usage is supporting evidence only because SwiftLLM
preallocates KV-cache memory.

If TTFT/TPOT degrade while logical KV capacity remains available,
characterize the bottleneck as compute-bound or mixed instead.

## Paper comparison policy

This experiment reproduces the qualitative behavior represented by
MorphServe Figure 1(b) and Figure 6.

Exact RPS, TTFT, hardware, and saturation values do NOT need to match
the paper.

The paper uses a 2-second TTFT SLO, but crossing exactly 2 seconds is
not a Phase-1 completion requirement.

## Non-goals

Do not implement:
- a new scheduler
- MorphServe
- layer swapping
- quantization
- KV resizing
- admission optimizations
- overload mitigation

The purpose of Phase 1 is to expose and measure the baseline failure
mode, not improve it.
