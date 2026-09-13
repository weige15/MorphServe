# MorphServe runtime morphing v8 final report

Status: **complete — RUNTIME MORPHING DECISION: GO**.

This phase implements and validates an explicit manual, one-process transition substrate for the fixed pair `{FP16, AWQ-Marlin W4-16}`. It does **not** implement a workload-pressure controller, use the v7 trigger candidates, tune crossover thresholds, alter strict-FCFS ordering, change the AWQ backend, or rerun the broad static frontier. All v2/v4/v5/v6/v7 namespaces remain unchanged.

## What is implemented

`Engine` exposes:

```python
await engine.morph_to_awq_w4_16()
await engine.restore_to_fp16()
```

Requests are serialized by one async lock and consumed by the single engine loop between complete forwards. The boundary explicitly synchronizes CUDA before mutation. A direct restore while allocated KV exceeds the retained base is rejected without changing state; with the engine loop running, new prefills and swap-ins are held while existing requests drain.

During initialization, before KV allocation, layers 0–15 are staged in pinned host memory in both final runtime layouts. FP16 staging preserves base norms and fused dense matrices. AWQ staging uses the exact v6 loader/repack path and preserves AutoAWQ-rescaled norms plus final Marlin qweight/scales/qzeros; source tensors are discarded, and workspace/g-index state is recreated on activation. Preparation took 10.800 s and moved 8,792,834,048 bytes D2H once. It is outside the hot transition measurements.

The inactive representation is host-only. A transition copies one incoming layer at a time, installs it in both owning references, and deletes the outgoing GPU representation. The runtime records the old+new coexistence point before deletion. Active immutable selected-layer tensors were copied back and SHA-256 compared against fresh static controls:

- AWQ: byte-exact across 1,813,250,048 bytes;
- restored FP16: byte-exact across 6,979,584,000 bytes.

The complete object/shape/ownership contract is in [state-memory-ownership.md](state-memory-ownership.md), and the legal boundary/KV design is in [transition-architecture.md](transition-architecture.md).

## Real segmented KV capacity

The original FP16 K/V tensors remain the base segment. One independently allocated extension adds a virtual-ID suffix. Existing base IDs and bytes do not move on expansion. KV-store and paged-attention Triton kernels resolve each virtual ID to base or extension on every access. GPU/CPU block swapping partitions virtual IDs and reuses the existing C++ copy primitive with segment-local IDs.

On shrink, live extension blocks are copied one block at a time into free base IDs, only valid block-table slots are remapped, hashes are checked, allocator metadata is shrunk, and extension tensors are released before dense FP16 restoration. Scheduler-visible capacity is published only after the physical operation succeeds.

### Measured capacities

Each block is 2,097,152 bytes across K and V and contains 16 token slots.

| state | base blocks | extension blocks | physical/scheduler blocks | token slots | physical K+V | versus runtime FP16 |
|---|---:|---:|---:|---:|---:|---:|
| fresh static FP16 | 1,768 | 0 | 1,768 | 28,288 | 3,707,764,736 B | static reference |
| runtime FP16 | 1,759 | 0 | 1,759 | 28,144 | 3,688,890,368 B | reference |
| fresh static AWQ-W4-16 | 4,286 | 0 | 4,286 | 68,576 | 8,988,393,472 B | static reference |
| dynamic AWQ-W4-16 | 1,759 | 2,411 | **4,170** | **66,720** | **8,745,123,840 B** | **+2,411 / +137.1%** |
| restored runtime FP16 | 1,759 | 0 | 1,759 | 28,144 | 3,688,890,368 B | returned |

Runtime FP16 loses nine blocks (18 MiB, 0.51%) because importing/preparing Marlin before profiling leaves a measured ~18 MiB larger process envelope than fresh static FP16.

The static 4,286 target was tested but not accepted blindly. It allocated, yet the frozen 32-sequence/49,152-token maximum-shape forward left only 11 MiB driver-free and exceeded the configured 0.99 budget by 241,426,105 bytes. The safe target was reduced by 116 blocks. At **4,170**, the same maximum-shape forward:

- completed 32/32 valid outputs with the full segmented KV allocation resident;
- preserved hashes for a live history spanning base ID 0 and extension ID 1,759;
- reached 24,141,170,688 allocated bytes;
- used 25,041,240,064 driver bytes against a 25,043,083,591-byte 0.99 budget;
- left 1,843,527 budget bytes, less than one 2-MiB block.

Thus 4,170 is the measured maximum at block granularity for this one-process layout, not a logical scheduler-only claim. The diagnostic rejected 4,286 trace is preserved under `benchmark-results/runtime-morphing-v8/diagnostics/`.

## State-preservation results

Two identical prompts were run for 24 greedy output steps in each required condition:

1. all FP16;
2. all static AWQ-W4-16;
3. FP16 prefix then manual AWQ;
4. FP16 prefix, manual AWQ, then manual FP16 restoration.

| condition | request 0 precision steps | request 1 precision steps | prefills/request | request IDs | positions/steps |
|---|---|---|---:|---|---|
| all FP16 | FP16 ×24 | FP16 ×24 | 1 | stable 0/1 | contiguous, +1 |
| static AWQ | AWQ ×24 | AWQ ×24 | 1 | stable 0/1 | contiguous, +1 |
| one-way | FP16 ×5, AWQ ×19 | FP16 ×5, AWQ ×19 | **1** | stable 0/1 | contiguous, +1 |
| round-trip | FP16 ×5, AWQ ×7, FP16 ×12 | same | **1** | stable 0/1 | contiguous, +1 |

Both mixed FP16 prefixes exactly match the all-FP16 control through the execution-time boundary. Every output token is a finite valid vocabulary ID. Batch traces show exactly one prefill for each request; after transition, every model call is a one-token decode at the next position. Both transitions in the round-trip occurred with two active requests, unchanged engine request IDs, and unchanged prompt/output state.

Expansion sampled six active 2-MiB logical blocks in each direction and produced identical before/after SHA-256 digests. A full-model AWQ sequence was then forced to span base block 0 and extension block 1,759. Its first five tokens matched static AWQ. With that live request retained, restoration remapped the table to base IDs `[0,1]`, preserved both complete-block hashes across all 32 layers, retained request ID 0 and position 17, and produced the next valid FP16 token without prefill. Synthetic extension remaps in the remaining two cycles independently repeated exact hash preservation.

Focused CUDA tests also cover:

- prefill and decode writes to extension storage;
- paged attention over one history spanning base and extension;
- segmented GPU→CPU→extension GPU swap round-trip;
- allocator grow/compact/shrink invariants;
- rejection while allocated blocks exceed retained capacity.

## Transition cost and HBM

The controlled one-process distribution contains three repetitions per direction. Initialization preparation is excluded.

| direction | total min / median / P95 / max | layer loop median | layer CUDA sync median | cleanup median | KV resize median | boundary sync median | bookkeeping median | H2D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP16→AWQ | 586.0 / **661.0** / 675.1 / 676.7 ms | 267.2 ms | 212.9 ms | 282.4 ms | 79.3 ms | 0.070 ms | 14.6 ms | 1,813,250,048 B |
| AWQ→FP16 | 1,478.7 / **1,510.2** / 1,584.4 / 1,592.6 ms | 1,044.5 ms | 963.7 ms | 200.6 ms | 251.2 ms | 0.074 ms | 13.8 ms | 6,979,584,000 B |

The maximum allocated peak across all nine transitions was 19,971,111,936 bytes (18.600 GiB), including explicitly sampled moments where outgoing and incoming layer representations coexist. Start/end, per-layer old+new peaks, allocated/reserved/free/total HBM, H2D/D2H/D2D bytes, synchronization, KV resize, active requests, used blocks, and physical/scheduler capacities are all retained in `raw/transition-traces.jsonl`; see [transition-trace-format.md](transition-trace-format.md).

Across three complete cycles, equal-state restored allocated memory differed by only 1,024 bytes from first to last. Reserved memory temporarily varied by 232,783,872 bytes in cycle 1 but returned to within 2,097,152 bytes of cycle 0 in cycle 2; it was not monotonic. This supports no material leak over three measured cycles, not unlimited-duration stability.

V7's eight AWQ-preferred static comparisons saved 2.436–13.018 s/request in P95 TTFT (median 8.601 s). The measured 0.661-s one-way transition corresponds to 0.08 request-equivalents at the median v7 advantage and 0.27 at the weakest. A full 2.171-s round trip corresponds to 0.25 and 0.89 request-equivalents. This is an intentionally simple amortization estimate, not a closed-loop result; it shows that measured transition cost does not obviously erase the prior high-pressure advantage.

## Reproduction

From repository root:

```bash
AWQ_ENV=/nfs/home/s314511048/.venvs/morphserve-vllm0112
AWQ_EXT=/nfs/home/s314511048/.cache/morphserve/swiftllm-c-torch29/lib
MODEL=/nfs/home/s314511048/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
AWQ_MODEL=/nfs/home/s314511048/.cache/morphserve/llama31-8b-autoawq-w4-g128-zp

MORPHSERVE_RUN_CUDA_TESTS=1 CUDA_VISIBLE_DEVICES=5 \
  PYTHONPATH="$AWQ_EXT:$PWD/swiftLLM" "$AWQ_ENV/bin/python" -m unittest \
  benchmark.test_runtime_morphing -v

# Replace CONDITION/OUTPUT for fp16, static_awq, morph, roundtrip, and capacity_cycles.
CUDA_VISIBLE_DEVICES=5 PYTHONPATH="$AWQ_EXT:$PWD/swiftLLM" \
  "$AWQ_ENV/bin/python" -m benchmark.run_runtime_morphing_validation \
  --condition capacity_cycles --cycles 3 \
  --model-path "$MODEL" --quantized-model-path "$AWQ_MODEL" \
  --output benchmark-results/runtime-morphing-v8/raw/capacity-cycles.json

benchmark-results/runtime-morphing-v8/regenerate.sh
```

Every final raw run reports the same RTX 3090 device 5, Git base commit, runtime-source diff SHA-256, and 12 per-file source hashes. Exact commands are in `benchmark-results/runtime-morphing-v8/execution_commands.json`.

## CONFIRMED FINDINGS

- One SwiftLLM engine process really transitions FP16→AWQ-W4-16→FP16 at explicit forward boundaries.
- AutoAWQ norms, packed matrices, dense matrices, backend flags, and runtime aliases restore byte-exactly to their validated static layouts.
- Two concurrent active requests preserve request IDs, KV state, decoding positions, and exactly-one-prefill behavior across both directions.
- Segmented KV increases physically safe capacity from 1,759 to 4,170 blocks (+137.1%), and scheduler visibility follows physical allocation.
- A live base+extension history survives extension→base remap and supplies the next FP16 decode token.
- Three repeated cycles return physical capacity and allocated/reserved HBM to the same bounded regime with no monotonic leak.
- Median hot transition costs are 0.661 s into AWQ and 1.510 s back to FP16, below the already-observed high-pressure v7 latency benefit.

## SUPPORTED BUT UNCERTAIN FINDINGS

- Three cycles strongly support bounded short-run stability, but do not establish unlimited soak stability.
- The 4,170-block maximum is measured for this exact process, allocator state, maximum-shape envelope, model, and RTX 3090; another process layout or GPU may have a different safe total.
- The v7 amortization estimate suggests rapid payback, but a future controller experiment must measure transition stalls inside sustained offered load.

## BLOCKED QUESTIONS

- Closed-loop entry/release behavior remains intentionally unimplemented and untested.
- Long-duration transition soak, cancellation during an in-progress transition, and recovery from real CUDA OOM/hardware faults are not established.
- Generalization to other GPUs, checkpoints, tensor parallelism, request shapes, or schedulers is unmeasured.

## REMAINING UNCERTAINTY

- The dynamic safe total is 116 blocks below fresh static AWQ because the one-process Marlin/runtime envelope and allocator reservations coexist; the maximum-shape measurement bounds the gap but does not attribute every byte.
- Segmented paged-attention correctness is directly tested, but its isolated performance overhead versus the original one-segment kernel was not separately benchmarked.
- Reserved-memory variability reached 222 MiB mid-series before returning; longer repetitions are needed to distinguish all allocator-cache modes from rare slow leaks.

## RUNTIME MORPHING DECISION: **GO**
