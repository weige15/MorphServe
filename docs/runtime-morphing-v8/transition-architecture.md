# Runtime morphing v8: transition architecture

Status: **contract frozen before implementation**. This namespace is independent of v2/v4/v5/v6/v7. It does not define or implement workload-pressure triggers, crossover thresholds, or a closed-loop controller.

## Scope and fixed state pair

The only legal runtime states are:

- `FP16`: the base Llama-3.1-8B checkpoint, all 32 decoder layers dense FP16.
- `AWQ-Marlin W4-16`: decoder layers 0–15 from the already-validated offline AutoAWQ W4/G128/asymmetric-zero-point checkpoint, layers 16–31 and non-decoder weights unchanged FP16.

The static v7 reference capacities on the same RTX 3090 are 1,768 FP16 GPU blocks and 4,286 AWQ-W4-16 GPU blocks. Each block contains 16 token slots and occupies 2,097,152 bytes across K and V. These are reference capacities, not permission to over-allocate when current physical memory does not safely support them.

## Legal transition boundary

A transition may run only between complete `LlamaModel.forward` calls, after synchronizing every CUDA stream. The engine event loop accepts an explicit manual request and services it before scheduling the next forward. It never mutates a layer while the default or decode piggyback stream can still reference that layer. No background pressure signal invokes the API.

For restoration, the engine pauses new admissions and swap-ins while active GPU block demand exceeds the retained base capacity. It continues decoding existing requests. Once total demand fits, extension-resident logical blocks may be copied to free base blocks and their block-table entries remapped at the same synchronized boundary. Requests are not discarded or re-prefilled.

## Weight staging and ownership

Alternate representations are prepared once during initialization, before KV allocation:

1. Copy the active dense FP16 representations for layers 0–15 into pinned host memory.
2. Load and Marlin-repack one AWQ layer at a time from the validated offline artifact, copy the final runtime tensors into pinned host memory, then release all temporary/source GPU tensors.
3. Empty the CUDA allocator cache before profiling and allocating the base FP16 KV pool.

The hot transition path performs pinned-host-to-device copies only; it does not synchronously reload or repack the checkpoint. At each layer boundary the replacement is installed in both `LlamaWeight.layers[i]` and `LlamaTransformerLayer.weight`, then the old GPU representation is deleted. The inactive state has no hidden GPU copy.

AutoAWQ rescales both input and post-attention norms. Therefore AWQ activation installs the AWQ checkpoint's norms with its packed matrices, while restoration installs the original base-checkpoint FP16 norms with the original dense matrices.

## Segmented KV design

The original contiguous allocation becomes the immutable base segment with virtual IDs `[0, base_blocks)`. After dense layers 0–15 are released and AWQ layers are active, one extension segment may be allocated with virtual IDs `[base_blocks, total_blocks)`. The GPU `BlockManager` expands its free bitmap to the same virtual-ID range while preserving the existing block table and all pre-existing IDs.

KV store and paged-attention kernels resolve every virtual block ID at use time:

- IDs below `base_blocks` address the original contiguous base K/V tensors.
- IDs at or above `base_blocks` address the extension tensors at `virtual_id - base_blocks`.

GPU/CPU block swapping partitions copies by segment and passes physical segment-local IDs to the existing C++ copy routine. Scheduler capacity is increased only after both extension tensors and the expanded allocator are ready.

Before shrink, allocated extension blocks are copied to free base blocks if total allocation fits the base. Only the affected live block-table entries are remapped. Logical `(request_id, sequence_block_index)` identity and KV bytes remain unchanged. The extension is released only after no live table entry references it; scheduler-visible capacity is reduced only after physical shrink completes.

## Transition sequence

### FP16 to AWQ-W4-16

1. Accept an explicit `morph_to_awq_w4_16()` request.
2. Finish the current forward and synchronize CUDA streams.
3. Record active requests, used blocks, precision state, and HBM snapshot.
4. For layers 0–15, materialize the prepared AWQ variant, atomically replace both references, release the dense variant, and record per-layer copy/time/memory.
5. Verify all 16 runtime layer flags and layouts are AWQ-Marlin.
6. Empty unused CUDA reservations, allocate the physical KV extension, then extend allocator bookkeeping.
7. Publish the new scheduler capacity and complete the manual request.

### AWQ-W4-16 to FP16

1. Accept an explicit `restore_to_fp16()` request and pause new admissions/swap-ins.
2. Continue decoding until active GPU demand is at most the base capacity.
3. Finish the current forward and synchronize CUDA streams.
4. Remap extension-resident live blocks into free base blocks and verify logical KV integrity.
5. Shrink allocator bookkeeping, release extension K/V tensors, and empty unused reservations.
6. For layers 0–15, materialize the prepared base FP16 variant, atomically replace both references, release AWQ tensors, and record per-layer copy/time/memory.
7. Verify all 16 layers use base norms/dense matrices, publish base scheduler capacity, resume admissions, and complete the request.

## Trace contract

Each JSONL transition row contains:

- transition ID, requested/start/end monotonic timestamps, elapsed time, status, and error;
- precision state before/after;
- active request count and used KV blocks at the boundary;
- base, extension, physical, and scheduler-visible block counts;
- HBM allocated/reserved/free/total snapshots before synchronization, after synchronization, after weights, after KV resize, and at completion;
- CUDA synchronization time;
- one row per layer with layer ID, elapsed time, H2D/D2H bytes, old/new backend, and before/after HBM snapshots;
- KV extension/shrink/remap time and bytes copied;
- total H2D/D2H bytes;
- optional sampled logical KV hashes immediately before and after resize.

Initialization preparation has a separate trace so checkpoint reads, one-time Marlin repacking, and D2H staging are not misreported as hot transition latency.

## Safety invariants

- Exactly one precision state is active for each selected layer after a successful transition.
- No selected inactive representation remains on GPU.
- Existing virtual block IDs and contents are unchanged by expansion.
- Shrink is illegal while total allocated GPU blocks exceed base capacity.
- Shrink is illegal until every extension reference has been remapped or freed.
- Scheduler-visible capacity never exceeds physically allocated KV blocks.
- A transition never invokes tokenization, prefill, request replacement, or scheduler policy selection.
- Any failed invariant makes the runtime morphing decision NO-GO.
