# Runtime morphing v8 state and memory ownership

This table freezes the exact selected-layer contract for decoder layers 0–15. Layers 16–31, embeddings, LM head, final norm, and RoPE are unchanged in both states.

## Per-selected-layer objects

| Logical object | FP16 active GPU representation | AWQ-W4-16 active GPU representation | Prepared inactive owner |
|---|---|---|---|
| Attention input norm | base checkpoint FP16 `[4096]`, 8,192 B | AutoAWQ-rescaled FP16 `[4096]`, 8,192 B | matching pinned-host tensor |
| `q_proj` | FP16 `[4096,4096]`, 33,554,432 B | Marlin qweight int32 `[256,8192]`; scales FP16 `[32,4096]`; qzeros int32 `[32,512]`; workspace int32 `[82]`; empty g-index tensors; logical `[4096,4096]` | immutable qweight/scales/qzeros in pinned host; workspace/g-index recreated |
| `k_proj` | FP16 `[1024,4096]`, 8,388,608 B | qweight int32 `[256,2048]`; scales FP16 `[32,1024]`; qzeros int32 `[32,128]`; workspace `[82]`; logical `[1024,4096]` | same rule |
| `v_proj` | FP16 `[1024,4096]`, 8,388,608 B | same shape/layout as `k_proj` | same rule |
| `o_proj` | FP16 `[4096,4096]`, 33,554,432 B | same packed shape/layout as `q_proj`; logical `[4096,4096]` | same rule |
| FFN norm | base checkpoint FP16 `[4096]`, 8,192 B | AutoAWQ-rescaled FP16 `[4096]`, 8,192 B | matching pinned-host tensor |
| fused `up_gate_proj` | FP16 `[28672,4096]`, 234,881,024 B in exact `[up,gate]` order | Marlin qweight int32 `[256,57344]`; scales FP16 `[32,28672]`; qzeros int32 `[32,3584]`; workspace `[82]`; logical `[28672,4096]`; source components are AutoAWQ up then gate | same rule; no separate active `up_proj`/`gate_proj` |
| `down_proj` | FP16 `[4096,14336]`, 117,440,512 B | Marlin qweight int32 `[896,8192]`; scales FP16 `[112,4096]`; qzeros int32 `[112,512]`; workspace `[82]`; logical `[4096,14336]` | same rule |
| layer flags | `quantized=False`; `quantization_backend="awq_marlin"` remains an inactive selector; dense fused MLP path | `quantized=True`; `quantization_backend="awq_marlin"` activates Marlin | immutable metadata |

All Marlin matrices use 4 bits, group size 128, asymmetric zero points, one FP16 activation path, backend marker `awq_marlin`, empty `g_idx`/`g_idx_sort_indices`, and a 328-byte workspace on this RTX 3090. There are six packed matrices per selected layer because up+gate is fused.

Measured/verified selected-layer totals:

- FP16 immutable tensors including both norms: 6,979,584,000 bytes for 16 layers.
- AWQ immutable qweight/scales/qzeros plus both norms: 1,813,250,048 bytes for 16 layers.
- AWQ runtime total adds 96 workspaces × 328 bytes = 31,488 bytes; g-index tensors are empty.
- Predicted active selected-layer reduction: 5,166,302,464 bytes (4.811 GiB).

`benchmark-results/runtime-morphing-v8/raw/capacity-cycles.json` independently copies every active immutable tensor back to CPU and reports byte-exact SHA-256 equality against both prepared variants.

## Process ownership by state

| Object | FP16 state | AWQ-W4-16 state |
|---|---|---|
| Selected active GPU weights | one dense/base copy | one packed/AutoAWQ copy |
| Selected inactive representation | pinned host only | pinned host only |
| Layers 16–31 and non-decoder GPU weights | unchanged FP16 | same objects, unchanged FP16 |
| Marlin source-layout checkpoint tensors | absent after initialization preparation | absent |
| Base K/V segment | retained, IDs `[0, base_blocks)` | retained with identical IDs/content |
| Extension K/V segment | absent (zero-length tensors only) | physically allocated, IDs `[base_blocks, total_blocks)` |
| GPU allocator bitmap | `base_blocks` entries | `total_blocks` entries |
| Block table and per-sequence counts | retained | same objects; expanded IDs are virtual |
| Scheduler capacity | published only after physical shrink: `base_blocks` | published only after physical extension: `total_blocks` |

## Measured capacity ownership

| State/process | Base blocks | Extension blocks | Total physical blocks | Physical K+V bytes | Source |
|---|---:|---:|---:|---:|---|
| Fresh static FP16 | 1,768 | 0 | 1,768 | 3,707,764,736 | `raw/state-fp16.json` |
| Runtime initial FP16 | 1,759 | 0 | 1,759 | 3,688,890,368 | `raw/capacity-cycles.json` |
| Fresh static AWQ-W4-16 | 4,286 | 0 | 4,286 | 8,988,393,472 | `raw/state-static-awq.json` |
| Runtime dynamic AWQ-W4-16 | 1,759 | 2,411 | 4,170 | 8,745,123,840 | `raw/capacity-cycles.json` |
| Runtime restored FP16 | 1,759 | 0 | 1,759 | 3,688,890,368 | `raw/capacity-cycles.json` |

The runtime FP16 gap is nine blocks (18 MiB, 0.51%) versus the static 1,768 reference. Preparing/importing the Marlin runtime before profiling leaves a measured ~18 MiB higher FP16 profile envelope (19.89 versus 19.87 GiB).

A diagnostic 4,286-block dynamic allocation succeeded but left only 11.0 MiB driver-free during the frozen 32-sequence/49,152-token maximum-shape forward and exceeded the 0.99 memory budget by 241,426,105 bytes. It is therefore not advertised as safe. Reducing by 116 blocks produced 4,170 physical blocks; the same maximum-shape forward preserved live cross-segment KV, used 25,041,240,064 driver bytes against a 25,043,083,591-byte budget, and left only 1,843,527 bytes—less than one 2-MiB block—inside the budget. Thus 4,170 is the measured maximum at block granularity for this one-process layout, 116 blocks (2.71%) below static AWQ and 2,411 blocks (+137.1%) above runtime FP16.
