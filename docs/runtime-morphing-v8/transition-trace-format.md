# Runtime morphing v8 transition timeline and trace format

Raw transition records are JSON Lines in `benchmark-results/runtime-morphing-v8/raw/transition-traces.jsonl`. One line is one complete boundary transition; per-layer events are nested in execution order under `weight_transition.layers`.

## Boundary timeline

`FP16_TO_AWQ_MARLIN_W4_16`:

1. `started_ns`, state/context/HBM snapshot.
2. `boundary_cuda_sync_ns`: all default and decode-piggyback CUDA work drained.
3. `weight_transition.layers[0..15]`: prepared immutable tensors copied H2D, runtime workspace recreated, replacement installed, outgoing dense tensors released.
4. `memory_after_weights`: proof point before KV growth.
5. `kv_resize`: active logical KV sampled, physical extension allocated, allocator bitmap extended, logical KV sampled again.
6. Engine publishes physical total to scheduler.
7. `ended_ns`, `elapsed_ns`, final state/context/HBM snapshot.

`AWQ_MARLIN_W4_16_TO_FP16`:

1. The pending request holds new admissions/swap-ins until allocated blocks fit base.
2. `started_ns`, context/HBM snapshot at the legal boundary.
3. `boundary_cuda_sync_ns`.
4. `kv_resize`: sampled logical KV; extension blocks copied D2D to free base IDs; valid block-table slots remapped; digest rechecked; allocator shrunk; extension tensors released.
5. `memory_after_kv_shrink`.
6. `weight_transition.layers[0..15]`: base norms/dense matrices copied H2D, installed, and AWQ tensors released.
7. Engine publishes base physical capacity and resumes admissions.
8. `ended_ns`, `elapsed_ns`, final state/context/HBM snapshot.

## Fields

| Field | Meaning |
|---|---|
| `transition_id`, `direction`, `status` | Process-local sequence, explicit target, and success/failure result |
| `started_ns`, `ended_ns`, `elapsed_ns` | monotonic `time.perf_counter_ns` boundary timestamps |
| `precision_before`, `precision_after` | runtime state, never inferred from the startup config |
| `active_request_count`, `used_kv_blocks` | active decode/swap state and physically allocated KV at start |
| `engine_context_before/after.active_request_states` | stable benchmark ID, engine request ID, prompt/output lengths, next position |
| `base_blocks`, `physical_blocks_after` | physically backed capacity, not a logical scheduler-only value |
| `scheduler_visible_blocks_before`, `engine_context_after.scheduler_visible_blocks` | publication ordering evidence |
| `memory_*` | CUDA allocated, reserved, max allocated, driver free, and total bytes |
| `boundary_cuda_sync_ns` | explicit boundary drain time |
| `weight_transition.h2d_bytes` | immutable incoming selected-layer bytes |
| `weight_transition.layers[*]` | layer ID, elapsed/sync time, H2D/D2H bytes, old/new backend, before/after HBM |
| `kv_resize.extension_blocks/extension_bytes` | newly allocated physical segment |
| `kv_resize.remapped_blocks/d2d_bytes` | safe shrink copies |
| `kv_resize.integrity_before/after` | sampled logical `(request, sequence-block)` SHA-256 records and D2H bytes |

Initialization checkpoint reads, Marlin repacking, and host preparation are separately recorded as `runtime_preparation_trace`; they are excluded from hot transition latency.

Derived flat timing rows are in `analysis/transition_costs.csv`, and controlled three-repeat direction summaries are in `analysis/transition_cost_summary.json`.
