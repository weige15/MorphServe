# Transactional real GPU executor result

All corrected gates passed on local Llama 3.1 8B:

- injected first KV expansion failure rolled back W4 layer 25 to exact FP16, original 4-block capacity, zero active layers/groups, unchanged FCFS queues;
- three successful accuracy-mode actions activated real W4 layers `[25,24,26]` and attached three physical groups of 615 blocks each;
- capacity grew from 4 to 1,849 blocks with all blocks free in this no-request pilot;
- non-descending third address automatically selected explicit-region remapping;
- recovery order was shrink/restore `26→24→25`;
- final capacity/state returned to 4 FP16 blocks/layers and logits were bit-exact to initial FP16;
- controller/coordinator/executor layer counts and waiting/running/swapped queue order stayed consistent.

The action log explicitly records morph, injected failure, rollback, expansions, LIFO shrinks, and restores.

## Classification

**Approximate/modified-condition transactional GPU integration.** Real low-bit copies, physical capacity, arbitrary profile-order fallback, rollback, and recovery are integrated. No active requests/concurrent kernels were present; multi-request ownership/OOM during serving and performance remain open.

Runner wall was 35.34 s including model loading; no serving latency claim is made.
