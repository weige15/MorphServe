# Transactional real GPU executor result

> **Historical evidence only.** The passing run is preserved under `results-before-atomic-repair/`; it predates the later asynchronous copier and atomic post-shrink recovery/poisoning repairs and is not current-revision GPU validation. `results/` is intentionally absent until the queued current runner succeeds.

All gates of that historical local Llama 3.1 8B run passed:

- injected first KV expansion failure rolled back W4 layer 25 to exact FP16, original 4-block capacity, zero active layers/groups, unchanged FCFS queues;
- three successful accuracy-mode actions activated real W4 layers `[25,24,26]` and attached three physical groups of 615 blocks each;
- capacity grew from 4 to 1,849 blocks with all blocks free in this no-request pilot;
- non-descending third address automatically selected explicit-region remapping;
- recovery order was shrink/restore `26→24→25`;
- final capacity/state returned to 4 FP16 blocks/layers and logits were bit-exact to initial FP16;
- controller/coordinator/executor layer counts and waiting/running/swapped queue order stayed consistent.

The action log explicitly records morph, injected failure, rollback, expansions, LIFO shrinks, and restores.

## Classification

**Historical approximate/modified-condition transactional GPU integration.** At the then-executed revision, real low-bit copies, physical capacity, arbitrary profile-order fallback, rollback, and recovery were integrated. No active requests/concurrent kernels were present; the current atomic revision awaits a GPU rerun, and performance remains open.

Runner wall was 35.34 s including model loading; no serving latency claim is made.
