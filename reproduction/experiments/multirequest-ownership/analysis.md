# Real multi-request ownership/recovery result

> **Historical evidence only.** The passing run is preserved under `results-before-atomic-repair/`; it predates later atomic recovery/poisoning repairs and is not current-revision GPU validation. `results/` is intentionally absent until the queued current runner succeeds.

All gates of that historical run passed:

- sequence 0 allocated blocks `[0,1,2,3,4]`; sequence 1 allocated `[5]`, so IDs 4/5 occupied reclaimed group 0;
- free groups for layers 26 and 24 shrank/restored in LIFO order;
- layer-25 recovery was refused while its group was occupied;
- refusal preserved both allocation counts, block-table rows, K/V sentinel bytes, executor/controller state and FCFS/swapped queue state;
- after real BlockManager frees for request IDs 0/1, recovery succeeded;
- final capacity was 4/4 free blocks, request counts zero, no W4 groups/layers, and FP16 logits bit-exact.

Block-table row storage retains stale IDs after free while `num_seq_allocated_blocks=0`; kernels use the count as ownership authority. This matches candidate BlockManager behavior but should not be interpreted as live ownership.

## Classification

**Historical approximate/modified-condition multi-request ownership reproduction.** At the then-executed revision, real GPU allocation/free and shrink refusal were covered for two request IDs. The pilot does not validate the current atomic revision or execute concurrent model decode, scheduled arrivals, preemption, or token timing. Its static scheduler has no cumulative preemption counter, so it cannot support a measured zero-preemption claim.
