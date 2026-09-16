# Protocol: real multi-request reclaimed-block ownership and recovery

Status: pre-registered before execution.

## Objective

Verify that real executor recovery respects occupied reclaimed blocks across multiple request IDs, preserves request/block-table state when shrink is refused, and completes only after those requests free their blocks.

## Frozen procedure

1. Load local Llama 3.1 8B, activate real W4/KV groups `[25,24,26]` through the accuracy-mode coordinator.
2. Allocate sequence 0 to length 20 with block size 4 (five blocks: original IDs 0–3 plus reclaimed group-0 ID 4). Allocate sequence 1 to length 4 (next reclaimed ID 5).
3. Write a sentinel into group-0 K/V block 0.
4. Recover layer 26 then 24 (their groups are free). Attempt layer-25 recovery while IDs 4/5 are occupied; require refusal, unchanged block tables/counts/sentinel, and no scheduler preemption/reordering.
5. Free sequences 0/1 through the real BlockManager, retry persistence recovery, and require group-0 shrink plus exact FP16 restoration.

## Gate

- Real block tables allocate `[0,1,2,3,4]` and `[5]` as predicted.
- First two LIFO groups shrink/restore; occupied group 0 refuses.
- Refusal preserves request allocation counts, block-table rows, sentinel bytes, exact allocator/cache metadata, executor/controller/group layers, and FCFS queues.
- After real frees, final recovery succeeds; all capacity/layer state returns to FP16 baseline and region bytes/logits are exact.
- All layer-ready events are consumed; ordinary preemption/swap count is derived from swapped-queue growth and reported separately.

## Boundary

Uses real GPU block ownership and storage but does not execute two concurrent language-model decode streams or timed arrivals.
