# Current-revision multi-request ownership result

The current runner `CUDA_VISIBLE_DEVICES=0 reproduction/scripts/run_multirequest_ownership.sh` passed at HEAD `67bbdc2d7094c2b6dcaab6e58c745d1a1a0a815b`, with the runner's recorded source-status sidecar. GPU 0 had 24,124 MiB free before setup, before model load, and after teardown; no external compute process was present in the snapshots.

All current gates passed:

- request 0 owns block row `[0,1,2,3,4]` and request 1 owns `[5]`, placing reclaimed blocks in active use;
- recovery removes free groups in LIFO order, then refuses the occupied layer-25 reclaimed region without mutating rows, counts, sentinels, allocator state, queues or event state;
- after genuine request/block release, recovery succeeds;
- final capacity returns to four fully free blocks, all W4 groups/layers and pending events are gone, FP16 bytes/logits are exact, and executor/coordinator are not poisoned;
- FCFS and swapped-queue state remain unchanged. `scheduler_preemptions_measured` is explicitly false because this bounded adapter has no valid cumulative scheduler counter.

The passing machine-readable artifact is `results/metrics.json`; raw stdout/stderr, command, source revision/status, manifest checks and GPU before/pre-run/after snapshots are in the same directory. The earlier `results-before-atomic-repair/` run remains historical evidence and was not overwritten.

## Classification

**Current-revision modified-condition multi-request ownership/recovery verification.** This validates real reclaimed-KV ownership and atomic recovery for two requests. It does not measure ordinary scheduler preemption, concurrent serving throughput, or paper workload latency.
