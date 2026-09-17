# Current-revision transactional real-executor result

The current runner `CUDA_VISIBLE_DEVICES=0 reproduction/scripts/run_real_executor_pilot.sh` passed at HEAD `67bbdc2d7094c2b6dcaab6e58c745d1a1a0a815b`, with the runner's recorded source-status sidecar. GPU 0 had 24,124 MiB free before setup, before model load, and after teardown; no external compute process was present in the snapshots.

All current gates passed:

- real AutoAWQ packed `WQLinear_GEMM` modules were installed for layers `[25,24,26]`;
- injected second KV expansion failure acquired a partial group, then rolled back to exact FP16 bytes/logits and the original four-block allocator state without poisoning;
- successful profile order `[25,24,26]` expanded physical capacity from 4 to 1,849 blocks using three 615-block groups and explicit non-contiguous mapping;
- recovery shrank before FP16 restore in LIFO order `26→24→25`;
- final FP16 region bytes/logits, allocator/cache/group state, controller/coordinator/executor state, pending events and FCFS queue state were exact/clean.

The passing machine-readable artifact is `results/metrics.json`; raw stdout/stderr, command, source revision/status, manifest checks and GPU before/pre-run/after snapshots are in the same directory. The earlier `results-before-atomic-repair/` run remains historical evidence and was not overwritten.

## Classification

**Current-revision modified-condition transactional GPU implementation verification.** This validates the independent atomic reconstruction on a real full model. It is not author-code provenance, a paper workload, or a headline latency result.
