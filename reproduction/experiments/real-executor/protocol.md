# Protocol: transactional real GPU executor integration

Status: pre-registered before implementation/execution.

## Objective

Connect `AdaptiveCoordinator` to a real normalized Llama 3.1 8B executor for profiled layers `[25,24,26]`, using real AutoAWQ W4 copies, physical KV expansion, explicit-region mapping selection, rollback, and safe recovery.

## Frozen procedure

- Accuracy mode (1 layer/action) with the already frozen controller persistence.
- Preload/register FP16 and W4 variants for layers 25, 24, 26 only.
- Initialize a small real KV pool; no active requests during transactional action tests.
- Inject failure on the first KV expansion after W4 copy. Require coordinator/executor rollback to exact FP16 bytes, original KV capacity, zero active layers/groups, and unchanged FCFS queues.
- Disable failure; issue three persistent-pressure actions to activate 25→24→26. Require three physical KV groups and automatic explicit-region fallback because the third address violates descending fixed stride.
- Issue recovery actions; remove groups and restore in LIFO order 26→24→25 only after all blocks are free.

## Gate

1. Injected expansion failure leaves exact FP16/capacity/state and records reverse rollback.
2. Successful actions follow `[25,24,26]`, use real WQLinear modules, and physically increase block capacity.
3. Executor selects explicit-region mapping for the non-descending third group.
4. Recovery shrinks each group before restoring its layer; final bytes/logits are exact to initial FP16.
5. Coordinator and executor active-layer counts agree; FCFS queue order never changes.
6. Process exits 0, vendor manifests unchanged, and memory/capacity/action logs saved.

## Boundary

No active requests or concurrent kernels in this transactional pilot. It validates real action/rollback plumbing, not workload latency, admission, or quality.
