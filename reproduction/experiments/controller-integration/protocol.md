# Protocol: controller-to-executor integration seam

Status: pre-registered before implementation/tests.

## Objective

Connect the frozen monitor/controller core to an executor command seam without modifying FCFS scheduler queues. Use a fake executor to verify action ordering, profile order, rollback, and shrink safety before real GPU actions.

## Contract

- Coordinator receives a scheduler snapshot containing ordered waiting/running/swapped request IDs and paper metrics.
- Positive layer delta selects the next unquantized layers from a frozen offline profile, copies W4 first, then expands KV. If KV expansion fails, restore those layers and leave coordinator state unchanged.
- Negative delta selects the most recently quantized layers, shrinks KV first, and restores FP16 only if shrink confirms no occupied/in-flight block remains.
- No command may reorder/add/drop scheduler request IDs.
- Executor returns explicit success/failure; exceptions fail closed and are logged.

## Gate

1. Persistent pressure issues actions in frozen profile order.
2. Expand occurs only after W4 copy; failed expand rolls back FP16 in reverse order.
3. Recovery calls shrink before restore; failed shrink defers restoration.
4. Waiting/running/swapped FCFS order is byte-for-byte unchanged by every controller action.
5. Quantized-layer state changes only after successful coordinated actions.
6. Accuracy/default/performance command sizes remain those frozen in controller protocol.

## Boundary

Fake-executor integration only. It does not establish GPU copy/KV behavior, workload latency, or scheduler admission correctness under real concurrency.
