# Controller-to-executor integration result

Eight fake-executor integration tests pass:

- persistent pressure selects layers in frozen profile order and performs W4 copy before KV expansion;
- failed expansion and raised exceptions restore FP16 in reverse order and leave controller state unchanged;
- an executor that already completed internal morph rollback is not restored a second time;
- failed post-shrink restore returns/raises are atomically compensated without losing active state;
- failed pressure rollback returns/raises poison the coordinator and preserve truthful executor/controller state;
- recovery shrinks KV before FP16 restore and defers restore when occupied/in-flight shrink refuses;
- successful recovery restores in reverse active order;
- waiting/running/swapped request-ID order remains unchanged in every action.

## Classification

**Approximate/modified-condition integration seam.** Control/action ordering, rollback, and FCFS non-interference are covered, but the executor is fake. Real GPU integration must connect these methods to the tested AutoAWQ/KV/event primitives and exercise allocation/OOM rollback under concurrent requests before serving claims.
