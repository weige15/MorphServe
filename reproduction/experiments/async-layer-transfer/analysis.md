# Asynchronous layer-transfer correctness pilot

Five CUDA tests pass on the public seams frozen in the protocol.

- A 4,194,304-byte pinned-host copy was enqueued in 0.206 ms while the injected prior-use event was still unfinished.
- The separate-stream copy itself measured 0.578 ms between CUDA events.
- The returned typed view retained the registered destination address and all bytes matched after the just-in-time wait.
- A synthetic `LlamaModel._forward` waited immediately before the affected layer, observed the copied value, consumed the pending event, and recorded a new end-of-forward lifetime event.
- Pageable and oversize sources failed closed.
- Async-model forwards do not accumulate the candidate blocking-restore event records; the single Python end-of-forward event is used instead.
- A failure on the second expansion publishes a current-stream rollback barrier before restoring metadata.
- Fifteen transaction/coordinator tests verify all-before-mutation shrink validation, partial morph rollback, exact post-shrink compensation, false/raised rollback poisoning, rollback-copy double failure, restore prevalidation, FCFS preservation and action ordering.
- Three CUDA-activity analyzer tests reject enclosing intervals, wrong copy sizes and same-stream activity, and require a size-matched H2D/kernel intersection on distinct streams.
- Three rebuilt C++ tests include a misaligned registered-region guard, preventing unsigned reclaimed-capacity underflow.

This establishes non-blocking host enqueue, event order, same-address views, transaction invariants, and the model-use barrier on a small CUDA region. It is supporting correctness evidence only: full-model attempt 1 was JIT-contaminated and used an insufficient overlap definition. The strengthened retry now treats event intersection only as a bound and requires raw CUDA-activity kernel/H2D intersection; it remains pending adequate GPU headroom.
