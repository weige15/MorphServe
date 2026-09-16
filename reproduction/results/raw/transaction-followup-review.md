## Review

- **Correct:** Multi-layer morph and restore now prevalidate batches and roll back committed layers in reverse order (`reproduction/runtime/morphserve/real_executor.py:48-86`, `165-198`). The focused transaction log records all four executor tests passing.
- **Correct:** KV shrink checks every target group’s occupancy before the first mutation (`reproduction/runtime/morphserve/real_executor.py:138-162`).
- **Correct:** Partial expansion failure publishes a CUDA barrier before metadata is restored (`reproduction/runtime/morphserve/real_executor.py:89-135`). The dedicated CUDA test passes.
- **Correct:** The asynchronous copier waits for prior layer copies and the latest model-use event (`reproduction/runtime/morphserve/autoawq_adapter.py:42-59`), while the model waits immediately before the affected layer and publishes an end-of-forward event (`reproduction/runtime/candidate-python/swiftllm/worker/model.py:308-314,343`).
- **Correct:** The aligned reclaimed-capacity subtraction is guarded (`reproduction/runtime/candidate-csrc/src/memory_manager.cpp:149-157`). Saved evidence records all three rebuilt memory-manager tests passing.
- **Correct:** The pending real pilots now require exact restored region bytes and empty pending-event state (`reproduction/scripts/real_executor_pilot.py:72-83,126-151`; `reproduction/scripts/multirequest_ownership.py:85-106`).
- **Correct:** Full-model attempt 1 remains honestly rejected and is not used as overlap evidence.

- **Finding: P1 — Recovery remains non-transactional when FP16 restoration fails after successful KV shrink.**  
  `AdaptiveCoordinator.step()` first destructively shrinks KV and then calls `restore_fp16()` (`reproduction/runtime/morphserve/integration.py:60-67`). If restore returns false or raises, the removed KV groups are never reconstructed; the exception handler only attempts compensation for `delta > 0` (`integration.py:69-79`). The controller and coordinator continue to report the layers active, but `RealMorphingExecutor.shrink_kv_before_restore()` has already removed their groups and capacity (`real_executor.py:151-162`). A later recovery retries shrink against groups that no longer exist and can remain permanently stuck. Existing `test_successful_recovery_restores_after_shrink` only covers a successful restore (`reproduction/tests/test_controller_integration.py:123`).
  
  **Smallest fix:** Add an atomic executor recovery operation, or have the coordinator re-expand removed groups in original order if restore fails. Mark the executor poisoned if compensation fails. Add tests for restore returning false and raising after a successful multi-group shrink, asserting exact restoration of groups, capacities, active layers, and controller state.

- **Finding: P1 — Failed pressure-action rollback ignores restoration failure and can hide executor/controller divergence.**  
  On expansion failure, the coordinator invokes `restore_fp16(rollback)` but discards its return value (`reproduction/runtime/morphserve/integration.py:50-57,71-78`). It then decrements the controller state regardless. Its invariant compares only controller state with `AdaptiveCoordinator.active_layers` (`integration.py:83-84`), not `executor.active_layers`. A failed restoration can therefore leave real W4 layers active while both controller-facing counts say FP16. Current fake-executor tests always return true from restore (`reproduction/tests/test_controller_integration.py:41-43`).
  
  **Smallest fix:** Require rollback success, compare coordinator and executor active-layer sets, and enter an explicit failed/poisoned state when rollback cannot complete. Add false-return and exception tests for rollback restoration.

- **Finding: P1 — The strengthened overlap gate can still report compute/copy overlap when the copy overlaps only CPU launch preparation or GPU idle time.**  
  `decode_start` is recorded before entering `model.forward()` (`reproduction/scripts/async_full_model_overlap.py:59`), while `pre_wait` is recorded only later in the layer loop (`reproduction/runtime/candidate-python/swiftllm/worker/model.py:308-312`). Their interval therefore includes host-side infer-state preparation, kernel-launch gaps, and possible device idle time. The intersection at `async_full_model_overlap.py:67` only proves overlap with that enclosing pre-wait interval, yet any positive value passes `pre_layer_compute_overlap_observed` (`async_full_model_overlap.py:81`). It correctly excludes the affected-layer wait, but does not prove a CUDA kernel and H2D copy were simultaneously active.
  
  **Smallest fix:** Save a CUDA activity trace from Nsight Systems/CUPTI or PyTorch profiler and gate on intersection between at least one actual pre-layer CUDA kernel activity interval and the H2D memcpy activity interval. Keep the current event metric as a supporting bound, not the definitive overlap gate.

- **Finding: P1 — Pending real-GPU protocol gates can pass with corrupted allocator or refusal state.**  
  The injected-failure and final-state gates in `reproduction/scripts/real_executor_pilot.py:140-150` check `num_blocks` but omit exact `num_free_blocks`, `is_block_free`, cache-list lengths, and restored `kv_cache_new_block_size`. The ownership refusal gate checks rows/counts/sentinel/queues but does not assert the captured active/group layer state (`reproduction/scripts/multirequest_ownership.py:72-79,101`), and `ordinary_preemptions_zero` is hard-coded true (`multirequest_ownership.py:105`) rather than derived from scheduler records.
  
  **Smallest fix:** Snapshot and compare all allocator fields before/after failure, require expected executor/coordinator/group layers after refusal, and either measure a real scheduler preemption counter or relabel that field as “not exercised.”

- **Finding: P2 — Rollback-copy failure itself is not fail-closed.**  
  The rollback loops call `copier.enqueue()` from inside exception handlers without a nested guard (`reproduction/runtime/morphserve/real_executor.py:69-82,186-196`). If a rollback copy also fails, the method escapes with partially modified Python state and no explicit poisoned state. Existing tests inject only the initiating copy failure, not a second rollback failure (`reproduction/tests/test_real_executor_transactions.py:36-61`).
  
  **Smallest fix:** Catch rollback failures separately, preserve detailed failure state, prevent further serving, and test double-failure behavior.

- **Merge verdict: BLOCK.** The original alignment, all-before-mutation shrink, partial-copy rollback, and cross-stream expansion barrier defects are substantially repaired. However, cross-operation shrink/restore atomicity and ignored rollback failures remain correctness blockers, and the proposed overlap gate still cannot establish actual kernel/copy concurrency.