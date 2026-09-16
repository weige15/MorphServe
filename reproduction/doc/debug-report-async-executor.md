# Debug report: asynchronous executor transactional and timing defects

Date: 2026-09-17  
Status: transactional/alignment repairs verified at unit/CUDA seams; full-model rerun pending  
Scope: independent reconstruction only; immutable vendor is unchanged

## Symptoms and evidence

1. Full-model attempt 1 exited 1: restored bytes/state passed, but two separate same-history FP16 request logits were not bit-exact (relative L2 0.00074–0.00130, top-1 exact). The first W4 decode took 3.67 s because PagedAttention decode JIT was not warmed. `experiments/async-layer-transfer/full-model-results-attempt-1/metrics.json`.
2. The overlap gate used enclosing interval intersection, which can count time spent waiting at the affected layer rather than actual concurrent pre-layer kernels.
3. Read-only review found partial mutation paths in `RealMorphingExecutor`: expansion rollback can race queued KV zeroing; multi-layer shrink can remove a free newest group before discovering an occupied older group; morph/restore can partially mutate on later-layer failure.
4. `acquire_new_kvcache` can unsigned-underflow if 256-byte alignment pushes the reclaimed start beyond the registered region.
5. Follow-up review found two remaining cross-operation failures: successful KV shrink followed by failed FP16 restore was not compensated, and failed pressure rollback ignored a false/raised restore. It also correctly rejected `[decode_start, pre_wait]` as definitive kernel/copy overlap evidence. `results/raw/transaction-followup-review.md`.

## Root causes

- `acquire_new_kvcache()` enqueues `zero_()` on the current stream. Failure rollback restores host metadata immediately, while FP16 restoration is queued on a different stream without an event dependency.
- `shrink_kv_before_restore()` validates and mutates one group at a time rather than validating the complete batch first.
- Multi-layer morph/restore has no committed-layer rollback boundary; coordinator rollback assumes every selected layer activated.
- Timing records decode start/end only; no event marks completion of actual layers 0..N-1 before the transfer wait.
- Warmup runs `ignore_kvcache=True`, so it never compiles decode PagedAttention.
- The FP16 comparison confuses weight-byte restoration with bit-exact equality between separate request executions; the latter is stricter than prior measured numerical-repeat behavior.
- Reclaimed-size subtraction uses unsigned `size_t` before validating aligned offset.
- Coordinator recovery previously composed two destructive calls without an atomic boundary, and rollback return values were ignored.
- CUDA events bound an enclosing interval but cannot distinguish kernel execution from CPU launch preparation or device idle gaps.

## Repair plan and verification

1. Prevalidate every multi-layer shrink range, then commit removals only if all are free.
2. On expansion failure, record a current-stream event after queued zeroing and make subsequent copier work wait through `last_forward_event`.
3. Make multi-layer morph rollback committed layers internally; coordinator restores only layers actually active. Validate restore batches before mutation.
4. Reject aligned offsets outside the region before unsigned subtraction.
5. Add a model event immediately before the layer-local transfer wait, but retain its intersection with `[decode_start, pre_wait]` only as a supporting upper bound.
6. Add a separate untimed `torch.profiler` diagnostic cycle and gate overlap only when a size-matched pinned H2D CUDA activity intersects an actual kernel activity on another stream.
7. Warm one real cached decode before timed work.
8. Keep byte-exact restoration as the FP16 storage gate; use the existing `<0.005` same-history numerical envelope plus top-1 match for separate request logits.
9. Add atomic `recover_fp16`: snapshot all KV allocator/cache/group metadata, shrink, restore, and reattach the exact snapshot on restore failure. Check every pressure rollback result, synchronize coordinator/executor active sets, and poison both serving paths when compensation cannot be proven.
10. Add targeted tests for occupied-second-group shrink, second-layer morph failure, second expansion failure/event ordering, rollback-copy double failure, post-shrink restore false/exception, exact allocator compensation, and alignment boundaries. Re-run full executor/ownership/full overlap only after these pass.

## Repair evidence

- 5 CUDA async tests: copy/order/use barrier, malformed source rejection, second-expansion rollback event, and no redundant event accumulation.
- 14 transaction/controller tests: partial morph/restore rollback, exact post-shrink compensation, ignored-rollback false/exception poisoning, rollback-copy double failure, all-before-mutation shrink, restore prevalidation and FCFS/action contracts.
- 3 fail-closed CUDA-trace analyzer tests: require size-matched H2D, another stream, and actual kernel/copy activity intersection; reject enclosing intervals, wrong sizes and same-stream activity.
- 3 rebuilt C++ memory-manager tests: normal physical reclaim/restore, tiny-tail rejection, and misaligned-region rejection.
- Retry script now warms a real cached decode, records the supporting pre-layer-wait event, and exports a separate CUDA activity trace; it has not yet run because the free-memory preflight requires 17 GiB.

## Boundaries

No paper overlap conclusion is valid from attempt 1. Its three W4 copies (15.174–15.475 ms) and FP16 copies (57.769–58.293 ms) are valid local modified-condition transfer diagnostics; the first timed decode is JIT-contaminated and the original overlap definition is insufficient.
