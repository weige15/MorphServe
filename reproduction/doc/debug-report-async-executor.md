# Debug report: asynchronous executor transactional and timing defects

Date: 2026-09-17  
Status: transactional/alignment repairs verified at unit/CUDA seams; full-model rerun pending  
Scope: independent reconstruction only; immutable vendor is unchanged

## Symptoms and evidence

1. Full-model attempt 1 exited 1: restored bytes/state passed, but two separate same-history FP16 request logits were not bit-exact (relative L2 0.00074–0.00130, top-1 exact). The first W4 decode took 3.67 s because PagedAttention decode JIT was not warmed. `experiments/async-layer-transfer/full-model-results-attempt-1/metrics.json`.
2. The overlap gate used enclosing interval intersection, which can count time spent waiting at the affected layer rather than actual concurrent pre-layer kernels.
3. Read-only review found partial mutation paths in `RealMorphingExecutor`: expansion rollback can race queued KV zeroing; multi-layer shrink can remove a free newest group before discovering an occupied older group; morph/restore can partially mutate on later-layer failure.
4. `acquire_new_kvcache` can unsigned-underflow if 256-byte alignment pushes the reclaimed start beyond the registered region.

## Root causes

- `acquire_new_kvcache()` enqueues `zero_()` on the current stream. Failure rollback restores host metadata immediately, while FP16 restoration is queued on a different stream without an event dependency.
- `shrink_kv_before_restore()` validates and mutates one group at a time rather than validating the complete batch first.
- Multi-layer morph/restore has no committed-layer rollback boundary; coordinator rollback assumes every selected layer activated.
- Timing records decode start/end only; no event marks completion of actual layers 0..N-1 before the transfer wait.
- Warmup runs `ignore_kvcache=True`, so it never compiles decode PagedAttention.
- The FP16 comparison confuses weight-byte restoration with bit-exact equality between separate request executions; the latter is stricter than prior measured numerical-repeat behavior.
- Reclaimed-size subtraction uses unsigned `size_t` before validating aligned offset.

## Repair plan and verification

1. Prevalidate every multi-layer shrink range, then commit removals only if all are free.
2. On expansion failure, record a current-stream event after queued zeroing and make subsequent copier work wait through `last_forward_event`.
3. Make multi-layer morph rollback committed layers internally; coordinator restores only layers actually active. Validate restore batches before mutation.
4. Reject aligned offsets outside the region before unsigned subtraction.
5. Add a model event immediately before the layer-local transfer wait; report intersection of copy with `[decode_start, pre_wait]` as actual pre-layer overlap.
6. Warm one real cached decode before timed work.
7. Keep byte-exact restoration as the FP16 storage gate; use the existing `<0.005` same-history numerical envelope plus top-1 match for separate request logits.
8. Add targeted tests for occupied-second-group shrink, second-layer morph failure, second expansion failure/event ordering, and alignment boundary. Re-run full executor/ownership/full overlap only after these pass.

## Repair evidence

- 5 CUDA async tests: copy/order/use barrier, malformed source rejection, second-expansion rollback event, and no redundant event accumulation.
- 9 transaction/controller tests: partial morph/restore rollback, all-before-mutation shrink, restore prevalidation and existing FCFS/action contracts.
- 3 rebuilt C++ memory-manager tests: normal physical reclaim/restore, tiny-tail rejection, and misaligned-region rejection.
- Retry script now warms a real cached decode and records a pre-layer-wait event; it has not yet run because the free-memory preflight requires 17 GiB.

## Boundaries

No paper overlap conclusion is valid from attempt 1. Its three W4 copies (15.174–15.475 ms) and FP16 copies (57.769–58.293 ms) are valid local modified-condition transfer diagnostics; the first timed decode is JIT-contaminated and the original overlap definition is insufficient.
