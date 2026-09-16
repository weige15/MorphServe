# Reclaimed-region CUDA lifetime barrier result

## Result

The new event primitive prevents the previously confirmed race without hiding it:

- Unrecorded delayed writer: restore returned while work was unfinished and later left 1,530 corrupt bytes (sensitivity control).
- Explicit host-wait control: 0 corrupt bytes.
- `record_layer_memory_use` control: writer was unfinished before restore, restore waited about 103.4 ms, and final corruption was 0 bytes.
- Three event-protected immediate restore cycles had stable base addresses, exact restoration, and zero allocator deltas.
- Five earlier host-synchronized cycles also remained exact.
- Process exit was 0.

Regression gates passed with the repaired extension:

- 2/2 memory-manager tests;
- 1/1 two-region candidate Triton store/attention test, still exact against the dense oracle.

## Classification

**Approximate/modified-condition event-safety reproduction.** The reconstruction now has a valid explicit per-region lifetime barrier. It only works when every stream touching the region calls `record_layer_memory_use` after its final queued operation. Candidate Python call sites do not yet do this, so full-model safety remains unverified.

The copy itself still synchronizes and therefore does not demonstrate asynchronous overlap or negligible exposed stall.

## Evidence

- `results/metrics.json`
- `results/test.{stdout,stderr}.log`, `test.exitcode`
- `results/regression-memory*`
- `results/regression-kv*`
- C++ source under `runtime/candidate-csrc`
