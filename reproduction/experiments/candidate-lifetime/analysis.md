# Reclaimed-KV lifetime and repeated adaptation result

## Negative in-flight result

The candidate restore boundary does not wait for in-flight work using reclaimed storage.

A delayed write was queued on a separate non-default CUDA stream, then `replace_layer_quant2org` restored FP16 bytes on its own stream and returned while the writer event was still incomplete. After the writer finished, **1,530 bytes** differed from the original layer. The explicit-wait control had **0 corrupt bytes**.

This is a storage-lifetime counterexample, not a measured serving-race probability. The released controller is mostly synchronous and may accidentally avoid overlap; however, the paper and objective require morph/decode overlap. Any asynchronous reconstruction must attach events/lifetime ownership before restoring a region.

## Positive repeated-cycle result

Five fully synchronized quantize → reclaim → write → release → restore cycles passed:

- stable owner base in all cycles;
- exact original bytes after every restore;
- allocator delta `[0,0,0,0,0]` bytes;
- process exit 0.

Thus repeated state transitions work when the caller enforces completion and releases views; they are unsafe without that barrier.

## Classification

- Occupied/in-flight safe shrink: **tested-not-reproduced** for candidate boundary.
- Repeated synchronized expand/restore: **approximate/modified-condition reproduced** in a synthetic RTX 3090 test.

## Evidence

- `results/metrics.json`
- `results/test.{stdout,stderr}.log`, `test.exitcode`
- `results/wall-time.json`
- `tests/test_candidate_lifetime.py`
