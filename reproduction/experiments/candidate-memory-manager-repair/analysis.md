# Native metadata repair result

## Result

The diagnosed repair succeeded without changing candidate address/capacity logic.

- Both GPU tests passed and the process exited 0.
- Torch-preloaded extension import exited 0 and exposed the same 13 bindings.
- Packed and owner base addresses were identical.
- The 3,072-byte tail again exposed 12 K/V blocks of shape `[12,2,1,4,8]`.
- K and V remained adjacent and ended exactly at the owner boundary.
- Writes affected owner storage, restore recovered original bytes, undersized allocation failed, and allocator delta remained 0.
- Immutable candidate vendor manifests passed before and after.

Modified files are isolated under `runtime/candidate-csrc`. The only semantic change is immediate conversion of registration dictionaries into native C++ metadata; the Python binding input shape is unchanged.

## Modified-condition classification

**Approximate/modified-condition mechanism reproduction.** The repaired reconstruction reproduces one synthetic physical-reclaim/copy/restore mechanism on RTX 3090. It does not establish candidate author identity, non-contiguous multi-layer Triton mapping, safe occupied/in-flight shrink, AWQ numerical execution, asynchronous overlap, or paper performance.

## Evidence

- `results/test.{stdout,stderr}.log`, `test.exitcode` (`0`)
- `results/metrics.json`
- `results/import-after-torch.{log,exitcode}`
- `results/manifest-{before,after}.log`
- source diff: `runtime/candidate-csrc` vs `vendor/author-morphserve/csrc`

## Timings

One verification run recorded 1.4909 ms packed copy, 0.1323 ms acquire/zero, and 0.1414 ms restore for tiny synthetic buffers; test-process wall time was 2.9414 s including Python/CUDA startup. These are not paper-comparable latency measurements.
