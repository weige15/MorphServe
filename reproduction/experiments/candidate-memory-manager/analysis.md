# Candidate memory-manager result

## In-process mechanism evidence

The unmodified candidate C++ source compiled and both synthetic RTX 3090 checks completed their assertions:

- packed bytes copied into the exact registered GPU base address;
- a 3,072-byte reclaimed tail exposed exactly 12 K/V blocks of shape `[12,2,1,4,8]`;
- K began at owner byte 1,024, V immediately followed K, and V ended exactly at owner byte 4,096;
- writes through K/V changed the owner buffer;
- `torch.cuda.memory_allocated()` delta for the storage views was 0 bytes;
- a tail smaller than one full block raised `RuntimeError`;
- original bytes restored at the same base after reclaimed views were released.

Single-run mechanism timings (not paper-comparable): packed copy 1.6916 ms, acquire/zero 0.1154 ms, restore 0.1254 ms. These tiny synthetic transfers are correctness instrumentation only.

## Fatal negative result

After unittest printed `OK`, the process segfaulted at exit (139). `cuda-gdb` traced the crash to destruction of `pybind11::object` values held in the static `MemoryManager::layer_memory_tensor_map_org` container after Python finalization. See `../../doc/debug-report.md`.

Classification: **tested but not reproduced** for the released extension. In-process physical reclaim works in the scoped synthetic case, but normal process lifetime is unsafe. A separate reconstruction must replace persisted Python objects with native C++ metadata and rerun the same gate before any larger test.

## Evidence

- `results/metrics.json`
- `results/test.{stdout,stderr}.log`, `test.exitcode`
- `results/wall-time.json`
- `results/diagnosis/cuda-gdb.stdout.log`
- component/control probe logs under `results/diagnosis/`
- `results/manifest-{before,after}.log`

## Resource cost

One RTX 3090 was used for sub-second test kernels in several isolated diagnostic processes. No model weights were loaded. The primary test wall time was recorded in `results/wall-time.json`; GPU experiment time remains separate from agent-token usage.
