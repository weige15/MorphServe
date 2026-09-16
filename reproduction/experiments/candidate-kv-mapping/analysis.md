# Two-region Triton KV mapping result

## Result

The repaired extension plus byte-identical candidate Triton kernels passed the frozen two-region test and exited 0.

- Virtual block table `[0,2,14,15]` addressed original storage, reclaimed group 0, and two blocks in reclaimed group 1.
- Prefill tokens 0–11 were stored exactly across original/group-0/group-1 regions.
- Decode token 12 was stored exactly in group-1 block 1.
- Combined candidate PagedAttention read all 13 logical tokens and matched the independent dense PyTorch oracle with maximum absolute error `0.0` in this FP16 case.
- Packed prefixes were unchanged, two group pointers differed by exactly one 4,096-byte layer stride, and storage-view allocator delta was zero.
- Process exit and both vendor-manifest checks were 0/clean.

## Classification

**Approximate/modified-condition mechanism reproduction.** This establishes the candidate's fixed-stride, back-to-front two-tail mapping on a synthetic RTX 3090 layout. It does not establish arbitrary non-contiguous allocator regions: the implementation assumes equal contiguous FP16 layer strides and computes later region pointers by subtraction from group 0. GQA, multiple heads/layers, safe shrink, in-flight CUDA events, repeated oscillation, and full-model behavior remain unverified.

## Timing caveat

The first invocation printed 877 ms inside PagedAttention and the process wall time was 8.06 s because Triton compiled kernels on demand. The paper requires precompilation before timed serving. These values are initialization/compilation evidence only and are excluded from performance comparisons.

## Evidence

- `results/metrics.json`
- `results/test.{stdout,stderr}.log`, `test.exitcode`
- `results/wall-time.json`
- `results/manifest-{before,after}.log`
- `tests/test_candidate_kv_mapping.py`
