# Protocol: explicit-region KV remapping repair

Status: pre-registered after the `[25,24,26]` counterexample and before repair.

## Hypothesis

Dispatching candidate Triton store/attention kernels once per explicitly registered region with a locally remapped block table will support arbitrary profiled layer order and prevent fixed-stride writes into unrelated layers.

## Scoped reconstruction

In `runtime/candidate-python` only:

- add negative-block guards to original prefill/decode KV-store kernels;
- add `store_kvcache_explicit_regions` that masks/remaps original and each reclaimed group independently;
- repair `paged_attention_multi_kernels` so its helper actually uses the passed cache tensors/block table rather than closed-over originals;
- retain the fixed-stride combined functions unchanged as an optional validated fast path for descending contiguous layouts.

## Gate

1. Re-run `[25,24,26]`: layer 26 receives exact K/V; layer 23 remains unchanged.
2. Explicit-region PagedAttention matches an independent dense oracle.
3. Existing descending two-tail combined fast-path oracle remains max-error 0.
4. Original-cache-only path remains correct.
5. Process exits 0 and vendor manifests remain unchanged.

## Boundary

Python multi-kernel dispatch may cost extra launches and is not a performance-equivalent reproduction of the paper. It is a correctness repair and a valid fallback until an explicit pointer-table fused kernel is implemented/measured.
