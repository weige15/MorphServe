# Explicit-region KV remapping repair result

The arbitrary-order counterexample is repaired in the independent runtime fallback:

- profile order `[25,24,26]` writes exact K/V into registered layer 26;
- unintended layer-23 tail remains unchanged;
- explicit-region PagedAttention matches the dense one-token oracle with max absolute error 0;
- the existing descending two-tail combined fast path still passes its dense oracle with max error 0;
- both processes exit 0.

## Classification

**Approximate/modified-condition correctness repair.** Arbitrary region order is now supported by multi-kernel dispatch and local block-table remapping. The candidate vendor fused path remains broken and immutable. This fallback adds kernel launches and has only JIT-contaminated timing; it is not performance-equivalent to the paper's unspecified custom mapping implementation.

Real executor integration should select the combined fast path only after validating descending equal-stride addresses, otherwise use explicit dispatch.
