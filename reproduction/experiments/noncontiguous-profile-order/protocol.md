# Protocol: arbitrary profiled-order KV mapping counterexample

Status: pre-registered before execution.

## Question

Does the candidate's fused pointer arithmetic support a real profiled order that is not monotonically descending/contiguous, such as the measured LIS prefix `[25,24,26]`?

## Frozen synthetic layout

- Four contiguous 4,096-byte layer regions representing layers 23–26.
- Register/reclaim layers in profile order 25, 24, 26; each has 1,024 packed bytes and 12 KV blocks under the prior 2-layer/1-head/block-4/head-8 synthetic shape.
- Candidate combined Triton store receives only group-0 pointer (layer 25) and `org_layer_param_size`; group 2 therefore computes `layer25_base - 2*stride`, which points to layer 23, not registered layer 26.
- Store a known token to virtual group-2 block 0 and inspect both expected layer-26 tail and unintended layer-23 tail.

## Gate

This is a negative test. It passes if:

1. profile-order addresses are `[layer25, layer24, layer26]` and not representable by fixed negative strides;
2. candidate fused store leaves expected layer-26 K/V unchanged;
3. the known token appears in unintended layer-23 tail;
4. process exits 0 and vendor manifests remain unchanged.

If the candidate unexpectedly maps correctly, the hypothesis is refuted and no fix is justified.

## Boundary

Synthetic store-only counterexample. It does not test a repaired pointer-table/multi-kernel implementation or quantify production probability.
