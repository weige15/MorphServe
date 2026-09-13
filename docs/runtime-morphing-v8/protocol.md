# Runtime morphing v8 validation protocol

Frozen before GPU validation. This experiment uses only `{FP16, AWQ-Marlin W4-16}` on the existing Llama-3.1-8B/base and AutoAWQ artifacts, torch-2.9/vLLM-0.11.2 environment, and RTX 3090. It does not implement a workload controller, tune v7 thresholds, change FCFS policy, change the quantization backend, or rerun the static frontier.

## Confirmatory checks

1. **Static controls:** fresh FP16 and static AWQ-W4-16 reproduce safe physical capacities near 1,768 and 4,286 blocks and produce finite deterministic output for two identical concurrent prompts.
2. **One-way state preservation:** two active requests decode at least four FP16 output steps, manually transition at a complete-forward boundary, then finish under AWQ without another prefill.
3. **Round-trip state preservation:** two active requests decode an FP16 prefix, continue under AWQ, restore to FP16 after at least twelve output steps, and finish without another prefill.
4. **KV expansion integrity:** sampled active logical K/V hashes and existing block-table entries are identical immediately before and after physical extension.
5. **Extension execution:** the real segmented store and paged-attention kernels read/write histories that cross the base/extension boundary.
6. **Safe shrink/remap:** live extension blocks are copied to free base IDs; logical hashes, request IDs, sequence-relative block slots, and allocation counts remain valid before extension release.
7. **Ownership/equivalence:** active runtime immutable tensors are byte-exact with the prepared output of the same validated static loader; selected inactive variants exist only in host memory.
8. **Repeated stability:** three complete FP16→AWQ→FP16 cycles in one process have consistent physical/scheduler capacity, allocator invariants, and no monotonic allocated/reserved HBM growth beyond 64 MiB after equal-state cleanup points.
9. **Cost:** report every transition and per-layer copy/synchronization/KV-resize component; summarize median/P95/min/max and compare total transition time with v7 high-pressure P95 benefit.

## Decision

GO requires real bidirectional one-process transitions, no re-prefill or active-state loss, meaningful physical KV growth, safe physical shrink, measured amortizable cost, and stable repeated cycles. Any logical-only capacity, lost KV/request state, unsafe peak, fixed capacity, or repeated corruption/leak is NO-GO.
