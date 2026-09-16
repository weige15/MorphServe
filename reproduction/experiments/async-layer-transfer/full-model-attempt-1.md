# Full-model attempt 1 — useful overlap evidence, failed FP16 equality gate

The first full Llama 3.1 8B run executed three W4 and three FP16 transitions. Host enqueue remained nonblocking, final FP16 bytes/state were exact, all W4 same-history comparisons stayed within the predeclared `<0.005` envelope with matching top-1, and transfer/decode intervals overlapped.

The run exited 1 because the gate incorrectly required two separately decoded same-history FP16 requests to have bit-exact logits. They matched top-1 with relative L2 0.00074–0.00130, inside the independently measured low-level nondeterminism envelope. Byte-exact restored weights still passed.

The first W4 decode was also contaminated by decode PagedAttention JIT (3.67 s), because warmup compiled W4 GEMM but used `ignore_kvcache=True`. It is excluded from steady-state timing. Finally, interval intersection alone does not prove kernels ran concurrently before the layer wait; the retry must record a pre-wait CUDA event.

Before retry, severity-P1 review findings require transactional partial-expansion/multi-layer shrink/morph repairs and an alignment guard. This attempt is retained as raw diagnostic evidence, not a valid paper latency result.
