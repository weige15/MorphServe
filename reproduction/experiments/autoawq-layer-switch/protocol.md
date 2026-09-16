# Protocol: one-layer real AutoAWQ in-place switch and restore

Status: pre-registered before adapter implementation/execution.

## Question

Can the normalized candidate execute one real AutoAWQ W4 decoder layer copied into its existing FP16 GPU layer region, then restore exact FP16 bytes and logits, without whole-model reload or duplicate GPU model storage?

## Frozen configuration

- Local Llama 3.1 8B FP16 and AutoAWQ W4 G128 zero-point assets.
- Decoder layer 31 (candidate artifact's back-to-front first swap; this is not the paper's Front-to-Back fallback).
- Same nine-token prompt; KV cache disabled to isolate weights.
- AutoAWQ 0.2.9 `WQLinear_GEMM`; source tensors `qweight/qzeros/scales` loaded directly from sharded safetensors into one contiguous pinned CPU layer buffer.
- FP16 layer bytes backed up into one contiguous pinned CPU buffer using their real offsets.
- Repaired candidate C++ in-place copy/storage reconstruction.

## Gate

1. W4 attention and MLP modules are real `WQLinear_GEMM` with source-exact INT32/FP16 tensors after copy.
2. Quant tensors use the same GPU base region and remain within its original byte boundary; record packed bytes including norms/scales/zeros.
3. No second full GPU model or whole-model reload; process-local memory evidence is saved.
4. Two mixed forwards have stable top-1/top-5 and relative repeat logit L2 `<0.005`, the independently measured static split-K envelope; exact equality is not required.
5. Mixed logits are finite and differ from FP16 (the low-bit path actually affects execution).
6. Restore returns every FP16 layer byte exactly, reuses the original base, and restored logits are bit-exact to the initial candidate FP16 logits.
7. Process exits 0 and vendor manifests remain unchanged.

## Reference rule

Compare switching only against histories with the same precision schedule. This no-KV pilot has one FP16 forward, two W4 forwards, then one restored FP16 forward; it does not claim that restoration erases prior W4 token history.

## Boundaries

Modified-condition independent adapter, not execution of the candidate's mismatched llm-awq loader. No KV state, active requests, controller, overlap, or paper latency/quality claim. Timing is mechanism instrumentation only and excludes no JIT unless separately warmed.
