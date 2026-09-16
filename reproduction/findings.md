# Evolving findings

## Current understanding

The supplied MorphServe paper is much more specific about mechanism shape than about reproducible experiment identity. Its strongest implementable contract is: real packed weight-only layer variants, pinned host storage, fixed GPU layer regions, conditioned offline LIS ordering, custom non-contiguous KV address mapping, and state-preserving runtime transitions. Exact controller modes, trace windows, task construction, and several baseline settings are not disclosed.

The official lab repository does not release implementation code. A separate project-account repository contains plausible MorphServe source that materially advances the reproduction: it has real AWQ modules, C++ copies into pre-registered layer regions, reclaimed-memory KV storage views, dynamic block registration, and Triton KV store/attention mapping. However, it is neither attributable to the authors from a primary link nor runnable as published. It is best treated as a candidate artifact whose mechanism can be tested, not as verified author code.

## Patterns and insights

1. **Paper mechanism and candidate code align structurally but diverge operationally.** The code implements the unusual reclaimed-layer-tail KV mapping described by the paper, which is unlikely to be generic SwiftLLM boilerplate. Yet its `cudaMemcpyAsync` path immediately synchronizes, the controller is capacity-triggered rather than mode-based, and no LIS profiler is present.
2. **The released physical-reclaim mechanism works only within a narrower layout than the paper's broad wording suggests.** After fixing its shutdown lifetime defect, candidate Triton kernels stored prefill/decode K/V across original plus two reclaimed tails and matched a dense attention oracle exactly. The mapping is not arbitrary fragmentation: it derives later pointers by subtracting equal contiguous FP16 layer strides from the first reclaimed pointer.
3. **The normalized candidate now has a trustworthy modified-condition FP16 base.** On local Llama 3.1 8B, all 291 loaded tensors were exact and next-token logits matched Transformers top-1/top-5 with relative L2 0.00204. Existing local work correctly predicted the Llama 3 RoPE/loader fixes needed. KV/scheduler parity is still separate, and fake quantization from prior work is not MorphServe evidence.
4. **An independent real-W4 adapter closes the numerical mechanism gap, not the provenance gap.** Layer 31 ran seven AutoAWQ packed modules from the original FP16 base, reclaimed 322.9 MB (74.02% after metadata), changed logits, then restored byte/logit exact with no allocator increase. The adapter remains a reconstruction because candidate code expects an incompatible llm-awq API/format.
5. **The supplied paper contains internal reference errors.** It has only Tables 1–8; user/objective labels for Tables 4/9 conflict with the rendered target PDF. The exact numeric `[TTFT, TPOT, F1]` target belongs to PDF Table 2. Reference values must follow the actual PDF and preserve this mismatch.

## Lessons and constraints

- Never call `ds2-lab/MorphServe` an implementation release.
- Never call `MorphServe/MorphServe` verified author code without a primary identity/link.
- Never credit `cudaMemcpyAsync` as overlap when the caller immediately synchronizes.
- Never retain `py::object` in process-static C++ containers; parse registration dictionaries into native metadata before storing them.
- Never restore an FP16 layer region based only on free-block counters. The reconstructed `record_layer_memory_use` barrier works, but every stream/call site must record use before restoration is safe.
- Keep physical GPU address reuse distinct from Python/C++ tensor-pointer or object reconstruction.
- The LIS equations are cosine similarities with `argmax`; appendix prose saying “angular distance,” “weight sensitivity,” or input independence is inconsistent and must not override the equations.
- FP16 restoration affects future tokens only; historical W4 tokens remain part of the trajectory.
- Local RTX 3090 results are modified-condition evidence, not exact L4/A100 reproduction.

6. **Real conditioned profiling is feasible and auditable, but suffix-only orders are incomplete.** The frozen 2,048-token run expanded to layers 24–31, executed all 36 sets in 58.14 s, and selected `[25,24,26,27,28,29,30,31]`; common tail order stayed `[29,30,31]`. Paper-style all-32 simultaneous pinning exceeds local memlock by 741 MB before overhead.

7. **Real LIS order exposed a vendor KV-address bug; correctness now requires a slower explicit fallback.** Vendor fused `[25,24,26]` maps group 2 into layer 23. Reconstructed per-region store/attention dispatch fixes the address and matches dense oracles, while descending equal-stride layouts may retain the fused fast path. Launch overhead remains unmeasured.

8. **Real ownership/recovery now covers multiple request IDs.** Accuracy-mode actions grow 4→1,849 blocks; request 0/1 occupy reclaimed IDs 4/5; LIFO recovery removes free groups but refuses occupied layer 25 without changing counts/rows/sentinel/FCFS, then succeeds after real frees with zero preemptions and exact FP16.

9. **Public trace identity is partially recoverable from Figure 1.** Figure 1a identifies Azure Code, and dense-window shape matching uniquely ranks Azure second 1073 and BurstGPT v1.1 second 1,781,278 (Pearson 0.8793/0.7356). These are frozen plot-derived candidates, not author-confirmed offsets; exact thinning/scaling and context mapping remain unknown.
10. **A minimal independent async seam is viable.** A persistent morph stream enqueued a 4 MiB pinned copy while prior use was unfinished, retained the GPU address, and became visible only through the layer-local wait. This repairs the candidate's host-blocking design but does not establish full-layer decode overlap until the frozen GPU pilot runs.
11. **The linked DuReader source does not contain the claimed English artifact.** The pinned 303-path tree has no English/translation file and no release. Exact translated-DuReader quality claims are an evidence blocker, not merely a local download problem.

## Open questions

- Does the candidate C++ extension build against the available PyTorch/CUDA toolchain without source repair?
- Can the candidate package be normalized with only naming/config/checkpoint-loader changes, or are deeper correctness fixes required?
- Does reclaimed KV memory stay within registered layer bounds for multiple non-contiguous swapped layers?
- Does the active-KV result survive multiple requests/layers, real block size 16, concurrent decode streams, and oscillating pressure without explicit occupied-block migration?
- Can a real AWQ W4 layer execute numerically in the fixed FP16 region on RTX 3090?
- What exact representation and reduction should be frozen for LIS when the paper gives only vector-level cosine notation?
