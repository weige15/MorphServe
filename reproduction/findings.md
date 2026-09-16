# Evolving findings

## Current understanding

The supplied MorphServe paper is much more specific about mechanism shape than about reproducible experiment identity. Its strongest implementable contract is: real packed weight-only layer variants, pinned host storage, fixed GPU layer regions, conditioned offline LIS ordering, custom non-contiguous KV address mapping, and state-preserving runtime transitions. Exact controller modes, trace windows, task construction, and several baseline settings are not disclosed.

The official lab repository does not release implementation code. A separate project-account repository contains plausible MorphServe source that materially advances the reproduction: it has real AWQ modules, C++ copies into pre-registered layer regions, reclaimed-memory KV storage views, dynamic block registration, and Triton KV store/attention mapping. However, it is neither attributable to the authors from a primary link nor runnable as published. It is best treated as a candidate artifact whose mechanism can be tested, not as verified author code.

## Patterns and insights

1. **Paper mechanism and candidate code align structurally but diverge operationally.** The code implements the unusual reclaimed-layer-tail KV mapping described by the paper, which is unlikely to be generic SwiftLLM boilerplate. Yet its `cudaMemcpyAsync` path immediately synchronizes, the controller is capacity-triggered rather than mode-based, and no LIS profiler is present.
2. **Mechanism correctness already exposed a release-blocking lifetime defect.** One synthetic reclaimed region produced correct bounded K/V views, zero allocator-byte increase, writable physical storage, and same-base restoration. The process then segfaulted at exit because static C++ maps retained `py::object` metadata past Python finalization. Exact Figure 4/Table 2 reproduction remains blocked; correctness repair precedes any performance run.
3. **Existing local work prevents redundant baseline discovery.** The prior `precision-batching` repository already pinned SwiftLLM, captured the same host, built its extension, located Llama assets, and found Llama 3 FP16 parity bugs. Those findings should inform the reconstruction, while its fake weight-quantization path must not be confused with MorphServe.
4. **The supplied paper contains internal reference errors.** It has only Tables 1–8; user/objective labels for Tables 4/9 conflict with the rendered target PDF. The exact numeric `[TTFT, TPOT, F1]` target belongs to PDF Table 2. Reference values must follow the actual PDF and preserve this mismatch.

## Lessons and constraints

- Never call `ds2-lab/MorphServe` an implementation release.
- Never call `MorphServe/MorphServe` verified author code without a primary identity/link.
- Never credit `cudaMemcpyAsync` as overlap when the caller immediately synchronizes.
- Never retain `py::object` in process-static C++ containers; parse registration dictionaries into native metadata before storing them.
- Keep physical GPU address reuse distinct from Python/C++ tensor-pointer or object reconstruction.
- The LIS equations are cosine similarities with `argmax`; appendix prose saying “angular distance,” “weight sensitivity,” or input independence is inconsistent and must not override the equations.
- FP16 restoration affects future tokens only; historical W4 tokens remain part of the trajectory.
- Local RTX 3090 results are modified-condition evidence, not exact L4/A100 reproduction.

## Open questions

- Does the candidate C++ extension build against the available PyTorch/CUDA toolchain without source repair?
- Can the candidate package be normalized with only naming/config/checkpoint-loader changes, or are deeper correctness fixes required?
- Does reclaimed KV memory stay within registered layer bounds for multiple non-contiguous swapped layers?
- Do store and attention kernels preserve content under expand/shrink/restore and repeated prefill/decode adaptation?
- Can a real AWQ W4 layer execute numerically in the fixed FP16 region on RTX 3090?
- What exact representation and reduction should be frozen for LIS when the paper gives only vector-level cosine notation?
