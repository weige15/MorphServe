# MorphServe source-to-implementation map

Status date: 2026-09-16 UTC. This map distinguishes the supplied conference-version PDF, verifiable public metadata, a candidate project-account code snapshot whose authorship is not yet independently linked, and reconstruction work. Reference values are isolated in `../configs/paper-reference-values.json`.

## Primary paper artifacts

| Artifact | Pinned evidence | Role |
|---|---|---|
| Supplied paper | `../../references/morphserve-2506.02006-v2.pdf`; SHA-256 `e2c0f12fcbc5188a04078a31a732acaa95e93e9662aff4b766a0b9d6a73fbe30` | Target conference version, 19 pages |
| arXiv v2 source | `../sources/morphserve-2506.02006v2-source.tar.gz`; SHA-256 `bd617e0f54c7a27aa8a46ab569fbe7f594c4067099967a864078b9dacdc4406c` | Exact equations, captions, tables, and vector figures; corroborating source only, not a replacement target |
| Extracted paper text | `../sources/morphserve-2506.02006-v2.txt` | Search aid; visually inspect the PDF/vector figures for ambiguous layout |
| Machine-readable references | `../configs/paper-reference-values.json` | Paper-reported values only; must remain separate from measured results |

## Public code/source audit

| Source | Pin / captured evidence | Classification and finding |
|---|---|---|
| `ds2-lab/MorphServe` | branch `main` at `1c42999ded0014717e48c947d13450848cc1b796`; API snapshots under `../sources/remote-audit/ds2-*.json` | Official lab repository. It contains only `LICENSE` and `README.md`, has no tags/releases/other public branches, and states that full code will be released later. It is not an implementation release. |
| `MorphServe/MorphServe` | branch `main` at `85c4fbf753b6eb47613bdd7ea423c38b48054544`; vendored exact archive under `../vendor/author-morphserve/`; API snapshots under `../sources/remote-audit/project-*.json` | **Candidate project-account artifact, authorship unresolved.** The account/repository predates the first arXiv posting and contains code closely matching the paper, but neither the supplied paper nor current `ds2-lab` README links it, and the GitHub account has no public author identity. Do not call this verified author code until linked by a primary author/lab source. |
| `interestingLSY/swiftLLM` | `682cf9a28f97f7490409981a2f181528f377eb5d`; `../vendor/swiftllm-upstream/` | Public reconstruction baseline and comparison source. This commit postdates the initial MorphServe code addition; the candidate MorphServe repository does not record its SwiftLLM base commit. |
| Prior local research | `/nfs/home/s314511048/precision-batching` at `9ae618b...` plus uncommitted result files | Reusable local evidence only. It includes a corrected SwiftLLM FP16 path, local model assets, and KV experiments, but its fake weight-quantization path is not MorphServe or low-bit execution. Nothing from it is treated as a paper result. |

Vendored trees have `MANIFEST.sha256` files. Runtime fixes must go in a separate reconstruction tree and never overwrite the candidate artifact.

## Paper mechanism to code map

| Paper requirement | Strongest explicit paper evidence | Candidate artifact mapping | Current fidelity assessment |
|---|---|---|---|
| Serving Monitor | Section 4.1; Fig. 3; queue, throughput, KV usage, TTFT/TPOT signals | `MorphServe/server/engine.py::_metrics_event_loop`; `ServerMetrics` in `server/structs.py` | Partial. Samples queue/running/swapped counts, KV utilization, and approximate throughput. No demonstrated TTFT/TPOT feedback controller, smoothing, persistence window, or mode-specific policy. |
| Morphing Controller | Section 4.1; example thresholds KV >85%, queue delay >100 ms; Section 5 modes | Reconstruction `controller.py`, `integration.py`, `real_executor.py` | Real accuracy-mode pilot morphs `[25,24,26]`, grows 4→1849 blocks, rolls back injected failure, recovers LIFO with exact FP16/FCFS. Thresholds reconstructed; no active workload/concurrency. |
| Per-worker Morphing Executor | Section 4.1; Sections 4.3–4.4 | `server/memory_utils.py`, C++ `MemoryManager`, model/layer wrappers | Mechanism exists but exact synchronization/fidelity is unverified. |
| Offline LIS profiling | Section 4.2; Appendix A; Algorithm 1 | Reconstruction `runtime/morphserve/profiling.py`; real collector `scripts/lis_real_pilot.py`; frozen profiles under `profiles/` | Eight-layer real WikiText-2/2048 run retains metrics, executes 36 conditioned calls, saves `[25,24,26,27,28,29,30,31]`, exact restore. Full 32-layer paper profile remains missing. |
| FP16/W8/W4 pinned variants | Section 4.3; Appendix C | FP16 CPU and GPU loaders in `worker/weight.py`; AWQ INT4 pinned loader in `server/quantized_layer_manager.py` | Partial. Candidate loader supports one AWQ path and FP16; no W8 path was located. The loader expects a single `torch.load` checkpoint, unlike the available sharded safetensors AWQ asset. |
| Contiguous pinned CPU weights | Section 4.3; Appendix C | `load_weights()` and `AWQLayerManager::_organize_layers_in_continuous_memory` | Present in source; must be validated for byte layout, metadata, memlock, lifetime, and actual pinned allocation. |
| Preallocated in-place GPU region | Section 4.3; Appendix C | `initialize_memory_tracking`; C++ `register_layer_memory_org_gpu`; `replace_layer_*` copies into the FP16 address | Present conceptually. The C++ code also creates new tensor/storage views and Python reconstructs layer objects, so the paper's “without pointer remapping” wording is not literally reflected at the object level. |
| Real AWQ W4 execution | Section 4.3; Appendix C | Candidate targets mismatched llm-awq; independent adapter `runtime/morphserve/autoawq_adapter.py` uses local AutoAWQ `WQLinear_GEMM` | One real layer switches in-place: 23/23 packed tensors exact, 7 low-bit modules, zero allocator delta, 74.02% tail, W4 repeat within fixed envelope, exact FP16 restore. Independent modified condition; not candidate author loader. |
| Asynchronous layer swapping | Section 4.3; Appendix C | Candidate C++ `replace_layer_org2quant` / `replace_layer_quant2org`; reconstruction `AsyncLayerCopier` | Candidate immediately synchronizes/destroys each stream. Independent reconstruction now uses a persistent morph stream, pinned non-blocking copies, last-forward and just-in-time layer events; small-region ordering passes, while full-layer decode-overlap timing awaits GPU headroom. |
| Kernel warmup/precompile | Section 4.3; Appendix C | `Engine.initialize` morphs/restores layers as “prewarming”; model profiling runs FP16 | Incomplete. No explicit dummy AWQ GEMM warmup covering all shapes/variants was found. |
| KV physical capacity expansion | Section 4.4; Appendix C | C++ `MemoryManager::acquire_new_kvcache`; `update_kvcache_after_replacing_layer`; `BlockManager` capacity extension; repaired copy in `runtime/candidate-csrc` | One synthetic RTX 3090 test proves bounded writable storage, zero allocator delta, failure on undersized tail, and same-base restore. Released source crashes at shutdown; repaired native metadata exits 0. Full-model capacity remains unverified. |
| Non-contiguous block mapping | Section 4.4; Appendix C | Vendor fused path fixed-stride; reconstruction explicit-region store + repaired multi-kernel attention | Vendor `[25,24,26]` corrupts layer 23. Explicit fallback writes registered layer 26 and both arbitrary/descending dense-attention oracles have max error 0. Added launches unbenchmarked; real integration pending. |
| Safe shrink/restore | Section 4.4 | Reconstructed event barrier + `RealMorphingExecutor.shrink_kv_before_restore` | Delayed-writer race repaired; two real request IDs occupy reclaimed IDs 4/5, recovery refuses group 0 while preserving rows/counts/sentinel/FCFS, then succeeds after BlockManager frees. Concurrent decode timing still pending. |
| No flushing/re-prefill/KV eviction caused by morphing | Sections 4.3–4.4 | Model state/block tables; independent active-KV pilot `experiments/active-kv-switch` | One request continues FP16→W4×4→FP16 with a same-history oracle, reclaimed block use, byte-exact migration, and no flush/eviction. One-layer/one-block modified condition; ordinary scheduler preemption remains separate. |
| Scheduler/attention compatibility | Sections 4.1, 4.4; Appendix C | Forked SwiftLLM scheduler plus custom Triton attention/store kernels | Material changes are larger than “minimal” by diff against current SwiftLLM; base commit is unknown. Correctness must be tested rather than inferred. |
| No-morph FP16 numerical base | Required verification surface | Normalized candidate `runtime/candidate-python`; `experiments/candidate-fp16-baseline/results-attempt-3/metrics.json` | Modified-condition Llama 3.1 8B: top-1/top-5 match Transformers, relative logit L2 0.00204, 291/291 weights exact. KV/scheduler path not covered. |
| Metrics and raw request records | Section 5 setup and objective verification surface | `RequestMetrics`, `ServerMetrics`, utilities | Partial. No supplied replay scripts/configs/raw paper logs; scheduled arrival and complete token timing schema are absent. |

## Offline profiling contract

The reconstruction must implement the explicit equations and Algorithm 1 exactly before exploring alternatives:

- `LTS[p] = cos(layer_input[p], full_precision_layer_output[p])`
- `LRS[p] = cos(full_precision_layer_output[p], quantized_layer_output[p])`
- `MDS[p|Q] = cos(model_output_with_Q, model_output_with_Q_union_p)`
- `LIS[p|Q] = 0.25*LTS[p] + 0.25*LRS[p] + 0.5*MDS[p|Q]`
- At every greedy step, evaluate MDS for **every** remaining candidate conditioned on current `Q`, select `argmax`, append to `Q`, and retain the final order offline.

Paper-specified settings: WikiText-2; calibration sequence length 2,048; cosine similarity; higher similarity means lower sensitivity; Front-to-Back is the default for unprofiled/new models; claimed 32-layer profiling time is under 15 minutes on one unspecified GPU. Calibration sample count/split/seed, representation reduction axes, exact “model output,” tie break, tokenizer/checkpoint revisions, and random-baseline repetitions are unresolved.

## Experiment-input map and unresolved exact settings

| Item | Paper-specified | Unresolved / unavailable |
|---|---|---|
| Models | Vicuna 7B v1.5, Llama 2 7B, Llama 3 8B, CodeLlama 34B | Exact repository/revision and base/chat/instruct status for all but named Vicuna version; only local Llama 3.1 8B base and derived AWQ assets are available. |
| Hardware | L4 24 GB + 256 GB RAM for 7B/8B; A100 80 GB + 2 TB for 34B | Local host is 7× RTX 3090 24 GB with 125 GiB RAM, not an exact hardware match. |
| Lengths | MHA 512/256; GQA 1024/512 | Prompt construction, truncation side, EOS, forced output length, and decoding parameters unresolved. |
| Traces | 72 s Azure and BurstGPT segments; 4.75× and 1.75× downscaling | Primary Azure Code/Conversation and pre-paper BurstGPT v1.1 files recovered/hashes audited in `docs/trace-audit.md`; exact file/window/scaling/context mapping remain unavailable. |
| Tasks | GovReport, QMSum, English-translated DuReader, Multi-News | Splits, translation artifact, prompts, preprocessing, metrics/packages/revisions, and sample IDs unavailable. |
| Controller modes | default, performance, accuracy | Thresholds, persistence, swap increments/limits, and recovery policy unavailable. Paper examples (85% KV, 100 ms queue) are not complete mode configurations. |
| Baselines | FP16, static AWQ INT4, LLM-PQ, PyramidKV, three MorphServe modes | LLM-PQ layer plan, PyramidKV configuration, common scheduler settings, and exact baseline source revisions unavailable. |

## Explicit inconsistencies to preserve

1. Appendix C says swapping occurs at the same addresses “without pointer remapping,” while the candidate C++ source creates new `Storage`/tensor views and Python reconstructs layer modules around them. Treat physical-address reuse and object/pointer remapping as separate claims.
2. Main Section 4.2 equations and Appendix A say higher cosine similarity means lower sensitivity and Algorithm 1 takes `argmax`. Appendix prose also calls LTS “angular distance” and elsewhere describes “weight sensitivity/norm-based changes.” Follow the explicit cosine equations and record prose ambiguity.
3. The paper says asynchronous overlapping; the candidate C++ path performs `cudaMemcpyAsync` followed immediately by `cudaStreamSynchronize`, so candidate-code execution cannot be credited with overlap until measured and, if necessary, repaired in a labeled reconstruction.
4. Restoring FP16 changes only future token computation. It cannot retroactively reproduce an all-FP16 trajectory for tokens generated while W4 layers were active.
