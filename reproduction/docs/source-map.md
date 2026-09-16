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
| Morphing Controller | Section 4.1; example thresholds KV >85%, queue delay >100 ms; Section 5 modes | Pressure actions embedded in `server/scheduler.py::get_next_batch` and `try_quant2org` | Partial and materially different from prose. No explicit default/performance/accuracy mode, no 85%/100 ms implementation, no smoothing/persistence configuration, and aggressive fixed-count morphing is hard-coded. |
| Per-worker Morphing Executor | Section 4.1; Sections 4.3–4.4 | `server/memory_utils.py`, C++ `MemoryManager`, model/layer wrappers | Mechanism exists but exact synchronization/fidelity is unverified. |
| Offline LIS profiling | Section 4.2; Appendix A; Algorithm 1 | No profiler in candidate repository. Reconstruction selection core: `runtime/morphserve/profiling.py`; tests: `tests/test_profiling.py` | Conditioned greedy selection, history, validation, and save format implemented/tested. Model metric collection and real profile remain missing. |
| FP16/W8/W4 pinned variants | Section 4.3; Appendix C | FP16 CPU and GPU loaders in `worker/weight.py`; AWQ INT4 pinned loader in `server/quantized_layer_manager.py` | Partial. Candidate loader supports one AWQ path and FP16; no W8 path was located. The loader expects a single `torch.load` checkpoint, unlike the available sharded safetensors AWQ asset. |
| Contiguous pinned CPU weights | Section 4.3; Appendix C | `load_weights()` and `AWQLayerManager::_organize_layers_in_continuous_memory` | Present in source; must be validated for byte layout, metadata, memlock, lifetime, and actual pinned allocation. |
| Preallocated in-place GPU region | Section 4.3; Appendix C | `initialize_memory_tracking`; C++ `register_layer_memory_org_gpu`; `replace_layer_*` copies into the FP16 address | Present conceptually. The C++ code also creates new tensor/storage views and Python reconstructs layer objects, so the paper's “without pointer remapping” wording is not literally reflected at the object level. |
| Real AWQ W4 execution | Section 4.3; Appendix C | `AWQTransformerLayer`, `awq.quantize.qmodule.WQLinear` | Source path is real packed AWQ execution, not fake quantization. It has not yet imported or executed in this workspace. |
| Asynchronous layer swapping | Section 4.3; Appendix C | C++ `replace_layer_org2quant` / `replace_layer_quant2org` call `cudaMemcpyAsync` on a newly created stream | Transfer API is asynchronous, but each function immediately calls `cudaStreamSynchronize`, destroys the stream, and returns. This blocks the caller and does not establish overlap with decode. Separate persistent morph/decode streams and lifetime events are absent in this path. |
| Kernel warmup/precompile | Section 4.3; Appendix C | `Engine.initialize` morphs/restores layers as “prewarming”; model profiling runs FP16 | Incomplete. No explicit dummy AWQ GEMM warmup covering all shapes/variants was found. |
| KV physical capacity expansion | Section 4.4; Appendix C | C++ `MemoryManager::acquire_new_kvcache`; `update_kvcache_after_replacing_layer`; `BlockManager` capacity extension; repaired copy in `runtime/candidate-csrc` | One synthetic RTX 3090 test proves bounded writable storage, zero allocator delta, failure on undersized tail, and same-base restore. Released source crashes at shutdown; repaired native metadata exits 0. Full-model capacity remains unverified. |
| Non-contiguous block mapping | Section 4.4; Appendix C | Triton kernels in `worker/kernels/kvcache_mgmt.py` and `paged_attn.py`; extended block IDs map into reclaimed layer regions | Two-tail synthetic store/decode/attention test passes an independent dense oracle with max error 0. The code assumes later regions are exactly one equal layer stride below the first; arbitrary fragmentation is not supported/verified. |
| Safe shrink/restore | Section 4.4 | `if_kv_cache_unused`; `quant2org_and_update_kvcache`; reconstructed `record_layer_memory_use` | Candidate occupancy check alone allowed a delayed write to corrupt 1,530 restored bytes. Reconstructed per-region events reduce corruption to 0 and pass three immediate protected cycles. Python kernel call sites must still record use; full integration is pending. |
| No flushing/re-prefill/KV eviction caused by morphing | Sections 4.3–4.4 | Model state and block tables are retained | Intended, but no executed state-preservation tests or same-precision-history oracle are included. Ordinary scheduler swapping remains possible and must be reported separately. |
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
| Traces | 72 s Azure and BurstGPT segments; 4.75× and 1.75× downscaling | Exact files, window offsets, operation defining “downscaling,” sampling seed, and request/context assignment unavailable. |
| Tasks | GovReport, QMSum, English-translated DuReader, Multi-News | Splits, translation artifact, prompts, preprocessing, metrics/packages/revisions, and sample IDs unavailable. |
| Controller modes | default, performance, accuracy | Thresholds, persistence, swap increments/limits, and recovery policy unavailable. Paper examples (85% KV, 100 ms queue) are not complete mode configurations. |
| Baselines | FP16, static AWQ INT4, LLM-PQ, PyramidKV, three MorphServe modes | LLM-PQ layer plan, PyramidKV configuration, common scheduler settings, and exact baseline source revisions unavailable. |

## Explicit inconsistencies to preserve

1. Appendix C says swapping occurs at the same addresses “without pointer remapping,” while the candidate C++ source creates new `Storage`/tensor views and Python reconstructs layer modules around them. Treat physical-address reuse and object/pointer remapping as separate claims.
2. Main Section 4.2 equations and Appendix A say higher cosine similarity means lower sensitivity and Algorithm 1 takes `argmax`. Appendix prose also calls LTS “angular distance” and elsewhere describes “weight sensitivity/norm-based changes.” Follow the explicit cosine equations and record prose ambiguity.
3. The paper says asynchronous overlapping; the candidate C++ path performs `cudaMemcpyAsync` followed immediately by `cudaStreamSynchronize`, so candidate-code execution cannot be credited with overlap until measured and, if necessary, repaired in a labeled reconstruction.
4. Restoring FP16 changes only future token computation. It cannot retroactively reproduce an all-FP16 trajectory for tokens generated while W4 layers were active.
