# Public author/project artifact audit

Captured: see `../sources/remote-audit/CAPTURED_AT`. All GitHub API responses used here are preserved under `../sources/remote-audit/`.

## Finding 1 — official lab repository is still README-only

Rechecked 2026-09-17 (`results/raw/official-repo-recheck-2026-09-17.txt`): `https://github.com/ds2-lab/MorphServe` still has one public branch (`main`) at commit `1c42999ded0014717e48c947d13450848cc1b796`, no public tags, no releases, and a recursive tree containing only `LICENSE` and `README.md`. Its README states: “The full code will be released soon.” This repository cannot support an author-code execution.

The local project remote is `weige15/MorphServe`, not the lab repository. It initially contained only `README.md` and the supplied paper.

## Finding 2 — an earlier project-account repository contains a partial implementation

`https://github.com/MorphServe/MorphServe` has one public branch (`main`) at commit `85c4fbf753b6eb47613bdd7ea423c38b48054544`, no tags, no releases, and no declared license. The repository was created 2025-05-23, one day before arXiv v1's publication date. C++/CUDA-adjacent code was first added in commit `3c7e3f54342f01b4a29be0b03f7cd6743903a846` on 2025-05-23; Python source was added in `7e019cd888780e2535cc1d1f4d5f3f23c9d902c6` on 2025-06-29. Its README names the same system, models, traces, hardware, and mechanisms as the paper.

This is strong circumstantial provenance, but not verified author attribution:

- the supplied PDF does not link it;
- current `ds2-lab/MorphServe` does not link it;
- the GitHub user `MorphServe` has no public person/lab identity;
- the candidate repository has no license file or release/tag.

Therefore the vendored snapshot at `../vendor/author-morphserve/` is labeled **candidate project-account artifact**, not verified author code. Its exact files are covered by `MANIFEST.sha256` and must remain immutable.

### Repository-history and bytecode provenance

A full local history audit (`results/raw/candidate-git-history-audit.txt`) finds 24 commits, one branch, and no tags. C++/build artifacts first appear in `3c7e3f5...`; all Python source arrives in one later commit, `7e019cd...`; subsequent non-README history contains only a requirements change. Thus no deleted experiment/config/profile files are recoverable from another public commit or ref.

The committed snapshot also contains 24 timestamp-based CPython 3.11 `.pyc` files. A non-executing header/marshal inventory (`results/raw/candidate-pyc-audit.json`) shows compilation timestamps from 2025-03-12 through 2025-04-27 and embedded source paths under `/mnt/ssd_smart/alex/swiftLLM/swiftllm`. Sixteen headers match the current source byte count, while eight—including engine config, scheduler, engine, memory utilities and model files—do not. This supports SwiftLLM-derived development provenance but proves some bytecode was compiled from different source bytes; it neither authenticates authorship nor supplies trustworthy hidden paper configurations. No candidate bytecode was executed.

## Candidate artifact contents and gaps

Present:

- SwiftLLM-derived scheduler, block manager, model, and Triton kernels;
- packed AWQ layer reconstruction using `WQLinear`;
- contiguous pinned CPU buffers for FP16 and quantized decoder weights;
- C++ `cudaMemcpyAsync` into registered FP16 GPU layer regions;
- C++ storage views over the freed tails of W4 layers;
- dynamic block-count extension and custom Triton store/attention pointer mapping.

Absent or unusable as released:

- no examples, experiment scripts, configurations, saved LIS sequences, replay traces, task preprocessing, raw results, tests, or plotting scripts;
- no offline LIS profiler at all;
- no default/performance/accuracy controller modes, threshold sets, smoothing, persistence, or hysteresis;
- no exact paper trace windows, DuReader translation, baseline configs, or model revisions;
- top-level `setup.py` declares package `morphserve`, but the directory is `MorphServe` and every internal import targets `swiftllm`; unmodified editable installation cannot provide the expected module;
- `EngineConfig.add_cli_args` defines only inherited core arguments, while the dataclass requires `quantized_model_path`, `quantized_q_config`, `gpu_kv_cache_threshold`, quantization limits/modes, `req_preempt`, and `block_swap`; `api_server.py` cannot construct the dataclass from its parser;
- `AWQLayerManager` calls `torch.load` on one path, but the available public/local AWQ format is sharded safetensors;
- candidate imports `mit-han-lab/llm-awq`'s `awq.quantize.qmodule.WQLinear` but constructs it with `qweight/scales/scaled_zeros` keyword arguments absent from the public class; the local AutoAWQ checkpoint instead stores `qweight/qzeros/scales`, so neither public dependency/checkpoint matches the released call site;
- C++ copies use a new stream followed immediately by `cudaStreamSynchronize`, so the released path is blocking despite `cudaMemcpyAsync`;
- Python destroys/reconstructs layer objects and C++ creates new tensor/storage views, contrary to a literal object-pointer interpretation of “without pointer remapping”;
- candidate runtime swaps layers from the end toward the front, while the paper says Front-to-Back is the default for unprofiled models;
- recovery checks only that the last reclaimed block range is free; it does not record/wait on in-flight CUDA consumers;
- preemption accounting is internally disconnected: swap-out increments required config field `req_preempt`, while the metrics loop reports `ServerMetrics.preemption_count`, which is initialized to zero and never incremented anywhere in the candidate Python tree;
- source contains explicit TODOs around new-KV attention correctness and profiling.

## SwiftLLM baseline

`https://github.com/interestingLSY/swiftLLM` was pinned at `682cf9a28f97f7490409981a2f181528f377eb5d` and archived to `../vendor/swiftllm-upstream/`. The candidate repository does not record its SwiftLLM base commit, so diff-derived “added LOC” or exact scheduler preservation cannot be proven from repository history.

A prior local project, `/nfs/home/s314511048/precision-batching`, found and repaired Llama 3 RoPE and Llama 3.1 untied-`lm_head` correctness issues in this later SwiftLLM snapshot. Those fixes are useful reconstruction evidence, but they are not part of the candidate artifact and cannot be silently attributed to MorphServe.

## Exact reproduction blockers after source audit

1. Verifiable author link or signed/official release for the candidate code.
2. Complete runnable release or missing configuration/examples for the candidate code.
3. Model/tokenizer revisions and AWQ generation parameters for all four models.
4. LIS calibration samples, seed, representation reduction, and saved orders.
5. Azure/BurstGPT files, 72-second offsets, scaling operation, context mapping, and seed.
6. English-translated DuReader artifact, splits, prompts, truncation/EOS/decoding rules, and metric package versions.
7. Default/performance/accuracy controller policies.
8. LLM-PQ and PyramidKV exact baseline configurations.
9. L4 24 GB + 256 GB host and A100 80 GB + 2 TB host hardware. Local RTX 3090 tests are necessarily modified-condition.

## Recheck commands

```bash
git ls-remote --heads --tags https://github.com/ds2-lab/MorphServe.git
git ls-remote --heads --tags https://github.com/MorphServe/MorphServe.git
curl -fsSL 'https://api.github.com/repos/ds2-lab/MorphServe/git/trees/main?recursive=1'
curl -fsSL 'https://api.github.com/repos/ds2-lab/MorphServe/releases?per_page=100'
curl -fsSL 'https://api.github.com/repos/MorphServe/MorphServe/git/trees/main?recursive=1'
curl -fsSL 'https://api.github.com/repos/MorphServe/MorphServe/commits?per_page=100'
```
