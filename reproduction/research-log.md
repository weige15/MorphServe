# Research log

## 2026-09-16 — checkpoint 1: materials and environment audit

### Hypothesis

The supplied repository may contain only the paper, so exact author-code execution may be blocked; before reconstructing anything, search all public primary repositories and local prior work.

### Actions

- Inspected local Git history, remote, and all top-level files.
- Read and visually checked the full 19-page supplied PDF through a delegated paper audit; extracted the exact arXiv v2 source and all reference tables.
- Queried and preserved GitHub API state for `ds2-lab/MorphServe` and `MorphServe/MorphServe`.
- Pinned and archived SwiftLLM commit `682cf9a...`.
- Audited `/nfs/home/s314511048/precision-batching` to avoid repeating existing SwiftLLM/asset discovery.
- Captured hardware, software, PCIe, memlock, process, disk, and model-file hash evidence in `results/raw/environment.json`.

### Commands/evidence

See `docs/paper-evidence-brief.md`, `docs/author-artifact-audit.md`, `docs/source-map.md`, and `sources/remote-audit/`. Environment command is reproducible via `scripts/capture_environment.py`.

### Observed outcome

- Official `ds2-lab/MorphServe` remains README-only and explicitly promises a future release.
- An earlier `MorphServe/MorphServe` project-account repository contains a partial implementation closely aligned with the paper, but no primary author/lab link verifies ownership.
- The candidate artifact lacks profiler, experiments, configs, tests, controller modes, and exact inputs; it also has packaging/config defects and synchronous behavior around nominally asynchronous copies.
- The supplied PDF has Tables 1–8 only. There is no Table 9. The objective's Table 4 numeric target is actually PDF Table 2.
- Local hardware is 7× RTX 3090 24 GB with 125 GiB RAM, not the L4/A100 systems in the paper.
- Local Llama 3.1 8B base and AWQ W4 assets are available; exact paper model revisions are not.

### Resource cost

No GPU experiments. Network traffic was limited to public source/API downloads. Approximately 30 MiB of paper/source/code evidence was added. No paid resources or external uploads.

### Decision / next step

Treat checkpoint 1 as complete. Prioritize the candidate artifact's unmodified build/import viability, record negative evidence, and make any compatibility repair only in a clearly separate reconstruction tree. Do not start headline serving runs before correctness gates.

## 2026-09-16 — candidate artifact smoke, initial run

### Hypothesis

The candidate source will not install/import unmodified because of package naming and config plumbing, while its C++ source may still build.

### Command

```bash
reproduction/scripts/run_candidate_artifact_smoke.sh
```

### Observed outcome

The editable install, all documented Python package imports, and CLI-to-dataclass construction failed exactly at the predicted seams. The C++ extension compiled successfully. Direct extension import failed because `libc10.so` was not loaded; since the probe did not import PyTorch first, this is not yet a binary-incompatibility result. Both vendor-manifest checks passed. Full evidence is under `experiments/candidate-artifact-smoke/results/` and summarized in `analysis.md`.

### Resource cost and next step

No GPU work or model load. Run one changed-hypothesis follow-up that preloads PyTorch before importing the extension; do not retry any unchanged packaging/config failure.

### Follow-up outcome

The frozen follow-up `import torch; import swiftllm_c` succeeded against the freshly built extension and exposed 13 bindings, including layer registration/replacement and reclaimed-KV operations. This narrows the failure: the C++ source is buildable/importable, while the released Python package and CLI remain non-runnable. The repeat runner's venv-creation step returned 2 because the venv already existed; dependency and build probes still ran, and this harness idempotence issue was fixed without changing the observed mechanism result.

## 2026-09-16 — Algorithm 1 conditioned-MDS selection

### Hypothesis and seam

A three-layer worked example with state-dependent MDS will distinguish the paper's greedy conditioned algorithm from a one-time/static MDS ranking. The public seam is `rank_layers(lts, lrs, mds)` plus canonical `save_profile`.

### Command and outcome

```bash
PYTHONPATH="$PWD/reproduction/runtime" python3 -m unittest reproduction.tests.test_profiling -v
```

The retained red/green loop ends with 4/4 passing tests. The conditioned algorithm makes six MDS calls, selects `[0,2,1]`, while a static-MDS comparator selects `[0,1,2]`. Literal LIS weights, `argmax`, deterministic reconstructed tie-breaking, validation, and full-history JSON are covered.

### Resource cost and next step

CPU-only and zero GPU experiment time. Next, freeze model representation/calibration choices and implement real WikiText-2 metric collection; selection-logic success alone is not a model-profile reproduction.

## 2026-09-16 — candidate C++ memory-manager GPU check

### Hypothesis

A synthetic FP16-sized owner and packed replacement will demonstrate whether the candidate extension converts reclaimed weight bytes into actual bounded KV storage rather than counters.

### Command and observed outcome

```bash
CUDA_VISIBLE_DEVICES=0 reproduction/scripts/run_candidate_memory_manager_test.sh
```

Both tests completed their assertions. The packed view reused the owner base; a 3,072-byte tail exposed 12 correctly bounded K/V blocks; writes changed the owner; allocator delta was zero; undersized allocation failed; restoration recovered original bytes. After unittest printed `OK`, however, Python segfaulted and returned 139.

Targeted controls plus `cuda-gdb` localized the crash to global destruction of static `pybind11::object` tensor metadata after Python finalization, not to CUDA storage. The diagnosis and minimal repair are in `doc/debug-report.md`. This is an evidence-backed negative result for the released extension's process lifetime.

### Resource cost and next step

Sub-second synthetic kernels ran on one RTX 3090 across isolated probes; no model load. Implement the diagnosed native-metadata repair only in a separate reconstruction source, then rerun the identical gate. Do not proceed to full-model or timing claims while exit safety fails.

## 2026-09-16 — native metadata lifetime repair

### Hypothesis and scoped change

Parsing registration dictionaries into native `TensorInfo` records, while leaving all copy/address/KV behavior unchanged, should eliminate the confirmed shutdown crash.

### Command and outcome

```bash
CUDA_VISIBLE_DEVICES=0 \
MORPHSERVE_CSRC="$PWD/reproduction/runtime/candidate-csrc" \
MORPHSERVE_TEST_RESULT_DIR="$PWD/reproduction/experiments/candidate-memory-manager-repair/results" \
reproduction/scripts/run_candidate_memory_manager_test.sh
```

The same two GPU tests passed and the process exited 0. The extension also imported after Torch preload with all 13 bindings. Address, 12-block capacity, writable owner storage, undersized-tail failure, same-base restore, and zero allocator delta were preserved. Vendor manifests remained unchanged.

### Resource cost and next step

Primary verification used 2.94 s process wall time on one RTX 3090, dominated by startup; CUDA-event operations were each under 1.5 ms for tiny buffers and are not paper-comparable. Next test the candidate Triton mapping across at least two non-contiguous reclaimed layer regions before model integration.
