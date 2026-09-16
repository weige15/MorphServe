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
