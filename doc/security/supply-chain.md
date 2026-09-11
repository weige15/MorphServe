# Supply Chain Check

Scope: audit of the pushed tip `0d7b6f0` (`origin/main` pointed to the same commit). This report records the pre-cleanup inventory; the naming refactor and obsolete-artifact cleanup are committed locally before the history rewrite.

## Dependencies

- Python dependency source: `swiftLLM/requirements.txt`.
- Requirements use lower bounds rather than exact versions (`fastapi>=0.111`, `transformers>=4.40`, etc.).
- The benchmark manifest records the environment versions used for the final run, but those versions are not reproducible from the repository alone.

## Lockfile

No Python lockfile (`uv.lock`, `poetry.lock`, or `Pipfile.lock`) is present. Exact environment recreation requires the external validated environment documented in the reports.

## Build Provenance

- The final static-quantization result is documented in `docs/table5-reproduction/final-report.md` and `benchmark-results/table5-reproduction/protocol_manifest.json` in the pushed tip.
- Final v2 runs are the eight `runs/v2-*` directories. The report's regeneration command consumes the saved manifest and raw run files.
- A clean checkout still cannot reproduce the benchmark without the external Llama checkpoint, BurstGPT/DuReader source data, CUDA GPU, and local paths such as `/tmp/BurstGPT` and `/tmp/DuReader`.

## Generated Artifacts

### Inventory

- The pushed tip contains **102 tracked JSONL files totaling 49,047,592 bytes (46.78 MiB)**.
- The latest commit added **78 JSONL files totaling 48,208,197 bytes (45.97 MiB)**, so the suspected upload is confirmed.
- `benchmark-results/table5-reproduction/` alone is **48,650,299 bytes (46.40 MiB)** across 189 files.
- The latest commit also added 25 run directories: 8 final v2 runs, 8 superseded v1 runs, and 9 pilot runs.

### Necessity assessment

| Artifact set | Role | Assessment |
|---|---|---|
| `input/` | Final frozen trace/dataset/workload | **Keep** for final provenance and regeneration. |
| `runs/v2-*/requests.jsonl` | Final per-request outputs, timestamps, references, and metrics inputs | **Keep**; this is the important raw quality/latency evidence. |
| `runs/v2-*/telemetry.jsonl` | Final queue/KV/GPU telemetry | **Keep** if the resource-behavior claims remain part of the result. |
| `runs/v2-*/quality_metrics.jsonl` | Per-request quality/SLO rows derived by `table5_analyze.py` from `requests.jsonl` | **Optional/redundant**. It is reproducible and accounts for about 3.56 MiB; remove only after updating the report/audit to state that it is regenerated. |
| `input-v1/` | Superseded workload inputs; DuReader and BurstGPT files are byte-identical to `input/`, while the workload differs because v1 prompt construction was replaced | **Not required** for final reproduction. Keep only if preserving iteration history is intentional. |
| `runs-v1/` and `logs-v1/` | Superseded prompt-construction experiment | **Not required** for final reproduction. The final report explicitly calls v2 final, but the completion audit currently mentions retaining v1. |
| `runs/pilot20-*` and `runs/pilot50-*` plus pilot logs | Pilot/setup runs not referenced by the final report or completion audit | **Strong removal candidate**. They are historical evidence, not final result evidence. |

The cleanup removed approximately **27.2 MiB**, including about **52 historical JSONL files**. The current tip retains only final v2 raw JSONL plus `input/` for the static-quantization benchmark; the full repository now has 50 JSONL files totaling 19.61 MiB, including the older Phase 1 evidence.

The JSONL artifacts contain benchmark outputs and public dataset-derived records, not model weights. No committed cache, virtual environment, wheel, checkpoint, or build directory was found.

## Secrets

No API keys, private keys, bearer tokens, or credential files were found in the source/configuration scan. The repository does contain absolute local paths (`/nfs/home/s314511048/...`) in documentation and manifests; these are reproducibility/privacy concerns, not secrets.

## Package Integrity

- Artifact SHA-256 hashes are recorded in the benchmark manifest.
- Final metadata records the manifest hash.
- Generated artifacts have documented commands, but the dependency environment and external input datasets are not pinned by repository lockfiles.
- `.gitignore` only excludes top-level `benchmark-results/*.log`; it does not prevent generated JSONL result directories from being committed accidentally.

## Known Vulnerabilities

`pip-audit` was not available in the environment, so dependency vulnerability scanning was not run. Installing or running an external vulnerability scanner requires separate tooling/approval. No network-based vulnerability query was attempted.

## Findings

- **HIGH — Confirmed:** the latest pushed commit added 48.2 MiB of JSONL benchmark output. Most of it is superseded v1/pilot history rather than required final evidence.
- **MEDIUM — Confirmed:** `quality_metrics.jsonl` is derived from `requests.jsonl`; the final eight copies add 3.56 MiB and can be regenerated.
- **MEDIUM — Confirmed:** no lockfile or exact dependency manifest exists, so the experiment environment is not clean-checkout reproducible.
- **LOW — Confirmed:** absolute user-specific paths are embedded in reports and manifests.
- **INFO — Confirmed:** no secrets or model weights were committed.

## Recommended Fixes

1. For the repository's current tip, retain `input/`, final `runs/v2-*` raw evidence, the manifest, and final derived summaries/plots.
2. **Completed locally:** removed superseded `input-v1/`, `runs-v1/`, and pilot run artifacts, and updated the final report/completion audit so they no longer claim those files are retained.
3. Consider removing final `quality_metrics.jsonl` only if the regeneration command is treated as the source of truth and the audit no longer claims those files are required.
4. Add an explicit generated-artifact policy to `.gitignore` or document an allowlist so future pilot runs are not committed accidentally.
5. If the goal is to reduce Git hosting/history size rather than only the latest checkout, a normal deletion commit is insufficient: the historical blobs remain in Git history. Rewriting history with `git filter-repo`/BFG and force-pushing would require explicit approval and coordination.
6. Add a lockfile or a fully pinned benchmark environment export before claiming clean-environment reproducibility.

## Verification Commands

Run during this audit:

```bash
git status --short --untracked-files=all
git log --oneline --decorate -8
git show --stat --summary --oneline HEAD
git ls-tree -r -l HEAD | awk '$5 ~ /\\.jsonl$/ {sum+=$4; n++} END {print n, sum}'
git diff-tree --no-commit-id -r --numstat HEAD -- '*.jsonl'
rg -n --hidden --glob '!.git/**' --glob '!benchmark-results/**' '(OPENAI_API_KEY|HF_TOKEN|AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|SECRET_KEY|DATABASE_URL|PASSWORD|Bearer |BEGIN .*PRIVATE KEY)'
```

Not run:

- `pip-audit`: command unavailable.
- History rewrite: completed locally after the cleanup commit was verified; the pre-rewrite repository is preserved in `/tmp/MorphServe-before-history-rewrite-299ca28.bundle`.

Artifact deletion and history rewrite are complete locally; the pushed remote is unchanged until the final force-with-lease push.

## Final Risk Rating

**Moderate risk for repository sharing.** No secret leakage or model-weight exposure was found, and the final v2 result has auditable raw evidence. The repository is unnecessarily large because superseded pilot/v1 JSONL artifacts were pushed, and exact dependency/environment reproduction is incomplete. The minimum cleanup is to remove or archive the superseded artifacts and update the audit/report references; history rewriting is only needed if remote storage size must also be reduced.
