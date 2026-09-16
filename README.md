# MorphServe Reproduction

## What This Project Does

Evidence-backed reproduction workspace for the supplied MLSys 2026 MorphServe paper. **Status: active, partial modified-condition reproduction.** The official lab repository remains README-only; this project preserves immutable candidate artifacts, an independent repaired reconstruction, real GPU evidence, negative results, and exact blockers without claiming unverified paper numbers.

## Quickstart

```bash
cd /nfs/home/s314511048/MorphServe
cat reproduction/research-state.yaml
python3 reproduction/scripts/verify_report_artifacts.py
```

Full CPU and GPU commands are in [`reproduction/doc/onboarding.md`](reproduction/doc/onboarding.md). GPU runners require CUDA 12.4, local checkpoints, and sufficient free memory; pending 8B reruns fail fast below 17 GiB free.

## Project Structure

- [`reproduction/REPORT.md`](reproduction/REPORT.md): current claim-by-claim report
- [`reproduction/docs/completion-audit.md`](reproduction/docs/completion-audit.md): prompt-to-artifact gap audit
- [`reproduction/runtime/`](reproduction/runtime/): isolated reconstruction and repairs
- [`reproduction/experiments/`](reproduction/experiments/): protocols, attempts, raw logs, metrics
- [`reproduction/vendor/`](reproduction/vendor/): immutable pinned upstream snapshots

## Testing and Documentation

Start with [`reproduction/README.md`](reproduction/README.md), [`reproduction/doc/onboarding.md`](reproduction/doc/onboarding.md), and the current checkpoint in [`reproduction/research-state.yaml`](reproduction/research-state.yaml). Known failures and restart commands are documented there; exact trace scaling, task mappings, model revisions, baselines, and paper hardware remain TODOs blocked on missing source/resources.
