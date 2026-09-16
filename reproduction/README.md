# MorphServe reproduction workspace

Evidence-backed reproduction of the supplied 19-page MorphServe conference paper. The investigation is **in progress**; no headline paper claim has been reproduced yet.

## Start here

- [`research-state.yaml`](research-state.yaml) — checkpoint and next action.
- [`docs/paper-evidence-brief.md`](docs/paper-evidence-brief.md) — full visual paper audit, exact Tables 1–8, Figures 1–7, equations, and Algorithm 1.
- [`docs/source-map.md`](docs/source-map.md) — paper requirement → source file/mechanism map.
- [`docs/claim-register.md`](docs/claim-register.md) — reported values, evidence requirements, and current status.
- [`docs/author-artifact-audit.md`](docs/author-artifact-audit.md) — official README-only repo and candidate project-account code audit.
- [`configs/paper-reference-values.json`](configs/paper-reference-values.json) — machine-readable reported values, never measured output.
- [`results/raw/environment.json`](results/raw/environment.json) — host, GPU, PCIe, software, model inventory, and hashes.

## Important source status

- `vendor/author-morphserve/` is an exact archive of `MorphServe/MorphServe@85c4fbf...`, but its author identity is not linked by the supplied paper or official lab repository. It is labeled a **candidate project-account artifact**, not verified author code.
- `vendor/swiftllm-upstream/` is `interestingLSY/swiftLLM@682cf9a...` for reconstruction/comparison.
- Both vendor trees are immutable and covered by `MANIFEST.sha256`; fixes belong in a separate runtime tree.
- `ds2-lab/MorphServe@1c42999...` is README-only and says full code will be released later.

The PDF contains Tables 1–8 only. There is no Table 9, and the objective's `[5.5223, 0.1241, 27.68]` target is PDF Table 2, not Table 4. This mismatch is retained explicitly.

## Re-capture environment

```bash
PY=/nfs/home/s314511048/precision-batching/.venv/bin/python
$PY scripts/capture_environment.py \
  --output results/raw/environment.json \
  --asset /path/to/model/snapshot \
  --asset /path/to/quantized/model
```

## Current checkpoint

Checkpoint 1 (materials/environment audit) is complete. Checkpoint 2 is testing the candidate source unmodified, then will build a clearly labeled reconstruction only where evidence requires it. See `experiments/candidate-artifact-smoke/protocol.md`.
