# Regenerated figures

## `transfer-diagnostics.{pdf,png,csv}`

This is a modified-condition diagnostic, **not a reproduction of a numbered paper figure**. It is generated directly from the immutable raw records in:

- `../experiments/async-layer-transfer/full-model-results-attempt-1/metrics.json`

The plot intentionally includes only transfer duration and size-normalized transfer rate. It excludes decode and overlap values because attempt 1 failed its steady-state overlap gate. The CSV is the exact tabular plotting input derived from those raw records.

Regenerate from a clean temporary environment:

```bash
rm -rf /tmp/morphserve-analysis-venv
uv venv --python python3.12 /tmp/morphserve-analysis-venv
uv pip install --python /tmp/morphserve-analysis-venv/bin/python \
  -r reproduction/configs/analysis-requirements.txt
/tmp/morphserve-analysis-venv/bin/python \
  reproduction/figures/gen_fig_transfer_diagnostics.py \
  --input reproduction/experiments/async-layer-transfer/full-model-results-attempt-1/metrics.json \
  --output-prefix reproduction/figures/transfer-diagnostics
```

Expected SHA-256 after regeneration:

```text
86bfb5f61bc609574062fa761d11bc43b7c7de39cd5a1e4083341fe64a6b713f  transfer-diagnostics.csv
04eefa73e916892f131db8cbd4ae1194927d92d4d9d590b0d26d580cfaae0ac8  transfer-diagnostics.pdf
744de99e381e68b38fa01cbbf5b1a5b93d1aba0bfd908e3e1ce54b16a2ab0e80  transfer-diagnostics.png
```

The generator removes time-varying PDF metadata; an immediate second run was byte-identical for CSV, PDF, and PNG.
