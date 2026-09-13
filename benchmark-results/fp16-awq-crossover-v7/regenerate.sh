#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON=/nfs/home/s314511048/.venv/bin/python
export PYTHONPATH="$ROOT/swiftLLM"

rm -rf benchmark-results/fp16-awq-crossover-v7/analysis
"$PYTHON" -m benchmark.analyze_crossover \
  --root benchmark-results/fp16-awq-crossover-v7 \
  --output-dir benchmark-results/fp16-awq-crossover-v7/analysis \
  > benchmark-results/fp16-awq-crossover-v7/logs/analysis.log
"$PYTHON" -m benchmark.audit_crossover \
  --root benchmark-results/fp16-awq-crossover-v7 \
  --output benchmark-results/fp16-awq-crossover-v7/completion_audit.json \
  > benchmark-results/fp16-awq-crossover-v7/logs/audit.log
