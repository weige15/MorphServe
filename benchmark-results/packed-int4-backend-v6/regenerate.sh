#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
VENV=/nfs/home/s314511048/.venv
export PYTHONPATH="$PWD/swiftLLM"

"$VENV/bin/python" -m benchmark.analyze_packed_backend \
  --root benchmark-results/packed-int4-backend-v6 \
  --output-dir benchmark-results/packed-int4-backend-v6/analysis

"$VENV/bin/python" -m benchmark.audit_packed_backend \
  --root benchmark-results/packed-int4-backend-v6 \
  --output benchmark-results/packed-int4-backend-v6/completion_audit.json
