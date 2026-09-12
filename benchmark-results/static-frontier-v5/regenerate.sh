#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
VENV=/nfs/home/s314511048/.venv
export PYTHONPATH="$PWD/swiftLLM"

"$VENV/bin/python" -m benchmark.analyze_static_frontier \
  --manifest benchmark-results/static-frontier-v5/protocol_manifest.json \
  --quality-runs benchmark-results/static-frontier-v5/quality/runs/quality-{fp16_0,w4_8,w4_16,w4_32} \
  --serving-runs benchmark-results/static-frontier-v5/serving/runs/serv-* \
  --profile-files benchmark-results/static-frontier-v5/mechanism/profile-*.json \
  --microbenchmark-files benchmark-results/static-frontier-v5/mechanism/micro-{fp16_0,w4_8,w4_16,w4_32}.json \
  --output-dir benchmark-results/static-frontier-v5/analysis

"$VENV/bin/python" -m benchmark.audit_static_frontier \
  --root benchmark-results/static-frontier-v5 \
  --output benchmark-results/static-frontier-v5/completion_audit.json
