#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
AWQ_ENV=/nfs/home/s314511048/.venvs/morphserve-vllm0112
AWQ_EXT=/nfs/home/s314511048/.cache/morphserve/swiftllm-c-torch29/lib
export PYTHONPATH="$AWQ_EXT:$ROOT/swiftLLM"

mkdir -p benchmark-results/closed-loop-runtime-v9/logs
"$AWQ_ENV/bin/python" -m unittest \
  benchmark.test_benchmark \
  benchmark.test_inference_substrate \
  benchmark.test_crossover \
  benchmark.test_runtime_morphing \
  benchmark.test_closed_loop_controller -v \
  2>&1 | tee benchmark-results/closed-loop-runtime-v9/logs/tests.log
"$AWQ_ENV/bin/python" -m benchmark.analyze_closed_loop \
  --root benchmark-results/closed-loop-runtime-v9 \
  2>&1 | tee benchmark-results/closed-loop-runtime-v9/logs/analysis.log
"$AWQ_ENV/bin/python" -m benchmark.verify_closed_loop_results \
  --root benchmark-results/closed-loop-runtime-v9 \
  2>&1 | tee benchmark-results/closed-loop-runtime-v9/logs/independent-verification.log
"$AWQ_ENV/bin/python" -m benchmark.audit_closed_loop \
  --root benchmark-results/closed-loop-runtime-v9 \
  2>&1 | tee benchmark-results/closed-loop-runtime-v9/logs/audit.log
