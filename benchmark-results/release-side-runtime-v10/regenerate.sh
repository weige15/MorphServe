#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
AWQ_ENV=/nfs/home/s314511048/.venvs/morphserve-vllm0112
AWQ_EXT=/nfs/home/s314511048/.cache/morphserve/swiftllm-c-torch29/lib
export PYTHONPATH="$AWQ_EXT:$ROOT/swiftLLM"

mkdir -p benchmark-results/release-side-runtime-v10/{analysis,development,logs}
"$AWQ_ENV/bin/python" -m unittest \
  benchmark.test_benchmark \
  benchmark.test_inference_substrate \
  benchmark.test_crossover \
  benchmark.test_runtime_morphing \
  benchmark.test_closed_loop_controller \
  benchmark.test_release_intent_controller \
  benchmark.test_release_runtime_telemetry \
  benchmark.test_release_workloads \
  benchmark.test_release_analysis -v \
  2>&1 | tee benchmark-results/release-side-runtime-v10/logs/tests.log
"$AWQ_ENV/bin/python" -m benchmark.analyze_release_latency \
  --output-dir benchmark-results/release-side-runtime-v10/development \
  2>&1 | tee benchmark-results/release-side-runtime-v10/logs/development-replay.log
"$AWQ_ENV/bin/python" -m benchmark.analyze_release_side \
  --root benchmark-results/release-side-runtime-v10 \
  2>&1 | tee benchmark-results/release-side-runtime-v10/logs/analysis.log
"$AWQ_ENV/bin/python" -m benchmark.audit_release_side \
  --root benchmark-results/release-side-runtime-v10 \
  2>&1 | tee benchmark-results/release-side-runtime-v10/logs/audit.log
