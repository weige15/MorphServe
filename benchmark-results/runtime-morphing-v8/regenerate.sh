#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
AWQ_ENV=${AWQ_ENV:-/nfs/home/s314511048/.venvs/morphserve-vllm0112}
AWQ_EXT=${AWQ_EXT:-/nfs/home/s314511048/.cache/morphserve/swiftllm-c-torch29/lib}
LEGACY_ENV=${LEGACY_ENV:-/nfs/home/s314511048/.venv}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}
export CUDA_VISIBLE_DEVICES
mkdir -p benchmark-results/runtime-morphing-v8/logs

PYTHONPATH="$AWQ_EXT:$ROOT/swiftLLM" "$AWQ_ENV/bin/python" \
  -m benchmark.analyze_runtime_morphing \
  --root benchmark-results/runtime-morphing-v8 \
  2>&1 | tee benchmark-results/runtime-morphing-v8/logs/analysis.log

PYTHONPATH="$ROOT/swiftLLM:$ROOT/swiftLLM/csrc" "$LEGACY_ENV/bin/python" -m unittest \
  benchmark.test_benchmark benchmark.test_inference_substrate \
  benchmark.test_crossover benchmark.test_runtime_morphing -v \
  2>&1 | tee benchmark-results/runtime-morphing-v8/logs/tests-legacy.log

PYTHONPATH="$AWQ_EXT:$ROOT/swiftLLM" "$AWQ_ENV/bin/python" -m unittest \
  benchmark.test_benchmark benchmark.test_inference_substrate \
  benchmark.test_crossover benchmark.test_runtime_morphing -v \
  2>&1 | tee benchmark-results/runtime-morphing-v8/logs/tests-awq.log

MORPHSERVE_RUN_CUDA_TESTS=1 PYTHONPATH="$AWQ_EXT:$ROOT/swiftLLM" \
  "$AWQ_ENV/bin/python" -m unittest benchmark.test_runtime_morphing -v \
  2>&1 | tee benchmark-results/runtime-morphing-v8/logs/cuda-tests.log

PYTHONPATH="$ROOT/swiftLLM:$ROOT/swiftLLM/csrc" "$LEGACY_ENV/bin/python" -m py_compile \
  swiftLLM/benchmark/*.py swiftLLM/swiftllm/server/*.py \
  swiftLLM/swiftllm/worker/*.py swiftLLM/swiftllm/worker/layers/*.py \
  swiftLLM/swiftllm/worker/kernels/*.py

git diff --check

PYTHONPATH="$AWQ_EXT:$ROOT/swiftLLM" "$AWQ_ENV/bin/python" \
  -m benchmark.audit_runtime_morphing \
  --root benchmark-results/runtime-morphing-v8 \
  2>&1 | tee benchmark-results/runtime-morphing-v8/logs/audit.log
