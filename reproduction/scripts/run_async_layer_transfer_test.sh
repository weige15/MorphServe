#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); VENV="$ROOT/.venv"; TMP="$ROOT/results/tmp/async-layer-transfer-csrc"; OUT="$ROOT/experiments/async-layer-transfer/results"; GPU=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p "$OUT" "$ROOT/results/tmp"; rm -rf "$TMP"; cp -a "$ROOT/runtime/candidate-csrc" "$TMP"
cat > "$OUT/commands.txt" <<EOF
(cd "$TMP" && CUDA_HOME=/usr/local/cuda-12.4 "$VENV/bin/python" setup.py build_ext --inplace)
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$TMP:$ROOT/runtime/candidate-python:$ROOT/runtime" MORPHSERVE_TEST_OUTPUT="$OUT/metrics.json" "$VENV/bin/python" -m unittest reproduction.tests.test_async_layer_transfer -v
EOF
(cd "$ROOT/vendor/author-morphserve" && sha256sum -c MANIFEST.sha256) > "$OUT/manifest-before.log"; nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$OUT/nvidia-before.csv"
(cd "$TMP" && CUDA_HOME=/usr/local/cuda-12.4 "$VENV/bin/python" setup.py build_ext --inplace) > "$OUT/build.log" 2>&1
set +e; CUDA_VISIBLE_DEVICES="$GPU" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$TMP:$ROOT/runtime/candidate-python:$ROOT/runtime" MORPHSERVE_TEST_OUTPUT="$OUT/metrics.json" "$VENV/bin/python" -m unittest reproduction.tests.test_async_layer_transfer -v > "$OUT/test.log" 2>&1; rc=$?; set -e; echo "$rc" > "$OUT/test.exitcode"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$OUT/nvidia-after.csv"; (cd "$ROOT/vendor/author-morphserve" && sha256sum -c MANIFEST.sha256) > "$OUT/manifest-after.log"; cat "$OUT/test.log"; exit "$rc"
