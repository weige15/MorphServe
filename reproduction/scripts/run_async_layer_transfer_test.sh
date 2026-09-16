#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); VENV="$ROOT/.venv"; LOCK="$ROOT/configs/fp16-requirements-lock.txt"; TMP="$ROOT/results/tmp/async-layer-transfer-csrc"; OUT="$ROOT/experiments/async-layer-transfer/results"; GPU=${CUDA_VISIBLE_DEVICES:-0}
rm -rf "$OUT" "$TMP"; mkdir -p "$OUT" "$ROOT/results/tmp"; cp -a "$ROOT/runtime/candidate-csrc" "$TMP"
git -C "$ROOT/.." rev-parse HEAD > "$OUT/source-revision.txt"
git -C "$ROOT/.." status --short -- reproduction/runtime reproduction/tests/test_async_layer_transfer.py reproduction/scripts/run_async_layer_transfer_test.sh reproduction/configs/fp16-requirements-lock.txt > "$OUT/source-status.txt"
cat > "$OUT/commands.txt" <<EOF
uv venv --clear --python python3.12 "$VENV" && uv pip install --python "$VENV/bin/python" -r "$LOCK"
(cd "$TMP" && CUDA_HOME=/usr/local/cuda-12.4 "$VENV/bin/python" setup.py build_ext --inplace)
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$TMP:$ROOT/runtime/candidate-python:$ROOT/runtime" MORPHSERVE_TEST_OUTPUT="$OUT/metrics.json" "$VENV/bin/python" -m unittest reproduction.tests.test_async_layer_transfer -v
EOF
(cd "$ROOT/vendor/author-morphserve" && sha256sum -c MANIFEST.sha256) > "$OUT/manifest-before.log"; nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$OUT/nvidia-before.csv"
uv venv --clear --python python3.12 "$VENV" > "$OUT/venv.log" 2>&1; uv pip install --python "$VENV/bin/python" -r "$LOCK" > "$OUT/install.log" 2>&1
(cd "$TMP" && CUDA_HOME=/usr/local/cuda-12.4 "$VENV/bin/python" setup.py build_ext --inplace) > "$OUT/build.log" 2>&1
start=$(date +%s%N); set +e; CUDA_VISIBLE_DEVICES="$GPU" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$TMP:$ROOT/runtime/candidate-python:$ROOT/runtime" MORPHSERVE_TEST_OUTPUT="$OUT/metrics.json" "$VENV/bin/python" -m unittest reproduction.tests.test_async_layer_transfer -v > "$OUT/test.log" 2>&1; rc=$?; set -e; end=$(date +%s%N); echo "$rc" > "$OUT/test.exitcode"
python3 - "$start" "$end" > "$OUT/wall-time.json" <<'PY'
import json,sys
s,e=map(int,sys.argv[1:]); print(json.dumps({'test_wall_seconds':(e-s)/1e9},indent=2))
PY
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$OUT/nvidia-after.csv"; (cd "$ROOT/vendor/author-morphserve" && sha256sum -c MANIFEST.sha256) > "$OUT/manifest-after.log"; cat "$OUT/test.log"; exit "$rc"
