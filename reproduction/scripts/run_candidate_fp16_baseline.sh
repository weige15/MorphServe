#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENDOR="$ROOT/vendor/author-morphserve"
CSRC="$ROOT/runtime/candidate-csrc"
PYTHON_SRC="$ROOT/runtime/candidate-python"
LOCK="$ROOT/configs/fp16-requirements-lock.txt"
VENV="$ROOT/.venv"
TMP="$ROOT/results/tmp/candidate-fp16"
OUT="$ROOT/experiments/candidate-fp16-baseline/results"
MODEL=${MODEL_PATH:-/nfs/home/s314511048/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b}
GPU=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p "$OUT" "$ROOT/results/tmp"
rm -rf "$TMP"
mkdir -p "$TMP"
cp -a "$CSRC" "$TMP/csrc"

cat > "$OUT/commands.txt" <<EOF
uv venv --clear --python python3.12 "$VENV"
uv pip install --python "$VENV/bin/python" -r "$LOCK"
(cd "$TMP/csrc" && CUDA_HOME=/usr/local/cuda-12.4 "$VENV/bin/python" setup.py build_ext --inplace)
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$TMP/csrc:$PYTHON_SRC" "$VENV/bin/python" "$ROOT/scripts/fp16_candidate_parity.py" --model-path "$MODEL" --output "$OUT/metrics.json"
EOF

(cd "$VENDOR" && sha256sum -c MANIFEST.sha256) > "$OUT/manifest-before.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$OUT/nvidia-before.csv"
uv venv --clear --python python3.12 "$VENV" > "$OUT/venv.log" 2>&1
uv pip install --python "$VENV/bin/python" -r "$LOCK" > "$OUT/install.log" 2>&1
(
  cd "$TMP/csrc"
  CUDA_HOME=/usr/local/cuda-12.4 "$VENV/bin/python" setup.py build_ext --inplace
) > "$OUT/build.log" 2>&1
sha256sum "$TMP/csrc"/swiftllm_c*.so > "$OUT/extension.sha256"
PYTHONPATH="$PYTHON_SRC" "$VENV/bin/python" -m unittest "$ROOT/tests/test_candidate_runtime.py" -v > "$OUT/config-test.log" 2>&1

start_ns=$(date +%s%N)
set +e
CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$TMP/csrc:$PYTHON_SRC" \
  "$VENV/bin/python" "$ROOT/scripts/fp16_candidate_parity.py" \
  --model-path "$MODEL" --output "$OUT/metrics.json" \
  > "$OUT/run.stdout.log" 2> "$OUT/run.stderr.log"
run_rc=$?
set -e
end_ns=$(date +%s%N)
printf '%s\n' "$run_rc" > "$OUT/run.exitcode"
python3 - "$start_ns" "$end_ns" > "$OUT/wall-time.json" <<'PY'
import json, sys
start, end = map(int, sys.argv[1:])
print(json.dumps({"run_wall_seconds": (end - start) / 1e9}, indent=2, sort_keys=True))
PY
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$OUT/nvidia-after.csv"
(cd "$VENDOR" && sha256sum -c MANIFEST.sha256) > "$OUT/manifest-after.log"
cat "$OUT/run.stdout.log"
cat "$OUT/run.stderr.log" >&2
exit "$run_rc"
