#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV="$ROOT/.venv"
LOCK="$ROOT/configs/fp16-requirements-lock.txt"
CSRC="$ROOT/runtime/candidate-csrc"
PYTHON_SRC="$ROOT/runtime/candidate-python"
TMP="$ROOT/results/tmp/active-kv-switch"
OUT=${MORPHSERVE_TEST_RESULT_DIR:-"$ROOT/experiments/active-kv-switch/results"}
FP16=${MORPHSERVE_FP16_MODEL_PATH:-/nfs/home/s314511048/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b}
W4=${MORPHSERVE_W4_MODEL_PATH:-/nfs/home/s314511048/.cache/morphserve/llama31-8b-autoawq-w4-g128-zp}
GPU=${CUDA_VISIBLE_DEVICES:-1}
for file in "$FP16/config.json" "$W4/config.json"; do
  [[ -f "$file" ]] || { echo "missing model config: $file" >&2; exit 2; }
done
mkdir -p "$OUT" "$ROOT/results/tmp"
rm -rf "$TMP"
mkdir -p "$TMP"
cp -a "$CSRC" "$TMP/csrc"
cat > "$OUT/commands.txt" <<EOF
uv venv --clear --python python3.12 "$VENV"
uv pip install --python "$VENV/bin/python" -r "$LOCK"
uv pip install --python "$VENV/bin/python" autoawq==0.2.9 zstandard
(cd "$TMP/csrc" && CUDA_HOME=/usr/local/cuda-12.4 "$VENV/bin/python" setup.py build_ext --inplace)
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$TMP/csrc:$PYTHON_SRC:$ROOT/runtime" "$VENV/bin/python" "$ROOT/scripts/active_kv_switch.py" --fp16-model "$FP16" --w4-model "$W4" --output "$OUT/metrics.json"
EOF
(cd "$ROOT/vendor/author-morphserve" && sha256sum -c MANIFEST.sha256) > "$OUT/manifest-before.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$OUT/nvidia-before.csv"
uv venv --clear --python python3.12 "$VENV" > "$OUT/venv.log" 2>&1
uv pip install --python "$VENV/bin/python" -r "$LOCK" > "$OUT/install-base.log" 2>&1
uv pip install --python "$VENV/bin/python" 'autoawq==0.2.9' 'zstandard' > "$OUT/install-autoawq.log" 2>&1
(
  cd "$TMP/csrc"
  CUDA_HOME=/usr/local/cuda-12.4 "$VENV/bin/python" setup.py build_ext --inplace
) > "$OUT/build.log" 2>&1
sha256sum "$TMP/csrc"/swiftllm_c*.so > "$OUT/extension.sha256"
start_ns=$(date +%s%N)
set +e
CUDA_VISIBLE_DEVICES="$GPU" PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$TMP/csrc:$PYTHON_SRC:$ROOT/runtime" \
  "$VENV/bin/python" "$ROOT/scripts/active_kv_switch.py" \
  --fp16-model "$FP16" --w4-model "$W4" --output "$OUT/metrics.json" \
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
(cd "$ROOT/vendor/author-morphserve" && sha256sum -c MANIFEST.sha256) > "$OUT/manifest-after.log"
cat "$OUT/run.stdout.log"
cat "$OUT/run.stderr.log" >&2
exit "$run_rc"
