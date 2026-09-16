#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENDOR="$ROOT/vendor/author-morphserve"
CSRC="$ROOT/runtime/candidate-csrc"
VENV="$ROOT/.venv"
TMP="$ROOT/results/tmp/candidate-lifetime"
OUT="$ROOT/experiments/candidate-lifetime/results"
GPU=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p "$OUT" "$ROOT/results/tmp"
rm -rf "$TMP"
mkdir -p "$TMP"
cp -a "$CSRC" "$TMP/csrc"

cat > "$OUT/commands.txt" <<EOF
uv venv --clear --python python3.12 "$VENV"
uv pip install --python "$VENV/bin/python" torch==2.4.0 setuptools==84.0.0 ninja==1.13.0
(cd "$TMP/csrc" && CUDA_HOME=/usr/local/cuda-12.4 "$VENV/bin/python" setup.py build_ext --inplace)
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$TMP/csrc:$ROOT/runtime:$ROOT/.." MORPHSERVE_TEST_OUTPUT="$OUT/metrics.json" "$VENV/bin/python" -m unittest "$ROOT/tests/test_candidate_lifetime.py" -v
EOF

(cd "$VENDOR" && sha256sum -c MANIFEST.sha256) > "$OUT/manifest-before.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$OUT/nvidia-before.csv"
uv venv --clear --python python3.12 "$VENV" > "$OUT/venv.log" 2>&1
uv pip install --python "$VENV/bin/python" 'torch==2.4.0' 'setuptools==84.0.0' 'ninja==1.13.0' > "$OUT/install.log" 2>&1
(
  cd "$TMP/csrc"
  CUDA_HOME=/usr/local/cuda-12.4 "$VENV/bin/python" setup.py build_ext --inplace
) > "$OUT/build.log" 2>&1
sha256sum "$TMP/csrc"/swiftllm_c*.so > "$OUT/extension.sha256"

start_ns=$(date +%s%N)
set +e
CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$TMP/csrc:$ROOT/runtime:$ROOT/.." \
MORPHSERVE_TEST_OUTPUT="$OUT/metrics.json" \
  "$VENV/bin/python" -m unittest "$ROOT/tests/test_candidate_lifetime.py" -v \
  > "$OUT/test.stdout.log" 2> "$OUT/test.stderr.log"
test_rc=$?
set -e
end_ns=$(date +%s%N)
printf '%s\n' "$test_rc" > "$OUT/test.exitcode"
python3 - "$start_ns" "$end_ns" > "$OUT/wall-time.json" <<'PY'
import json, sys
start, end = map(int, sys.argv[1:])
print(json.dumps({"test_wall_seconds": (end - start) / 1e9}, indent=2, sort_keys=True))
PY
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$OUT/nvidia-after.csv"
(cd "$VENDOR" && sha256sum -c MANIFEST.sha256) > "$OUT/manifest-after.log"
cat "$OUT/test.stdout.log"
cat "$OUT/test.stderr.log" >&2
exit "$test_rc"
