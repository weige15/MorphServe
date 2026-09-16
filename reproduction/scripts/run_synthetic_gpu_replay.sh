#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); VENV="$ROOT/.venv"; LOCK="$ROOT/configs/fp16-requirements-lock.txt"; TMP="$ROOT/results/tmp/synthetic-replay"; OUT="$ROOT/experiments/synthetic-gpu-replay/results"; MODEL=${MORPHSERVE_MODEL_PATH:-/nfs/home/s314511048/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b}; GPU=${CUDA_VISIBLE_DEVICES:-1}
mkdir -p "$OUT" "$ROOT/results/tmp"
git -C "$ROOT/.." rev-parse HEAD > "$OUT/source-revision.txt"
git -C "$ROOT/.." status --short -- reproduction/runtime reproduction/scripts/synthetic_gpu_replay.py reproduction/scripts/run_synthetic_gpu_replay.sh > "$OUT/source-status.txt"
free_mib=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
if (( free_mib < 17408 )); then printf 'GPU %s has %s MiB free; full-model replay requires at least 17408 MiB.\n' "$GPU" "$free_mib" >&2; exit 75; fi
rm -rf "$TMP"; mkdir -p "$TMP"; cp -a "$ROOT/runtime/candidate-csrc" "$TMP/csrc"
cat > "$OUT/commands.txt" <<EOF
uv venv --clear --python python3.12 "$VENV" && uv pip install --python "$VENV/bin/python" -r "$LOCK"
(cd "$TMP/csrc" && CUDA_HOME=/usr/local/cuda-12.4 "$VENV/bin/python" setup.py build_ext --inplace)
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$TMP/csrc:$ROOT/runtime/candidate-python:$ROOT/runtime" "$VENV/bin/python" "$ROOT/scripts/synthetic_gpu_replay.py" --model "$MODEL" --config "$ROOT/configs/synthetic-gpu-replay.json" --output-dir "$OUT"
EOF
(cd "$ROOT/vendor/author-morphserve" && sha256sum -c MANIFEST.sha256) > "$OUT/manifest-before.log"; nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$OUT/nvidia-before.csv"
uv venv --clear --python python3.12 "$VENV" > "$OUT/venv.log" 2>&1; uv pip install --python "$VENV/bin/python" -r "$LOCK" > "$OUT/install.log" 2>&1; (cd "$TMP/csrc" && CUDA_HOME=/usr/local/cuda-12.4 "$VENV/bin/python" setup.py build_ext --inplace) > "$OUT/build.log" 2>&1
start=$(date +%s%N); set +e; CUDA_VISIBLE_DEVICES="$GPU" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$TMP/csrc:$ROOT/runtime/candidate-python:$ROOT/runtime" "$VENV/bin/python" "$ROOT/scripts/synthetic_gpu_replay.py" --model "$MODEL" --config "$ROOT/configs/synthetic-gpu-replay.json" --output-dir "$OUT" > "$OUT/run.stdout.log" 2> "$OUT/run.stderr.log"; rc=$?; set -e; end=$(date +%s%N); echo "$rc" > "$OUT/run.exitcode"
python3 - "$start" "$end" > "$OUT/wall-time.json" <<'PY'
import json,sys
s,e=map(int,sys.argv[1:]); print(json.dumps({'run_wall_seconds':(e-s)/1e9},indent=2))
PY
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$OUT/nvidia-after.csv"; (cd "$ROOT/vendor/author-morphserve" && sha256sum -c MANIFEST.sha256) > "$OUT/manifest-after.log"; cat "$OUT/run.stdout.log"; cat "$OUT/run.stderr.log" >&2; exit "$rc"
