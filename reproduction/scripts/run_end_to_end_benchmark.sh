#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV="$ROOT/.venv"; LOCK="$ROOT/configs/fp16-requirements-lock.txt"; CSRC="$ROOT/runtime/candidate-csrc"; TMP="$ROOT/results/tmp/end-to-end"
FP16=${MORPHSERVE_FP16_MODEL_PATH:-/nfs/home/s314511048/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b}
W4=${MORPHSERVE_W4_MODEL_PATH:-/nfs/home/s314511048/.cache/morphserve/llama31-8b-autoawq-w4-g128-zp}
GPU=${CUDA_VISIBLE_DEVICES:-1}; KIND=; TRACE=; OUT=; PREFLIGHT=0; MODE=default
usage(){ echo "usage: CUDA_VISIBLE_DEVICES=N $0 --kind fp16|static_w4|morphserve-default --trace MANIFEST --output-dir DIR [--preflight] [--mode default]" >&2; exit 2; }
while (($#)); do case "$1" in
  --kind) KIND=$2; shift 2;; --trace) TRACE=$2; shift 2;; --output-dir) OUT=$2; shift 2;; --preflight) PREFLIGHT=1; shift;; --mode) MODE=$2; shift 2;; *) usage;; esac; done
[[ -n "$KIND" && -n "$TRACE" && -n "$OUT" ]] || usage
if [[ "$OUT" != /* ]]; then OUT="$ROOT/$OUT"; fi
if [[ "$TRACE" != /* ]]; then TRACE="$ROOT/$TRACE"; fi
mkdir -p "$ROOT/results/tmp" "$OUT"; rm -rf "$TMP"; mkdir -p "$TMP"; cp -a "$CSRC" "$TMP/csrc"
command -v nvidia-smi >/dev/null; command -v uv >/dev/null
[[ -f "$FP16/config.json" && -f "$W4/model.safetensors.index.json" ]] || { echo "model/W4 asset missing" >&2; exit 75; }
free_mib=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
if (( free_mib < 17408 )); then echo "GPU $GPU has $free_mib MiB free; refusing run" | tee "$OUT/run.stderr.log" >&2; echo 75 > "$OUT/run.exitcode"; exit 75; fi
printf '%s\n' "$(git -C "$ROOT/.." rev-parse HEAD)" > "$OUT/source-revision.txt"
git -C "$ROOT/.." status --short -- reproduction/runtime reproduction/scripts reproduction/configs > "$OUT/source-status.txt"
cat > "$OUT/commands.txt" <<EOF
uv pip install --python "$VENV/bin/python" -r "$LOCK" && uv pip install --python "$VENV/bin/python" autoawq==0.2.9 zstandard
(cd "$TMP/csrc" && CUDA_HOME=/usr/local/cuda-12.4 "$VENV/bin/python" setup.py build_ext --inplace)
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$TMP/csrc:$ROOT/runtime/candidate-python:$ROOT/runtime" "$VENV/bin/python" "$ROOT/scripts/end_to_end_benchmark.py" --fp16-model "$FP16" --w4-model "$W4" --config "$ROOT/configs/end-to-end-benchmark.json" --payload "$ROOT/configs/request-payload-1024.json" --trace-manifest "$TRACE" --output-dir "$OUT" --kind "$KIND" --mode "$MODE" $([[ $PREFLIGHT == 1 ]] && echo --preflight)
EOF
(cd "$ROOT/vendor/author-morphserve" && sha256sum -c MANIFEST.sha256) > "$OUT/manifest-before.log"
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu,driver_version --format=csv,noheader > "$OUT/nvidia-before.csv"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$OUT/processes-before.csv" || true
uv pip install --python "$VENV/bin/python" -r "$LOCK" > "$OUT/install-base.log" 2>&1
uv pip install --python "$VENV/bin/python" autoawq==0.2.9 zstandard > "$OUT/install-autoawq.log" 2>&1
(cd "$TMP/csrc" && CUDA_HOME=/usr/local/cuda-12.4 "$VENV/bin/python" setup.py build_ext --inplace) > "$OUT/build.log" 2>&1
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu,driver_version --format=csv,noheader > "$OUT/nvidia-pre-run.csv"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$OUT/processes-pre-run.csv" || true
free_mib=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
if (( free_mib < 17408 )); then echo "GPU $GPU fell to $free_mib MiB free after setup; refusing model load" | tee "$OUT/run.stderr.log" >&2; echo 75 > "$OUT/run.exitcode"; exit 75; fi
start=$(date +%s%N); set +e
CUDA_VISIBLE_DEVICES="$GPU" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$TMP/csrc:$ROOT/runtime/candidate-python:$ROOT/runtime" "$VENV/bin/python" "$ROOT/scripts/end_to_end_benchmark.py" --fp16-model "$FP16" --w4-model "$W4" --config "$ROOT/configs/end-to-end-benchmark.json" --payload "$ROOT/configs/request-payload-1024.json" --trace-manifest "$TRACE" --output-dir "$OUT" --kind "$KIND" --mode "$MODE" $([[ $PREFLIGHT == 1 ]] && echo --preflight) > "$OUT/run.stdout.log" 2> "$OUT/run.stderr.log"
rc=$?; set -e; end=$(date +%s%N); echo "$rc" > "$OUT/run.exitcode"
python3 - "$start" "$end" > "$OUT/wall-time.json" <<'PY'
import json,sys
s,e=map(int,sys.argv[1:]); print(json.dumps({'run_wall_seconds':(e-s)/1e9},indent=2))
PY
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu,driver_version --format=csv,noheader > "$OUT/nvidia-after.csv"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$OUT/processes-after.csv" || true
(cd "$ROOT/vendor/author-morphserve" && sha256sum -c MANIFEST.sha256) > "$OUT/manifest-after.log"
cat "$OUT/run.stdout.log"; cat "$OUT/run.stderr.log" >&2; exit "$rc"
