#!/usr/bin/env bash
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENDOR="$ROOT/vendor/author-morphserve"
VENV="$ROOT/.venv"
TMP="$ROOT/results/tmp/candidate-artifact-smoke"
OUT="$ROOT/experiments/candidate-artifact-smoke/results"
mkdir -p "$OUT" "$ROOT/results/tmp"
rm -rf "$TMP"
cp -a "$VENDOR" "$TMP"
: > "$OUT/status.tsv"
: > "$OUT/commands.txt"

run_logged() {
  local name=$1
  shift
  printf '%q ' "$@" >> "$OUT/commands.txt"
  printf '\n' >> "$OUT/commands.txt"
  "$@" >"$OUT/${name}.stdout.log" 2>"$OUT/${name}.stderr.log"
  local rc=$?
  printf '%s\t%s\n' "$name" "$rc" >> "$OUT/status.tsv"
  return 0
}

(
  cd "$VENDOR"
  sha256sum -c MANIFEST.sha256
) > "$OUT/manifest-before.log" 2>&1
printf 'manifest_before\t%s\n' "$?" >> "$OUT/status.tsv"

run_logged create_venv uv venv --python python3.12 "$VENV"

# Exact documented top-level editable install, on a byte-identical disposable copy.
run_logged editable_install_unmodified uv pip install --python "$VENV/bin/python" -e "$TMP"

# Import probes do not write bytecode into the immutable vendor tree.
run_logged import_MorphServe env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$TMP" "$VENV/bin/python" -c 'import MorphServe; print(MorphServe.__file__)'
run_logged import_morphserve env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$TMP" "$VENV/bin/python" -c 'import morphserve; print(morphserve.__file__)'
run_logged import_swiftllm env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$TMP" "$VENV/bin/python" -c 'import swiftllm; print(swiftllm.__file__)'

# Install only the pinned build stack used by the already-audited local SwiftLLM environment.
run_logged install_build_stack uv pip install --python "$VENV/bin/python" \
  'torch==2.4.0' 'setuptools==84.0.0' 'ninja==1.13.0'

# Build the released extension from the disposable exact copy.
run_logged build_extension bash -lc "cd '$TMP/csrc' && CUDA_HOME=/usr/local/cuda-12.4 '$VENV/bin/python' setup.py build_ext --inplace"
run_logged import_extension env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$TMP/csrc" "$VENV/bin/python" -c \
  'import swiftllm_c; print(swiftllm_c.__file__); print(sorted(x for x in dir(swiftllm_c) if not x.startswith("_")))'

# Static dataclass/parser construction probe, loaded directly to isolate it from broken package imports.
run_logged engine_config_probe env PYTHONDONTWRITEBYTECODE=1 "$VENV/bin/python" - "$TMP/MorphServe/engine_config.py" <<'PY'
import argparse
import dataclasses
import importlib.util
import sys
path = sys.argv[1]
spec = importlib.util.spec_from_file_location("candidate_engine_config", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
parser = argparse.ArgumentParser()
module.EngineConfig.add_cli_args(parser)
args = vars(parser.parse_args(["--model-path", "/tmp/model"]))
required = {field.name for field in dataclasses.fields(module.EngineConfig)}
print("parser_keys", sorted(args))
print("missing", sorted(required - set(args)))
try:
    module.EngineConfig(**args)
except Exception as exc:
    print(type(exc).__name__, str(exc))
    raise
PY

(
  cd "$VENDOR"
  sha256sum -c MANIFEST.sha256
) > "$OUT/manifest-after.log" 2>&1
printf 'manifest_after\t%s\n' "$?" >> "$OUT/status.tsv"

"$VENV/bin/python" - "$OUT" <<'PY'
import hashlib
import json
import pathlib
import sys
out = pathlib.Path(sys.argv[1])
status = {}
for line in (out / "status.tsv").read_text().splitlines():
    name, rc = line.split("\t")
    status[name] = int(rc)
artifacts = []
for path in sorted(out.glob("*")):
    if path.is_file() and path.name != "summary.json":
        artifacts.append({
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
summary = {
    "schema_version": 1,
    "candidate_commit": "85c4fbf753b6eb47613bdd7ea423c38b48054544",
    "status": status,
    "classification": (
        "unmodified_artifact_viable" if all(status.get(k) == 0 for k in (
            "editable_install_unmodified", "import_MorphServe", "build_extension",
            "import_extension", "engine_config_probe"
        )) else "not_unmodified_runnable"
    ),
    "artifacts": artifacts,
}
(out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary["status"], sort_keys=True))
print(summary["classification"])
PY
