#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON=/nfs/home/s314511048/.venvs/morphserve-vllm0112/bin/python
export PYTHONPATH="$ROOT/swiftLLM:$ROOT/swiftLLM/csrc"
V11=benchmark-results/throughput-confirmation-v11

"$PYTHON" -m unittest benchmark.test_throughput_confirmation -v
"$PYTHON" -m benchmark.replay_throughput_controller_parity \
  --output "$V11/archived_policy_parity.json"
"$PYTHON" -m benchmark.analyze_throughput_confirmation \
  --root "$V11" \
  --plan "$V11/phase-a-run-plan.json" \
  --status "$V11/phase-a-execution-status.json" \
  --output "$V11/analysis/phase_a"
"$PYTHON" -m benchmark.analyze_throughput_confirmation \
  --root "$V11" \
  --plan "$V11/phase-c-run-plan.json" \
  --status "$V11/phase-c-execution-status.json" \
  --output "$V11/analysis/phase_c"
"$PYTHON" - <<'PY'
from pathlib import Path
for path in Path("benchmark-results/throughput-confirmation-v11/analysis/phase_c").glob("*.csv"):
    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
PY
"$PYTHON" -m benchmark.audit_throughput_confirmation \
  --root "$V11" \
  --manifest "$V11/phase-c-protocol-manifest.json" \
  --plan-key phase_c \
  --plan "$V11/phase-c-run-plan.json" \
  --status "$V11/phase-c-execution-status.json" \
  --output "$V11/phase-c-audit.json"
"$PYTHON" - <<'PY'
import json
from pathlib import Path
root = Path("benchmark-results/throughput-confirmation-v11")
assert json.loads((root / "phase-a-audit.json").read_text())["status"] == "PASS"
assert json.loads((root / "phase-c-audit.json").read_text())["status"] == "PASS"
assert json.loads((root / "archived_policy_parity.json").read_text())["status"] == "PASS"
final = json.loads((root / "analysis/final_decision.json").read_text())
assert final["v10_release_side_systems_decision"] == "NO-GO (unchanged)"
assert final["throughput_confirmation_decision"] == "NO-GO"
print("throughput-confirmation v11 regeneration: PASS; scientific decision: NO-GO")
PY
