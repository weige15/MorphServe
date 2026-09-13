"""Execute the immutable FP16/AWQ crossover run plan serially."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def valid_completed_run(run_dir: Path) -> bool:
    required = ("metadata.json", "requests.jsonl", "telemetry.jsonl", "batches.jsonl", "summary.json")
    if not all((run_dir / name).is_file() for name in required):
        return False
    metadata = read_json(run_dir / "metadata.json")
    summary = read_json(run_dir / "summary.json")
    return (
        metadata.get("completed_request_count") == 64
        and summary.get("completed_request_count") == 64
        and summary.get("failed_or_incomplete_request_count") == 0
        and metadata.get("observed_forward_batch_count", 0) > 0
        and metadata.get("observed_prefill_batch_count", 0) > 0
        and (run_dir / "batches.jsonl").stat().st_size > 0
    )


def command_for(row: dict[str, Any], args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "benchmark.run_static_quantization_condition",
        "--model-path",
        str(args.model_path),
        "--workload",
        str(row["workload"]),
        "--manifest",
        str(args.manifest),
        "--condition",
        str(row["condition"]),
        "--quantized-layer-count",
        str(row["quantized_layer_count"]),
        "--output-dir",
        str(args.output_dir),
        "--run-id",
        str(row["run_id"]),
        "--launch-mode",
        "trace_open_loop",
        "--warmup-requests",
        "1",
        "--warmup-output-token-count",
        "8",
        "--seed",
        "2025",
        "--telemetry-interval-s",
        "0.25",
        "--max-batch-size",
        "32",
        "--max-tokens-in-batch",
        "49152",
        "--num-cpu-blocks",
        "4096",
        "--gpu-mem-utilization",
        "0.99",
        "--timeout-s",
        "7200",
    ]
    if row["condition"] == "awq_w4_16":
        command.extend(
            [
                "--quantization-backend",
                "awq_marlin",
                "--quantized-model-path",
                str(args.quantized_model_path),
            ]
        )
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--quantized-model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "5":
        raise RuntimeError("the frozen crossover protocol requires CUDA_VISIBLE_DEVICES=5")
    plan = read_json(args.plan)
    runs = plan.get("runs", [])
    if len(runs) != 36:
        raise ValueError(f"expected the frozen 36-run plan, got {len(runs)} rows")
    if len({row["run_id"] for row in runs}) != len(runs):
        raise ValueError("run IDs are not unique")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {
        "schema_version": 1,
        "plan": str(args.plan),
        "manifest": str(args.manifest),
        "runs": [],
    }
    if args.status.exists() and args.resume:
        status = read_json(args.status)

    for row in runs:
        run_dir = args.output_dir / row["run_id"]
        if run_dir.exists():
            if args.resume and valid_completed_run(run_dir):
                print(f"skip completed {row['run_id']}", flush=True)
                skipped = status.setdefault("preexisting_valid_runs", [])
                if row["run_id"] not in skipped:
                    skipped.append(row["run_id"])
                    args.status.write_text(
                        json.dumps(status, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                continue
            raise FileExistsError(f"refusing to overwrite or silently replace {run_dir}")
        command = command_for(row, args)
        log_path = args.log_dir / f"{row['run_id']}.log"
        print(f"run {row['ordinal'] + 1}/36: {row['run_id']}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        entry = {
            "ordinal": row["ordinal"],
            "run_id": row["run_id"],
            "returncode": completed.returncode,
            "command": command,
            "log": str(log_path),
            "valid_completed_run": valid_completed_run(run_dir) if run_dir.exists() else False,
        }
        status.setdefault("runs", []).append(entry)
        args.status.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if completed.returncode or not entry["valid_completed_run"]:
            raise RuntimeError(f"run failed validation: {row['run_id']} (see {log_path})")

    missing = [row["run_id"] for row in runs if not valid_completed_run(args.output_dir / row["run_id"])]
    status["complete"] = not missing
    status["missing_or_invalid"] = missing
    args.status.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if missing:
        raise RuntimeError(f"missing or invalid frozen runs: {missing}")
    print("crossover run plan complete: 36/36 valid", flush=True)


if __name__ == "__main__":
    main()
