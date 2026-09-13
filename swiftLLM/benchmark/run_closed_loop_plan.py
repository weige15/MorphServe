"""Execute or resume the immutable 18-run v9 plan serially."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REQUIRED_FILES = (
    "metadata.json",
    "requests.jsonl",
    "telemetry.jsonl",
    "batches.jsonl",
    "controller.jsonl",
    "transitions.jsonl",
    "initialization_transitions.jsonl",
    "runtime_preparation.json",
    "summary.json",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def valid_completed_run(run_dir: Path, plan_row: dict[str, Any]) -> bool:
    if not all((run_dir / name).is_file() for name in REQUIRED_FILES):
        return False
    metadata = read_json(run_dir / "metadata.json")
    summary = read_json(run_dir / "summary.json")
    expected = int(plan_row["request_count"])
    return bool(
        metadata.get("run_id") == run_dir.name
        and metadata.get("planned_run_id") == plan_row["run_id"]
        and metadata.get("condition") == plan_row["condition"]
        and metadata.get("workload_class") == plan_row["workload_class"]
        and metadata.get("workload_sha256") == plan_row["workload_sha256"]
        and metadata.get("completed_request_count") == expected
        and metadata.get("generated_output_token_count") == expected * 512
        and summary.get("completed_request_count") == expected
        and summary.get("failed_or_incomplete_request_count") == 0
        and line_count(run_dir / "requests.jsonl") == expected
        and line_count(run_dir / "telemetry.jsonl") > 0
        and line_count(run_dir / "batches.jsonl") > 0
        and not (run_dir / "failure.json").exists()
    )


def command_for(
    row: dict[str, Any], args: argparse.Namespace, actual_run_id: str, attempt_index: int
) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "benchmark.run_closed_loop_condition",
        "--model-path",
        str(args.model_path),
        "--quantized-model-path",
        str(args.quantized_model_path),
        "--workload",
        str(row["workload"]),
        "--manifest",
        str(args.manifest),
        "--output-dir",
        str(args.output_dir),
        "--run-id",
        actual_run_id,
        "--planned-run-id",
        str(row["run_id"]),
        "--attempt-index",
        str(attempt_index),
        "--condition",
        str(row["condition"]),
        "--seed",
        "2025",
        "--telemetry-interval-s",
        "0.25",
        "--warmup-requests",
        "1",
        "--warmup-output-token-count",
        "8",
        "--timeout-s",
        "7200",
    ]


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
        raise RuntimeError("the frozen v9 plan requires CUDA_VISIBLE_DEVICES=5")

    plan = read_json(args.plan)
    rows = list(plan.get("runs", []))
    if len(rows) != 18 or len({row["run_id"] for row in rows}) != 18:
        raise ValueError("the frozen plan must contain 18 unique runs")
    expected_cells = {
        (workload, condition, repeat)
        for workload in ("low", "high", "low_high_low")
        for condition in (
            "runtime_static_fp16",
            "runtime_static_awq_w4_16",
            "closed_loop_dynamic",
        )
        for repeat in (0, 1)
    }
    if {(row["workload_class"], row["condition"], row["repeat"]) for row in rows} != expected_cells:
        raise ValueError("run plan does not cover the frozen 3x3x2 matrix")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {
        "schema_version": 1,
        "plan": str(args.plan),
        "manifest": str(args.manifest),
        "attempts": [],
        "selected_runs": {},
    }
    if args.resume and args.status.exists():
        status = read_json(args.status)

    for row in rows:
        planned_id = str(row["run_id"])
        selected_id = status.setdefault("selected_runs", {}).get(planned_id)
        if selected_id and valid_completed_run(args.output_dir / selected_id, row):
            print(f"skip completed {planned_id} -> {selected_id}", flush=True)
            continue
        if selected_id:
            raise RuntimeError(f"selected attempt is no longer valid: {planned_id} -> {selected_id}")

        attempt_index = 0
        while True:
            actual_run_id = planned_id if attempt_index == 0 else f"{planned_id}-attempt{attempt_index}"
            run_dir = args.output_dir / actual_run_id
            if not run_dir.exists():
                break
            if valid_completed_run(run_dir, row):
                status["selected_runs"][planned_id] = actual_run_id
                args.status.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
                print(f"adopt completed {planned_id} -> {actual_run_id}", flush=True)
                break
            if not args.resume:
                raise FileExistsError(f"refusing to overwrite failed/incomplete attempt {run_dir}")
            attempt_index += 1
        if status["selected_runs"].get(planned_id):
            continue

        command = command_for(row, args, actual_run_id, attempt_index)
        log_path = args.log_dir / f"{actual_run_id}.log"
        print(f"run {row['ordinal'] + 1}/18: {planned_id} attempt {attempt_index}", flush=True)
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
            "planned_run_id": planned_id,
            "actual_run_id": actual_run_id,
            "attempt_index": attempt_index,
            "returncode": completed.returncode,
            "command": command,
            "log": str(log_path),
            "valid_completed_run": valid_completed_run(run_dir, row) if run_dir.exists() else False,
        }
        status.setdefault("attempts", []).append(entry)
        if entry["valid_completed_run"]:
            status["selected_runs"][planned_id] = actual_run_id
        args.status.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        if completed.returncode or not entry["valid_completed_run"]:
            raise RuntimeError(
                f"run attempt failed and was preserved: {planned_id} -> {actual_run_id} "
                f"(resume creates attempt {attempt_index + 1}; see {log_path})"
            )

    missing = [
        row["run_id"] for row in rows
        if not status.get("selected_runs", {}).get(row["run_id"])
        or not valid_completed_run(
            args.output_dir / status["selected_runs"][row["run_id"]], row
        )
    ]
    status["complete"] = not missing
    status["missing_or_invalid"] = missing
    args.status.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    if missing:
        raise RuntimeError(f"missing or invalid frozen runs: {missing}")
    print("v9 run plan complete: 18/18 valid", flush=True)


if __name__ == "__main__":
    main()
