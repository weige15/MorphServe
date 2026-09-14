"""Execute or resume one immutable paired v11 run plan serially."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from .run_throughput_confirmation_condition import (
    ARTIFACT_VERSION,
    sha256_file,
    source_provenance,
)


REQUIRED_FILES = (
    "metadata.json",
    "requests.jsonl",
    "telemetry.jsonl",
    "batches.jsonl",
    "controller.jsonl",
    "gpu_environment.jsonl",
    "event_loop_lag.jsonl",
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
        metadata.get("experiment") == ARTIFACT_VERSION
        and metadata.get("run_id") == run_dir.name
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
        and line_count(run_dir / "event_loop_lag.jsonl") > 0
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
        "benchmark.run_throughput_confirmation_condition",
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


def validate_plan(rows: list[dict[str, Any]], expected_pairs: int) -> None:
    if len(rows) != 2 * expected_pairs or len({row["run_id"] for row in rows}) != len(rows):
        raise ValueError("plan does not contain the frozen number of unique runs")
    if [int(row["ordinal"]) for row in rows] != list(range(len(rows))):
        raise ValueError("plan ordinals are not contiguous")
    pairs: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["condition"] not in {"runtime_static_fp16", "closed_loop_dynamic"}:
            raise ValueError("v11 confirmatory plans contain only FP16 and Dynamic")
        pairs.setdefault(str(row["pair_id"]), []).append(row)
    if len(pairs) != expected_pairs:
        raise ValueError("pair count differs from the preregistered plan")
    for pair_id, pair in pairs.items():
        if len(pair) != 2 or {row["condition"] for row in pair} != {
            "runtime_static_fp16",
            "closed_loop_dynamic",
        }:
            raise ValueError(f"invalid matched pair {pair_id}")
        if len({row["workload_sha256"] for row in pair}) != 1:
            raise ValueError(f"workload mismatch within pair {pair_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-key", choices=("phase_a", "phase_c"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--quantized-model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "5":
        raise RuntimeError("the v11 plan requires CUDA_VISIBLE_DEVICES=5")

    repo = Path(__file__).resolve().parents[2]
    manifest = read_json(args.manifest)
    if manifest.get("artifact_version") != ARTIFACT_VERSION:
        raise RuntimeError("wrong throughput-confirmation manifest")
    plan_record = manifest["run_plans"].get(args.plan_key)
    if not plan_record or sha256_file(args.plan) != plan_record["sha256"]:
        raise RuntimeError("run plan differs from the preregistered manifest")
    protocol = repo / manifest["repository"]["protocol_path"]
    if sha256_file(protocol) != manifest["repository"]["protocol_sha256"]:
        raise RuntimeError("protocol differs from the preregistered manifest")
    if source_provenance() != manifest["source_sha256_at_preregistration"]:
        raise RuntimeError("runtime/source hashes differ from the preregistered manifest")
    git_status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True
    )
    if git_status.strip():
        raise RuntimeError(f"v11 runs require a clean Git tree:\n{git_status}")

    plan = read_json(args.plan)
    rows = list(plan.get("runs", []))
    validate_plan(rows, int(plan_record["pair_count"]))
    for row in rows:
        workload = repo / row["workload"]
        if sha256_file(workload) != row["workload_sha256"]:
            raise RuntimeError(f"workload hash mismatch: {workload}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {
        "schema_version": 1,
        "plan": str(args.plan),
        "plan_key": args.plan_key,
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
        print(
            f"run {int(row['ordinal']) + 1}/{len(rows)}: {planned_id} attempt {attempt_index}",
            flush=True,
        )
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, text=True, check=False
            )
        entry = {
            "ordinal": row["ordinal"],
            "pair_id": row["pair_id"],
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
                f"run attempt failed and was preserved: {planned_id} -> {actual_run_id}; "
                "resume creates the next attempt"
            )

    missing = [
        row["run_id"]
        for row in rows
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
    print(f"v11 {args.plan_key} run plan complete: {len(rows)}/{len(rows)} valid", flush=True)


if __name__ == "__main__":
    main()
