"""Audit exact v10 policy parity on every archived v7/v9/v10 telemetry stream."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
from typing import Any

from .instrumented_release_intent_controller import InstrumentedReleaseIntentController
from .release_intent_controller import ReleaseIntentController


TIMING_KEYS = {
    "controller_evaluate_wall_ns",
    "controller_evaluate_cpu_ns",
    "controller_window_wall_ns",
    "controller_window_cpu_ns",
    "controller_decision_bookkeeping_wall_ns",
    "controller_decision_bookkeeping_cpu_ns",
    "controller_policy_history_length",
    "controller_release_history_length",
    "controller_policy_segment_count",
    "controller_release_segment_count",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result.setdefault("scheduler_used_kv_blocks", result["num_decoding_gpu_blocks"])
    result.setdefault("physical_used_kv_blocks", result["scheduler_used_kv_blocks"])
    result.setdefault("preemption_count", result.get("swap_out_count", 0))
    return result


def fake_trace(timestamp_ns: int) -> dict[str, Any]:
    return {
        "status": "success",
        "started_ns": timestamp_ns,
        "ended_ns": timestamp_ns,
        "elapsed_ns": 0,
    }


def compare_decisions(
    original: dict[str, Any], instrumented: dict[str, Any], context: str
) -> None:
    projected = {key: value for key, value in instrumented.items() if key not in TIMING_KEYS}
    if original != projected:
        differing = {
            key: {"original": original.get(key), "instrumented": projected.get(key)}
            for key in set(original) | set(projected)
            if original.get(key) != projected.get(key)
        }
        raise AssertionError(f"policy mismatch at {context}: {differing}")


def replay_lockstep(path: Path) -> dict[str, Any]:
    original = ReleaseIntentController(0)
    instrumented = InstrumentedReleaseIntentController(0)
    actions: list[dict[str, Any]] = []
    rows = read_jsonl(path)
    for index, raw in enumerate(rows):
        row = normalize(raw)
        original_decision = original.evaluate(row)
        instrumented_decision = instrumented.evaluate(row)
        compare_decisions(original_decision, instrumented_decision, f"{path}:{index}")
        action = original_decision["requested_action"]
        if action is not None:
            timestamp_ns = int(round(float(row["elapsed_s"]) * 1e9))
            original_id = original.request_transition(action, timestamp_ns)
            instrumented_id = instrumented.request_transition(action, timestamp_ns)
            original.complete_transition(original_id, fake_trace(timestamp_ns), timestamp_ns)
            instrumented.complete_transition(instrumented_id, fake_trace(timestamp_ns), timestamp_ns)
            actions.append(
                {
                    "sample_index": int(row.get("sample_index", index)),
                    "scheduled_elapsed_s": row.get("scheduled_elapsed_s"),
                    "elapsed_s": float(row["elapsed_s"]),
                    "action": action,
                }
            )
    return {"sample_count": len(rows), "actions": actions}


def replay_v10_actual(run_dir: Path) -> dict[str, Any]:
    archived = read_jsonl(run_dir / "controller.jsonl")
    transitions = read_jsonl(run_dir / "transitions.jsonl")
    original = ReleaseIntentController(int(read_json(run_dir / "metadata.json")["measurement_start_time_ns"]))
    instrumented = InstrumentedReleaseIntentController(original.measurement_start_ns)
    pending: list[tuple[dict[str, Any], int, int]] = []
    actions: list[dict[str, Any]] = []

    for index, archived_row in enumerate(archived):
        timestamp_ns = int(archived_row["timestamp_ns"])
        remaining: list[tuple[dict[str, Any], int, int]] = []
        for event, original_id, instrumented_id in pending:
            ended_ns = int(event["api_task_ended_ns"])
            if ended_ns <= timestamp_ns:
                original.complete_transition(original_id, event["engine_trace"], ended_ns)
                instrumented.complete_transition(instrumented_id, event["engine_trace"], ended_ns)
            else:
                remaining.append((event, original_id, instrumented_id))
        pending = remaining

        row = normalize(archived_row)
        original_decision = original.evaluate(row)
        instrumented_decision = instrumented.evaluate(row)
        compare_decisions(original_decision, instrumented_decision, f"{run_dir}:{index}")
        for key, value in original_decision.items():
            if archived_row.get(key) != value:
                raise AssertionError(
                    f"archived v10 decision mismatch at {run_dir}:{index}:{key}: "
                    f"{archived_row.get(key)!r} != {value!r}"
                )
        action = original_decision["requested_action"]
        if action is not None:
            event = next(
                event
                for event in transitions
                if event["direction"] == action
                and int(event["requested_ns"]) >= timestamp_ns
                and not any(event is item[0] for item in pending)
                and int(event["controller_transition_id"]) > len(actions)
            )
            original_id = original.request_transition(action, int(event["requested_ns"]))
            instrumented_id = instrumented.request_transition(action, int(event["requested_ns"]))
            pending.append((event, original_id, instrumented_id))
            actions.append(
                {
                    "sample_index": int(archived_row["sample_index"]),
                    "scheduled_elapsed_s": archived_row["scheduled_elapsed_s"],
                    "elapsed_s": archived_row["elapsed_s"],
                    "action": action,
                }
            )
    return {"sample_count": len(archived), "actions": actions}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()

    v7 = [Path(path) for path in sorted(glob.glob(str(repo / "benchmark-results/fp16-awq-crossover-v7/runs/*/telemetry.jsonl")))]
    v9 = [
        Path(path)
        for path in sorted(glob.glob(str(repo / "benchmark-results/closed-loop-runtime-v9/runs/*closed_loop_dynamic*/telemetry.jsonl")))
    ]
    v10_dirs = [
        path.parent
        for path in sorted((repo / "benchmark-results/release-side-runtime-v10/runs").glob("*closed_loop_dynamic*/controller.jsonl"))
    ]
    if len(v7) != 36 or len(v9) != 6 or len(v10_dirs) != 6:
        raise RuntimeError(
            f"unexpected archived coverage: v7={len(v7)}, v9={len(v9)}, v10={len(v10_dirs)}"
        )

    records: list[dict[str, Any]] = []
    for version, paths in (("v7", v7), ("v9", v9)):
        for path in paths:
            result = replay_lockstep(path)
            records.append(
                {
                    "archive": version,
                    "path": str(path.relative_to(repo)),
                    "sha256": sha256_file(path),
                    **result,
                }
            )
    for run_dir in v10_dirs:
        result = replay_v10_actual(run_dir)
        path = run_dir / "controller.jsonl"
        records.append(
            {
                "archive": "v10",
                "path": str(path.relative_to(repo)),
                "sha256": sha256_file(path),
                **result,
            }
        )

    output = {
        "schema_version": 1,
        "status": "PASS",
        "comparison": (
            "instrumented evaluator equals current v10 evaluator for every decision field; "
            "v10 additionally equals every archived decision row"
        ),
        "action_timestamp_definition": "archived sample_index and frozen scheduled_elapsed_s grid",
        "coverage": {
            "v7_streams": len(v7),
            "v9_dynamic_streams": len(v9),
            "v10_dynamic_streams": len(v10_dirs),
            "total_streams": len(records),
            "total_samples": sum(record["sample_count"] for record in records),
            "total_actions": sum(len(record["actions"]) for record in records),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output["coverage"], indent=2, sort_keys=True))
    print("status: PASS")


if __name__ == "__main__":
    main()
