"""Freeze the three v9 workloads and counterbalanced 18-run plan."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any


CORE_FIELDS = (
    "source_sequence",
    "source_trace_timestamp_s",
    "dataset_question_id",
    "dataset_reference_answers",
    "prompt",
    "prompt_token_count",
    "prompt_token_ids_sha256",
    "requested_output_token_count",
    "planned_arrival_offset_s",
)
CONDITIONS = (
    "runtime_static_fp16",
    "runtime_static_awq_w4_16",
    "closed_loop_dynamic",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def core_digest(rows: list[dict[str, Any]], *, include_offsets: bool = True) -> str:
    fields = CORE_FIELDS if include_offsets else CORE_FIELDS[:-1]
    payload = [{field: row.get(field) for field in fields} for row in rows]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def phase_span(rows: list[dict[str, Any]]) -> float:
    offsets = sorted({float(row["planned_arrival_offset_s"]) for row in rows})
    if len(offsets) < 2 or offsets[0] != 0.0:
        raise ValueError("phase template must start at zero and contain a positive trace interval")
    return offsets[-1] + offsets[1]


def tagged(rows: list[dict[str, Any]], workload_class: str, phase: str) -> list[dict[str, Any]]:
    result = deepcopy(rows)
    for sequence, row in enumerate(result):
        row["sequence"] = sequence
        row["v9_workload_class"] = workload_class
        row["v9_phase"] = phase
    return result


def make_phased(low: list[dict[str, Any]], high: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    low_span = phase_span(low)
    high_span = phase_span(high)
    specs = (
        ("low_initial", low, 0.0),
        ("high", high, low_span),
        ("low_recovery", low, low_span + high_span),
    )
    output: list[dict[str, Any]] = []
    boundaries: dict[str, Any] = {}
    for phase, source, shift in specs:
        boundaries[phase] = {
            "start_offset_s": shift,
            "last_arrival_offset_s": shift + float(source[-1]["planned_arrival_offset_s"]),
            "request_count": len(source),
        }
        for source_row in source:
            row = deepcopy(source_row)
            row["sequence"] = len(output)
            row["v9_workload_class"] = "low_high_low"
            row["v9_phase"] = phase
            row["v9_original_request_id"] = source_row["request_id"]
            row["request_id"] = f"v9-{phase}-{source_row['request_id']}"
            row["planned_arrival_offset_s"] = shift + float(source_row["planned_arrival_offset_s"])
            output.append(row)
    boundaries["natural_phase_spans_s"] = {
        "scale_5p75": low_span,
        "scale_4p25": high_span,
        "definition": "last trace offset plus its existing first positive inter-cohort interval",
    }
    return output, boundaries


def build_plan(paths: dict[str, Path]) -> list[dict[str, Any]]:
    order = (
        (0, "low", "runtime_static_fp16"),
        (0, "low", "closed_loop_dynamic"),
        (0, "low", "runtime_static_awq_w4_16"),
        (0, "high", "runtime_static_awq_w4_16"),
        (0, "high", "closed_loop_dynamic"),
        (0, "high", "runtime_static_fp16"),
        (0, "low_high_low", "runtime_static_fp16"),
        (0, "low_high_low", "runtime_static_awq_w4_16"),
        (0, "low_high_low", "closed_loop_dynamic"),
        (1, "low_high_low", "closed_loop_dynamic"),
        (1, "low_high_low", "runtime_static_awq_w4_16"),
        (1, "low_high_low", "runtime_static_fp16"),
        (1, "high", "runtime_static_fp16"),
        (1, "high", "closed_loop_dynamic"),
        (1, "high", "runtime_static_awq_w4_16"),
        (1, "low", "runtime_static_awq_w4_16"),
        (1, "low", "closed_loop_dynamic"),
        (1, "low", "runtime_static_fp16"),
    )
    rows = []
    for ordinal, (repeat, workload_class, condition) in enumerate(order):
        path = paths[workload_class]
        rows.append(
            {
                "ordinal": ordinal,
                "repeat": repeat,
                "workload_class": workload_class,
                "condition": condition,
                "run_id": f"v9-{workload_class}-{condition}-rep{repeat}",
                "workload": str(path),
                "workload_sha256": sha256_file(path),
                "request_count": 192 if workload_class == "low_high_low" else 64,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    low_source_path = args.v7_input_dir / "latency-scale-5p75.jsonl"
    high_source_path = args.v7_input_dir / "latency-scale-4p25.jsonl"
    low_source = read_jsonl(low_source_path)
    high_source = read_jsonl(high_source_path)
    if len(low_source) != 64 or len(high_source) != 64:
        raise ValueError("v7 low/high templates must each contain exactly 64 requests")
    if core_digest(low_source, include_offsets=False) != core_digest(high_source, include_offsets=False):
        raise RuntimeError("v7 low/high request contents are not identical")

    low = tagged(low_source, "low", "low")
    high = tagged(high_source, "high", "high")
    phased, boundaries = make_phased(low_source, high_source)
    paths = {
        "low": args.output_dir / "low-scale-5p75.jsonl",
        "high": args.output_dir / "high-scale-4p25.jsonl",
        "low_high_low": args.output_dir / "low-high-low.jsonl",
    }
    write_jsonl(paths["low"], low)
    write_jsonl(paths["high"], high)
    write_jsonl(paths["low_high_low"], phased)

    if core_digest(low) != core_digest(low_source) or core_digest(high) != core_digest(high_source):
        raise RuntimeError("tagging changed a frozen v7 request or arrival")
    for phase, source in (("low_initial", low_source), ("high", high_source), ("low_recovery", low_source)):
        rows = [row for row in phased if row["v9_phase"] == phase]
        if core_digest(rows, include_offsets=False) != core_digest(source, include_offsets=False):
            raise RuntimeError(f"phased workload changed request contents in {phase}")

    metadata = {
        "schema_version": 1,
        "status": "frozen_before_closed_loop_measurements",
        "source": {
            "low_scale": 5.75,
            "low_path": str(low_source_path),
            "low_sha256": sha256_file(low_source_path),
            "high_scale": 4.25,
            "high_path": str(high_source_path),
            "high_sha256": sha256_file(high_source_path),
            "common_64_request_content_sha256": core_digest(low_source, include_offsets=False),
        },
        "outputs": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "request_count": len(read_jsonl(path)),
                "last_arrival_offset_s": read_jsonl(path)[-1]["planned_arrival_offset_s"],
            }
            for name, path in paths.items()
        },
        "phased_boundaries": boundaries,
        "phased_content_independence": (
            "The same 64 questions are reused in three phases; quality inference must collapse "
            "duplicates by original question rather than count 192 independent samples."
        ),
        "no_artificial_idle_gap": True,
    }
    (args.output_dir / "workload_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plan_rows = build_plan(paths)
    plan = {
        "schema_version": 1,
        "status": "frozen_before_closed_loop_measurements",
        "order_policy": "two fresh-process counterbalanced passes over three workloads and three runtime-envelope conditions",
        "run_count": len(plan_rows),
        "runs": plan_rows,
    }
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    args.plan.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"workloads": metadata, "plan": plan}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
