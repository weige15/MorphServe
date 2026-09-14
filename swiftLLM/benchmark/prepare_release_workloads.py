"""Freeze held-out release-side v10 workloads and the 16-run plan."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any


CONDITIONS = (
    "runtime_static_fp16",
    "runtime_static_awq_w4_16",
    "closed_loop_dynamic",
)
CONTENT_FIELDS = (
    "dataset_question_id",
    "dataset_question_type",
    "dataset_reference_answers",
    "prompt",
    "prompt_token_count",
    "prompt_token_ids_sha256",
    "requested_output_token_count",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> str:
    payload = [{field: row.get(field) for field in fields} for row in rows]
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def phase_span(rows: list[dict[str, Any]]) -> float:
    offsets = sorted({float(row["planned_arrival_offset_s"]) for row in rows})
    if len(offsets) < 2 or offsets[0] != 0.0:
        raise ValueError("arrival template must begin at zero and have a positive cohort interval")
    return offsets[-1] + offsets[1]


def remap_content(
    arrival_rows: list[dict[str, Any]], content_rows: list[dict[str, Any]], regime: str
) -> list[dict[str, Any]]:
    if len(arrival_rows) != 64 or len(content_rows) != 64:
        raise ValueError("each held-out phase requires exactly 64 requests")
    output: list[dict[str, Any]] = []
    for sequence, (arrival, content) in enumerate(zip(arrival_rows, content_rows)):
        row = deepcopy(arrival)
        row.update({field: content[field] for field in CONTENT_FIELDS})
        row.update(
            {
                "sequence": sequence,
                "request_id": f"v10-{regime}-c{int(content['source_sequence']):04d}",
                "arrival_source_sequence": arrival["source_sequence"],
                "arrival_source_trace_timestamp_s": arrival["source_trace_timestamp_s"],
                "content_source_sequence": content["source_sequence"],
                "source_sequence": content["source_sequence"],
                "source_trace_timestamp_s": content["source_trace_timestamp_s"],
                "v10_workload_class": regime,
                "v10_phase": regime,
                "v10_phase_index": 0,
            }
        )
        output.append(row)
    return output


def make_alternating(
    templates: dict[str, list[dict[str, Any]]], phases: tuple[str, ...], workload: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output: list[dict[str, Any]] = []
    boundaries: list[dict[str, Any]] = []
    shift = 0.0
    for phase_index, regime in enumerate(phases):
        source = templates[regime]
        phase_name = f"{regime}_{phase_index}"
        boundaries.append(
            {
                "phase_index": phase_index,
                "phase": phase_name,
                "regime": regime,
                "start_offset_s": shift,
                "last_arrival_offset_s": shift
                + float(source[-1]["planned_arrival_offset_s"]),
                "request_count": len(source),
            }
        )
        for source_row in source:
            row = deepcopy(source_row)
            row["sequence"] = len(output)
            row["request_id"] = f"v10-{workload}-p{phase_index}-{source_row['request_id']}"
            row["planned_arrival_offset_s"] = shift + float(
                source_row["planned_arrival_offset_s"]
            )
            row["v10_workload_class"] = workload
            row["v10_phase"] = phase_name
            row["v10_phase_index"] = phase_index
            output.append(row)
        shift += phase_span(source)
    return output, boundaries


def build_plan(paths: dict[str, Path], counts: dict[str, int]) -> list[dict[str, Any]]:
    order = (
        (0, "low_only", "runtime_static_fp16"),
        (0, "low_only", "closed_loop_dynamic"),
        (0, "heldout_one_cycle", "runtime_static_fp16"),
        (0, "heldout_one_cycle", "runtime_static_awq_w4_16"),
        (0, "heldout_one_cycle", "closed_loop_dynamic"),
        (0, "heldout_two_cycle", "runtime_static_awq_w4_16"),
        (0, "heldout_two_cycle", "closed_loop_dynamic"),
        (0, "heldout_two_cycle", "runtime_static_fp16"),
        (1, "heldout_two_cycle", "runtime_static_fp16"),
        (1, "heldout_two_cycle", "closed_loop_dynamic"),
        (1, "heldout_two_cycle", "runtime_static_awq_w4_16"),
        (1, "heldout_one_cycle", "closed_loop_dynamic"),
        (1, "heldout_one_cycle", "runtime_static_awq_w4_16"),
        (1, "heldout_one_cycle", "runtime_static_fp16"),
        (1, "low_only", "closed_loop_dynamic"),
        (1, "low_only", "runtime_static_fp16"),
    )
    plan: list[dict[str, Any]] = []
    for ordinal, (repeat, workload, condition) in enumerate(order):
        path = paths[workload]
        plan.append(
            {
                "ordinal": ordinal,
                "repeat": repeat,
                "workload_class": workload,
                "condition": condition,
                "run_id": f"v10-{workload}-{condition}-rep{repeat}",
                "workload": str(path),
                "workload_sha256": sha256_file(path),
                "request_count": counts[workload],
            }
        )
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-workload", type=Path, required=True)
    parser.add_argument("--v7-input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = read_jsonl(args.source_workload)
    low_arrivals = read_jsonl(args.v7_input_dir / "latency-scale-5p75.jsonl")
    high_arrivals = read_jsonl(args.v7_input_dir / "latency-scale-4p25.jsonl")
    if len(source) != 106:
        raise ValueError(f"expected 106 frozen content rows, got {len(source)}")
    content = source[:64]
    if any(
        int(row["prompt_token_count"]) != 1024
        or int(row["requested_output_token_count"]) != 512
        for row in content
    ):
        raise ValueError("held-out content must preserve exact 1024/512 shape")

    templates = {
        "low": remap_content(low_arrivals, content, "low"),
        "high": remap_content(high_arrivals, content, "high"),
    }
    low_only = deepcopy(templates["low"])
    for row in low_only:
        row["v10_workload_class"] = "low_only"
    one_cycle, one_boundaries = make_alternating(
        templates, ("low", "high", "low"), "heldout_one_cycle"
    )
    two_cycle, two_boundaries = make_alternating(
        templates,
        ("low", "high", "low", "high", "low"),
        "heldout_two_cycle",
    )
    workloads = {
        "low_only": low_only,
        "heldout_one_cycle": one_cycle,
        "heldout_two_cycle": two_cycle,
    }
    paths = {
        name: args.output_dir / f"{name.replace('_', '-')}.jsonl"
        for name in workloads
    }
    for name, rows in workloads.items():
        if len({row["request_id"] for row in rows}) != len(rows):
            raise RuntimeError(f"duplicate request ID in {name}")
        if any(
            int(row["sequence"]) != index
            for index, row in enumerate(rows)
        ):
            raise RuntimeError(f"noncontiguous sequence in {name}")
        write_jsonl(paths[name], rows)

    content_hash = digest(content, CONTENT_FIELDS)
    metadata = {
        "schema_version": 1,
        "status": "frozen_before_any_v10_serving_result",
        "source_workload": str(args.source_workload),
        "source_workload_sha256": sha256_file(args.source_workload),
        "content_selection": {
            "source_sequence_start": 0,
            "source_sequence_end": 63,
            "request_count": 64,
            "content_sha256": content_hash,
            "new_questions_relative_to_v9_42_105": 42,
            "overlapping_questions_relative_to_v9_42_105": 22,
            "limitation": (
                "The frozen source has 106 questions, so no disjoint 64-row slice exists. "
                "Rows 0-63 maximize novelty while preserving deterministic content."
            ),
        },
        "arrival_templates": {
            "low": {
                "scale": 5.75,
                "path": str(args.v7_input_dir / "latency-scale-5p75.jsonl"),
                "sha256": sha256_file(args.v7_input_dir / "latency-scale-5p75.jsonl"),
                "span_s": phase_span(low_arrivals),
            },
            "high": {
                "scale": 4.25,
                "path": str(args.v7_input_dir / "latency-scale-4p25.jsonl"),
                "sha256": sha256_file(args.v7_input_dir / "latency-scale-4p25.jsonl"),
                "span_s": phase_span(high_arrivals),
            },
        },
        "derivation": (
            "Map frozen source content rows 0-63 positionally onto the validated v7 "
            "scale-5.75/4.25 arrival offsets; concatenate natural phase spans without gaps."
        ),
        "no_artificial_idle_gap": True,
        "phase_boundaries": {
            "low_only": [
                {
                    "phase_index": 0,
                    "phase": "low_0",
                    "regime": "low",
                    "start_offset_s": 0.0,
                    "last_arrival_offset_s": float(low_only[-1]["planned_arrival_offset_s"]),
                    "request_count": 64,
                }
            ],
            "heldout_one_cycle": one_boundaries,
            "heldout_two_cycle": two_boundaries,
        },
        "outputs": {
            name: {
                "path": str(paths[name]),
                "sha256": sha256_file(paths[name]),
                "request_count": len(rows),
                "last_arrival_offset_s": float(rows[-1]["planned_arrival_offset_s"]),
            }
            for name, rows in workloads.items()
        },
    }
    metadata_path = args.output_dir / "workload_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    plan_rows = build_plan(paths, {name: len(rows) for name, rows in workloads.items()})
    plan = {
        "schema_version": 1,
        "status": "frozen_before_any_v10_serving_result",
        "order_policy": "two counterbalanced fresh-process passes; AWQ only on alternating workloads",
        "run_count": len(plan_rows),
        "runs": plan_rows,
    }
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    args.plan.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"workloads": metadata, "plan": plan}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
