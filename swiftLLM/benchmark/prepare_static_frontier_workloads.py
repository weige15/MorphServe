"""Derive v5 quality, calibration, and serving workloads from the frozen 106 rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scale_label(value: float) -> str:
    return format(value, "g").replace(".", "p")


def derive_rows(
    source_rows: list[dict[str, Any]],
    *,
    start: int,
    count: int,
    time_scale: float,
    kind: str,
) -> list[dict[str, Any]]:
    selected = source_rows[start : start + count]
    if len(selected) != count:
        raise ValueError(f"requested rows [{start}:{start + count}], but source has {len(source_rows)}")
    if time_scale <= 0:
        raise ValueError("time scale must be positive")
    origin = float(selected[0]["planned_arrival_offset_s"])
    scaled_span = (
        (float(selected[-1]["planned_arrival_offset_s"]) - origin) * time_scale
    )
    nominal_rps = (count - 1) / scaled_span if count > 1 and scaled_span > 0 else None
    rows: list[dict[str, Any]] = []
    for sequence, source in enumerate(selected):
        row = dict(source)
        row.update(
            {
                "sequence": sequence,
                "source_sequence": source["sequence"],
                "planned_arrival_offset_s": (
                    float(source["planned_arrival_offset_s"]) - origin
                )
                * time_scale,
                "frontier_workload_kind": kind,
                "frontier_time_scale": time_scale,
                "frontier_nominal_offered_rps": nominal_rps,
            }
        )
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-workload", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-scales", type=float, nargs="*", default=())
    parser.add_argument("--latency-scales", type=float, nargs="*", default=())
    parser.add_argument("--dense-start", type=int, default=42)
    parser.add_argument("--calibration-count", type=int, default=32)
    parser.add_argument("--latency-count", type=int, default=64)
    args = parser.parse_args()

    source_rows = read_jsonl(args.source_workload)
    if len(source_rows) != 106:
        raise ValueError(f"expected frozen 106-row source workload, got {len(source_rows)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[dict[str, Any]] = []
    quality = derive_rows(
        source_rows,
        start=0,
        count=len(source_rows),
        time_scale=1.0,
        kind="quality_sequential_106",
    )
    quality_path = args.output_dir / "quality-106.jsonl"
    write_jsonl(quality_path, quality)
    outputs.append(
        {
            "kind": "quality",
            "path": str(quality_path),
            "sha256": sha256_file(quality_path),
            "request_count": len(quality),
            "source_sequences": [row["source_sequence"] for row in quality],
        }
    )

    for family, scales, count in (
        ("calibration", args.calibration_scales, args.calibration_count),
        ("latency", args.latency_scales, args.latency_count),
    ):
        for scale in scales:
            rows = derive_rows(
                source_rows,
                start=args.dense_start,
                count=count,
                time_scale=scale,
                kind=f"{family}_dense_trace",
            )
            path = args.output_dir / f"{family}-scale-{scale_label(scale)}.jsonl"
            write_jsonl(path, rows)
            outputs.append(
                {
                    "kind": family,
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "request_count": len(rows),
                    "source_sequence_start": rows[0]["source_sequence"],
                    "source_sequence_end": rows[-1]["source_sequence"],
                    "source_timestamp_start_s": rows[0]["source_trace_timestamp_s"],
                    "source_timestamp_end_s": rows[-1]["source_trace_timestamp_s"],
                    "time_scale_relative_to_frozen_1p75_offsets": scale,
                    "nominal_offered_rps": rows[0]["frontier_nominal_offered_rps"],
                    "last_arrival_offset_s": rows[-1]["planned_arrival_offset_s"],
                }
            )

    metadata = {
        "schema_version": 1,
        "source_workload": str(args.source_workload),
        "source_workload_sha256": sha256_file(args.source_workload),
        "source_request_count": len(source_rows),
        "dense_selection_rule": (
            f"contiguous frozen source sequences {args.dense_start} onward; "
            "selected before v5 W4 results"
        ),
        "time_scale_definition": (
            "multiply the already-frozen 1.75x BurstGPT offsets after normalizing "
            "the selected first arrival to zero"
        ),
        "outputs": outputs,
    }
    (args.output_dir / "workload_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
