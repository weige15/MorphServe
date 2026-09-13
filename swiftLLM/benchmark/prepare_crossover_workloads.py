"""Create the fixed FP16/AWQ crossover load grid from the frozen v5 trace."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .prepare_static_frontier_workloads import derive_rows, read_jsonl, scale_label, write_jsonl


DEFAULT_SCALES = (6.0, 5.5, 5.0, 4.5, 4.0)
CONTENT_FIELDS = (
    "request_id",
    "source_sequence",
    "source_trace_timestamp_s",
    "dataset_question_id",
    "dataset_reference_answers",
    "prompt",
    "prompt_token_count",
    "requested_output_token_count",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_sha256(rows: list[dict[str, Any]]) -> str:
    content = [{key: row.get(key) for key in CONTENT_FIELDS} for row in rows]
    encoded = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-workload", type=Path, required=True)
    parser.add_argument("--reference-input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scales", type=float, nargs="+", default=DEFAULT_SCALES)
    parser.add_argument("--dense-start", type=int, default=42)
    parser.add_argument("--request-count", type=int, default=64)
    args = parser.parse_args()

    scales = tuple(args.scales)
    if scales != tuple(sorted(set(scales), reverse=True)):
        raise ValueError("scales must be unique and listed from low to high offered load")
    if scales[0] != 6.0 or scales[-1] != 4.0:
        raise ValueError("the fixed grid must retain scale-6 and scale-4 endpoints")

    source_rows = read_jsonl(args.source_workload)
    if len(source_rows) != 106:
        raise ValueError(f"expected the frozen 106-row source workload, got {len(source_rows)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[dict[str, Any]] = []
    expected_content_hash: str | None = None
    for scale in scales:
        rows = derive_rows(
            source_rows,
            start=args.dense_start,
            count=args.request_count,
            time_scale=scale,
            kind="latency_dense_trace",
        )
        row_content_hash = content_sha256(rows)
        if expected_content_hash is None:
            expected_content_hash = row_content_hash
        elif row_content_hash != expected_content_hash:
            raise RuntimeError("request contents changed across the crossover load grid")
        path = args.output_dir / f"latency-scale-{scale_label(scale)}.jsonl"
        write_jsonl(path, rows)
        output: dict[str, Any] = {
            "scale": scale,
            "nominal_offered_rps": rows[0]["frontier_nominal_offered_rps"],
            "last_arrival_offset_s": rows[-1]["planned_arrival_offset_s"],
            "path": str(path),
            "sha256": sha256_file(path),
            "request_content_sha256": row_content_hash,
            "request_count": len(rows),
            "source_sequence_start": rows[0]["source_sequence"],
            "source_sequence_end": rows[-1]["source_sequence"],
        }
        if scale in (6.0, 4.0):
            reference = args.reference_input_dir / f"latency-scale-{scale_label(scale)}.jsonl"
            output["reference_path"] = str(reference)
            output["reference_sha256"] = sha256_file(reference)
            output["byte_identical_to_v5_endpoint"] = path.read_bytes() == reference.read_bytes()
            if not output["byte_identical_to_v5_endpoint"]:
                raise RuntimeError(f"derived scale-{scale:g} endpoint differs from frozen v5 input")
        outputs.append(output)

    metadata = {
        "schema_version": 1,
        "status": "pre_registered_before_new_intermediate_serving_results",
        "source_workload": str(args.source_workload),
        "source_workload_sha256": sha256_file(args.source_workload),
        "reference_input_dir": str(args.reference_input_dir),
        "dense_selection": {
            "source_sequence_start": args.dense_start,
            "request_count": args.request_count,
        },
        "derivation": (
            "Use the existing v5 derive_rows rule: normalize source sequence 42 to zero "
            "and multiply every frozen 1.75x BurstGPT offset by the declared scale."
        ),
        "fixed_scales_low_to_high_load": list(scales),
        "request_content_sha256": expected_content_hash,
        "outputs": outputs,
    }
    metadata_path = args.output_dir / "workload_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
