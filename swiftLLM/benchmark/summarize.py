"""Regenerate a Phase-1 summary from saved JSONL raw files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any

from .metrics import derive_request_metrics, distribution


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def _finite_numbers(rows: list[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(field)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _memory_stats(telemetry: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in ("gpu_memory_used_bytes", "gpu_memory_free_bytes", "gpu_memory_total_bytes"):
        values = _finite_numbers(telemetry, field)
        result[field] = {
            "count": len(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": statistics.fmean(values) if values else None,
        }
    return result


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    """Read only metadata and raw JSONL files and return a summary."""
    run_path = Path(run_dir)
    metadata = json.loads((run_path / "metadata.json").read_text(encoding="utf-8"))
    requests = _read_jsonl(run_path / "requests.jsonl")
    telemetry = _read_jsonl(run_path / "telemetry.jsonl")

    completed = [row for row in requests if row.get("status") == "completed"]
    derived = [derive_request_metrics(row) for row in completed]

    def metric_values(name: str) -> list[float]:
        return [float(item[name]) for item in derived if item[name] is not None]

    measurement_duration = metadata.get("measurement_duration_s")
    if not isinstance(measurement_duration, (int, float)) or measurement_duration <= 0:
        start = metadata.get("measurement_start_time_ns")
        end = metadata.get("measurement_end_time_ns")
        measurement_duration = ((int(end) - int(start)) / 1_000_000_000) if start and end else None

    actual_arrivals = [
        int(row["arrival_time_ns"])
        for row in requests
        if row.get("arrival_time_ns") is not None
    ]
    actual_span_s = None
    if len(actual_arrivals) > 1:
        actual_span_s = (max(actual_arrivals) - min(actual_arrivals)) / 1_000_000_000

    schedule_window = metadata.get("schedule_window_duration_s")
    offered_rate = None
    if isinstance(schedule_window, (int, float)) and schedule_window > 0:
        offered_rate = len(requests) / schedule_window
    actual_arrival_rate = None
    if actual_span_s and actual_span_s > 0 and len(actual_arrivals) > 1:
        # Inter-arrival rate is based on the N-1 observed gaps, not an
        # endpoint-inclusive request count.
        actual_arrival_rate = (len(actual_arrivals) - 1) / actual_span_s

    output_tokens = sum(int(row.get("output_token_count", 0) or 0) for row in completed)
    completed_count = len(completed)
    completed_throughput = (
        completed_count / float(measurement_duration)
        if measurement_duration and measurement_duration > 0
        else None
    )
    token_throughput = (
        output_tokens / float(measurement_duration)
        if measurement_duration and measurement_duration > 0
        else None
    )

    telemetry_fields = (
        "waiting_q_depth",
        "running_q_count",
        "swapped_q_count",
        "num_decoding_gpu_blocks",
        "num_gpu_blocks",
        "logical_kv_utilization",
    )
    telemetry_stats: dict[str, dict[str, Any]] = {}
    for field in telemetry_fields:
        values = _finite_numbers(telemetry, field)
        stats = distribution(values)
        stats["min"] = min(values) if values else None
        stats["max"] = max(values) if values else None
        telemetry_stats[field] = stats

    peak = {
        "waiting_q_depth": telemetry_stats["waiting_q_depth"]["max"],
        "running_q_count": telemetry_stats["running_q_count"]["max"],
        "swapped_q_count": telemetry_stats["swapped_q_count"]["max"],
        "num_decoding_gpu_blocks": telemetry_stats["num_decoding_gpu_blocks"]["max"],
        "logical_kv_utilization": telemetry_stats["logical_kv_utilization"]["max"],
    }
    swap_in = max((int(row.get("swap_in_count", 0)) for row in telemetry), default=0)
    swap_out = max((int(row.get("swap_out_count", 0)) for row in telemetry), default=0)

    ttft = distribution(metric_values("ttft_s"))
    queueing = distribution(metric_values("queueing_delay_s"))
    tpot = distribution(metric_values("tpot_s"))
    engine_ttft = distribution(metric_values("engine_ttft_s"))
    engine_tpot = distribution(metric_values("engine_tpot_s"))

    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_id": metadata.get("run_id"),
        "request_count": len(requests),
        "completed_request_count": completed_count,
        "failed_or_incomplete_request_count": len(requests) - completed_count,
        "target_rps": metadata.get("target_rps"),
        "achieved_offered_arrival_rate_rps": offered_rate,
        "achieved_arrival_rate_rps": actual_arrival_rate,
        "actual_interarrival_rate_rps": actual_arrival_rate,
        "completed_request_throughput_rps": completed_throughput,
        "generated_output_token_count": output_tokens,
        "generated_token_throughput_tps": token_throughput,
        "token_throughput_tps": token_throughput,
        "measurement_duration_s": measurement_duration,
        "ttft_s": ttft,
        "queueing_delay_s": queueing,
        "tpot_s": tpot,
        "engine_ttft_s": engine_ttft,
        "engine_tpot_s": engine_tpot,
        "peak_waiting_queue_depth": peak["waiting_q_depth"],
        "peak_running_count": peak["running_q_count"],
        "peak_swapped_queue_depth": peak["swapped_q_count"],
        "peak_num_decoding_gpu_blocks": peak["num_decoding_gpu_blocks"],
        "peak_logical_kv_utilization": peak["logical_kv_utilization"],
        "telemetry_stats": telemetry_stats,
        "telemetry_sample_count": len(telemetry),
        "swap_in_count": swap_in,
        "swap_out_count": swap_out,
        "preemption_count": swap_out,
        "gpu_memory": _memory_stats(telemetry),
        "num_gpu_blocks_observed": sorted({row.get("num_gpu_blocks") for row in telemetry if row.get("num_gpu_blocks") is not None}),
        "source_files": ["metadata.json", "requests.jsonl", "telemetry.jsonl"],
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, help="Summary path (default: RUN_DIR/summary.json)")
    args = parser.parse_args()
    summary = summarize_run(args.run_dir)
    output = args.output or args.run_dir / "summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
