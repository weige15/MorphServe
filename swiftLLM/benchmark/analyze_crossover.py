"""Regenerate the pre-registered FP16/AWQ crossover characterization."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Iterable

import matplotlib.pyplot as plt

from .metrics import derive_request_metrics, distribution


CONDITIONS = ("fp16_0", "awq_w4_16")
EXPECTED_BLOCKS = {"fp16_0": 1768, "awq_w4_16": 4286}
REQUIRED_BATCH_FIELDS = {
    "timestamp_ns",
    "precision_state",
    "num_prefill_sequences",
    "total_prefill_tokens",
    "effective_gemm_m",
    "num_decoding_sequences",
    "prefill_execution_duration_ns",
    "waiting_queue_depth_before_scheduling",
    "running_request_count_before_scheduling",
    "logical_kv_utilization_before_scheduling",
    "safe_kv_blocks",
    "used_kv_blocks_before_scheduling",
    "swap_in_count",
    "swap_out_count",
    "preemption_count",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def numeric(rows: Iterable[dict[str, Any]], field: str) -> list[float]:
    return [float(row[field]) for row in rows if isinstance(row.get(field), (int, float))]


def flatten(prefix: str, values: list[float]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in distribution(values).items()}


def delta_s(row: dict[str, Any], end: str, start: str) -> float | None:
    if row.get(end) is None or row.get(start) is None:
        return None
    return (int(row[end]) - int(row[start])) / 1_000_000_000


def telemetry_segments(
    telemetry: list[dict[str, Any]], duration: float
) -> list[tuple[float, float, dict[str, Any]]]:
    rows = sorted(telemetry, key=lambda row: float(row["elapsed_s"]))
    segments = []
    for index, row in enumerate(rows):
        start = max(0.0, float(row["elapsed_s"]))
        end = min(
            duration,
            float(rows[index + 1]["elapsed_s"]) if index + 1 < len(rows) else duration,
        )
        if end > start:
            segments.append((start, end, row))
    return segments


def integrate(
    segments: list[tuple[float, float, dict[str, Any]]],
    value: Callable[[dict[str, Any]], float],
    *,
    start: float = 0.0,
    end: float | None = None,
) -> float:
    stop = math.inf if end is None else end
    total = 0.0
    for left, right, row in segments:
        overlap = max(0.0, min(right, stop) - max(left, start))
        if overlap:
            total += overlap * value(row)
    return total


def longest_duration(
    segments: list[tuple[float, float, dict[str, Any]]],
    predicate: Callable[[dict[str, Any]], bool],
) -> float:
    longest = current = 0.0
    for left, right, row in segments:
        if predicate(row):
            current += right - left
            longest = max(longest, current)
        else:
            current = 0.0
    return longest


def counter_value_at(
    telemetry: list[dict[str, Any]], field: str, timestamp: float
) -> int:
    value = 0
    for row in telemetry:
        if float(row["elapsed_s"]) > timestamp:
            break
        value = int(row.get(field, value))
    return value


def window_stats(
    segments: list[tuple[float, float, dict[str, Any]]],
    end: float,
    window: float,
) -> dict[str, float]:
    start = max(0.0, end - window)
    elapsed = end - start
    if elapsed <= 0:
        return {
            "waiting_mean": 0.0,
            "waiting_area": 0.0,
            "waiting_ge4_fraction": 0.0,
            "kv_ge095_fraction": 0.0,
            "fp16_equiv_le070_fraction": 0.0,
        }
    waiting_area = integrate(
        segments, lambda row: float(row["waiting_q_depth"]), start=start, end=end
    )
    return {
        "waiting_mean": waiting_area / elapsed,
        "waiting_area": waiting_area,
        "waiting_ge4_fraction": integrate(
            segments,
            lambda row: float(float(row["waiting_q_depth"]) >= 4),
            start=start,
            end=end,
        )
        / elapsed,
        "kv_ge095_fraction": integrate(
            segments,
            lambda row: float(float(row["logical_kv_utilization"]) >= 0.95),
            start=start,
            end=end,
        )
        / elapsed,
        "fp16_equiv_le070_fraction": integrate(
            segments,
            lambda row: float(float(row["num_decoding_gpu_blocks"]) / 1768 <= 0.70),
            start=start,
            end=end,
        )
        / elapsed,
    }


def release_time(
    telemetry: list[dict[str, Any]],
    duration: float,
    candidate: str,
    not_before: float,
) -> float | None:
    rows = sorted(telemetry, key=lambda row: float(row["elapsed_s"]))
    segments = telemetry_segments(rows, duration)
    for row in rows:
        now = float(row["elapsed_s"])
        if now < not_before + 10.0:
            continue
        stats = window_stats(segments, now, 10.0)
        recent_preemptions = int(row.get("preemption_count", 0)) - counter_value_at(
            rows, "preemption_count", now - 10.0
        )
        waiting_ok = (
            stats["waiting_mean"] < 0.5
            if candidate == "preemption_confirmed_pressure"
            else stats["waiting_mean"] <= 0.5
        )
        if (
            waiting_ok
            and recent_preemptions == 0
            and stats["fp16_equiv_le070_fraction"] >= 0.90
        ):
            return now
    return None


def episode_stats(
    statuses: list[tuple[float, bool]], duration: float
) -> tuple[int, float, float]:
    count = 0
    total = longest = current = 0.0
    active = False
    for index, (start, enabled) in enumerate(statuses):
        end = statuses[index + 1][0] if index + 1 < len(statuses) else duration
        interval = max(0.0, end - start)
        if enabled:
            if not active:
                count += 1
                current = 0.0
            active = True
            current += interval
            total += interval
            longest = max(longest, current)
        else:
            active = False
            current = 0.0
    return count, total, longest


def evaluate_signals(
    telemetry: list[dict[str, Any]], duration: float
) -> dict[str, dict[str, float | bool | None]]:
    rows = sorted(telemetry, key=lambda row: float(row["elapsed_s"]))
    segments = telemetry_segments(rows, duration)
    candidates = (
        "sustained_compound_pressure",
        "capacity_margin_queue_integral",
        "preemption_confirmed_pressure",
    )
    first_entry: dict[str, float | None] = {name: None for name in candidates}
    statuses: dict[str, list[tuple[float, bool]]] = {name: [] for name in candidates}
    free_low_since: float | None = None
    for row in rows:
        now = float(row["elapsed_s"])
        free_blocks = int(row["num_gpu_blocks"]) - int(row["num_decoding_gpu_blocks"])
        if free_blocks < 65:
            free_low_since = now if free_low_since is None else free_low_since
        else:
            free_low_since = None

        stats3 = window_stats(segments, now, 3.0)
        compound = (
            now >= 3
            and stats3["kv_ge095_fraction"] >= 0.80
            and stats3["waiting_ge4_fraction"] >= 0.50
            and float(row["waiting_q_depth"]) >= 4
        )
        stats5 = window_stats(segments, now, 5.0)
        capacity = (
            now >= 5
            and free_low_since is not None
            and now - free_low_since >= 1.0
            and stats5["waiting_area"] >= 15.0
        )
        recent_preemptions = int(row.get("preemption_count", 0)) - counter_value_at(
            rows, "preemption_count", now - 5.0
        )
        preemption = (
            now >= 5
            and recent_preemptions >= 2
            and window_stats(segments, now, 2.0)["waiting_mean"] >= 2.0
        )
        current = {
            "sustained_compound_pressure": compound,
            "capacity_margin_queue_integral": capacity,
            "preemption_confirmed_pressure": preemption,
        }
        for name, enabled in current.items():
            statuses[name].append((now, enabled))
            if enabled and first_entry[name] is None:
                first_entry[name] = now

    results: dict[str, dict[str, float | bool | None]] = {}
    for name in candidates:
        episode_count, total_active, longest_active = episode_stats(statuses[name], duration)
        results[name] = {
            "entry_fired": first_entry[name] is not None,
            "first_entry_s": first_entry[name],
            "entry_condition_episode_count": episode_count,
            "entry_condition_total_duration_s": total_active,
            "entry_condition_longest_duration_s": longest_active,
            "release_from_run_start_s": release_time(telemetry, duration, name, 0.0),
        }
    return results


def analyze_run(run_dir: Path, plan_row: dict[str, Any]) -> tuple[
    dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    metadata = read_json(run_dir / "metadata.json")
    requests = read_jsonl(run_dir / "requests.jsonl")
    telemetry = read_jsonl(run_dir / "telemetry.jsonl")
    batches = read_jsonl(run_dir / "batches.jsonl")
    summary = read_json(run_dir / "summary.json")
    condition = str(metadata["condition"])
    scale = float(metadata["workload_time_scale"])
    repeat_match = re.search(r"-rep(\d+)$", str(metadata["run_id"]))
    repeat = int(repeat_match.group(1)) if repeat_match else -1
    completed = [row for row in requests if row.get("status") == "completed"]
    duration = float(metadata["measurement_duration_s"])

    validation = {
        "run_id_matches_plan": metadata["run_id"] == plan_row["run_id"],
        "condition_matches_plan": condition == plan_row["condition"],
        "scale_matches_plan": scale == float(plan_row["scale"]),
        "repeat_matches_plan": repeat == int(plan_row["repeat"]),
        "workload_hash_matches_plan": metadata["workload_sha256"] == plan_row["workload_sha256"],
        "request_count_64": len(requests) == 64,
        "complete_64": len(completed) == 64,
        "exact_lengths": all(
            row.get("prompt_token_count") == 1024 and row.get("output_token_count") == 512
            for row in completed
        ),
        "batch_stream_nonempty": bool(batches),
        "all_required_batch_fields": all(REQUIRED_BATCH_FIELDS <= row.keys() for row in batches),
        "batch_indices_contiguous": [row.get("batch_index") for row in batches] == list(range(len(batches))),
        "batch_timestamps_monotonic": all(
            float(left["timestamp_ns"]) <= float(right["timestamp_ns"])
            for left, right in zip(batches, batches[1:])
        ),
        "effective_m_identity": all(
            int(row["effective_gemm_m"])
            == int(row["total_prefill_tokens"]) + int(row["num_decoding_sequences"])
            for row in batches
        ),
        "expected_safe_blocks": metadata.get("num_gpu_blocks") == EXPECTED_BLOCKS[condition]
        and all(int(row["safe_kv_blocks"]) == EXPECTED_BLOCKS[condition] for row in batches),
        "prefill_m_in_benchmarked_range": all(
            1024 <= int(row["effective_gemm_m"]) <= 32768
            for row in batches
            if int(row["num_prefill_sequences"]) > 0
        ),
    }
    valid = all(validation.values())

    derived = [derive_request_metrics(row) for row in completed]
    ttft = [float(row["ttft_s"]) for row in derived if row["ttft_s"] is not None]
    queue = [float(row["queueing_delay_s"]) for row in derived if row["queueing_delay_s"] is not None]
    tpot = [float(row["tpot_s"]) for row in derived if row["tpot_s"] is not None]
    prefill_to_output = [
        value
        for value in (
            delta_s(row, "first_output_token_time_ns", "first_prefill_time_ns")
            for row in completed
        )
        if value is not None
    ]
    arrival_to_eligible = [
        value
        for value in (
            delta_s(row, "scheduler_eligible_time_ns", "arrival_time_ns")
            for row in completed
        )
        if value is not None
    ]
    segments = telemetry_segments(telemetry, duration)
    waiting_area = integrate(segments, lambda row: float(row["waiting_q_depth"]))
    waiting_ge4_s = integrate(
        segments, lambda row: float(float(row["waiting_q_depth"]) >= 4)
    )
    kv_ge090_s = integrate(
        segments, lambda row: float(float(row["logical_kv_utilization"]) >= 0.90)
    )
    kv_ge095_s = integrate(
        segments, lambda row: float(float(row["logical_kv_utilization"]) >= 0.95)
    )
    kv_ge098_s = integrate(
        segments, lambda row: float(float(row["logical_kv_utilization"]) >= 0.98)
    )
    prefills = [row for row in batches if int(row["num_prefill_sequences"]) > 0]
    decodes = [row for row in batches if int(row["num_decoding_sequences"]) > 0]
    batch_sizes = [
        float(int(row["num_prefill_sequences"]) + int(row["num_decoding_sequences"]))
        for row in batches
    ]
    violations = sum(value > 2.0 for value in ttft)

    row: dict[str, Any] = {
        "run_id": metadata["run_id"],
        "condition": condition,
        "precision_state": "FP16" if condition == "fp16_0" else "AWQ-Marlin W4-16",
        "scale": scale,
        "nominal_offered_rps": float(metadata["nominal_offered_rps"]),
        "repeat": repeat,
        "protocol_valid": valid,
        "request_count": len(requests),
        "completed_request_count": len(completed),
        "failed_or_incomplete_request_count": len(requests) - len(completed),
        "strict_ttft_gt_2s_count": violations,
        "strict_ttft_gt_2s_percent": 100 * violations / len(requests),
        "measurement_duration_s": duration,
        "completed_throughput_rps": len(completed) / duration,
        "generated_token_throughput_tps": sum(int(item["output_token_count"]) for item in completed) / duration,
        "safe_kv_blocks": metadata["num_gpu_blocks"],
        "safe_kv_token_slots": metadata["gpu_kv_token_slots"],
        "swap_in_count": max((int(item.get("swap_in_count", 0)) for item in telemetry), default=0),
        "swap_out_count": max((int(item.get("swap_out_count", 0)) for item in telemetry), default=0),
        "preemption_count": max((int(item.get("preemption_count", 0)) for item in telemetry), default=0),
        "waiting_queue_area_request_s": waiting_area,
        "waiting_ge4_duration_s": waiting_ge4_s,
        "waiting_gt0_longest_duration_s": longest_duration(segments, lambda item: float(item["waiting_q_depth"]) > 0),
        "waiting_ge4_longest_duration_s": longest_duration(segments, lambda item: float(item["waiting_q_depth"]) >= 4),
        "kv_ge090_duration_s": kv_ge090_s,
        "kv_ge095_duration_s": kv_ge095_s,
        "kv_ge098_duration_s": kv_ge098_s,
        "kv_ge095_fraction": kv_ge095_s / duration,
        "peak_waiting_q_depth": max(numeric(telemetry, "waiting_q_depth"), default=0),
        "peak_running_q_count": max(numeric(telemetry, "running_q_count"), default=0),
        "peak_swapped_q_count": max(numeric(telemetry, "swapped_q_count"), default=0),
        "peak_logical_kv_utilization": max(numeric(telemetry, "logical_kv_utilization"), default=0),
        "minimum_logical_free_kv_blocks": min(
            (
                int(item["num_gpu_blocks"]) - int(item["num_decoding_gpu_blocks"])
                for item in telemetry
            ),
            default=None,
        ),
        "forward_batch_count": len(batches),
        "prefill_batch_count": len(prefills),
        "decode_batch_count": len(decodes),
    }
    row.update(flatten("ttft_s", ttft))
    row.update(flatten("queueing_delay_s", queue))
    row.update(flatten("first_prefill_to_first_output_s", prefill_to_output))
    row.update(flatten("tpot_s", tpot))
    row.update(flatten("arrival_to_scheduler_eligible_s", arrival_to_eligible))
    for field in (
        "waiting_q_depth",
        "running_q_count",
        "swapped_q_count",
        "num_decoding_gpu_blocks",
        "logical_kv_utilization",
    ):
        row.update(flatten(field, numeric(telemetry, field)))
    row.update(flatten("forward_batch_size", batch_sizes))
    row.update(flatten("prefill_sequence_count", numeric(prefills, "num_prefill_sequences")))
    row.update(flatten("prefill_total_tokens", numeric(prefills, "total_prefill_tokens")))
    row.update(flatten("prefill_effective_gemm_m", numeric(prefills, "effective_gemm_m")))
    row.update(flatten("prefill_execution_duration_s", numeric(prefills, "prefill_execution_duration_s")))
    row.update(flatten("decode_sequence_count", numeric(decodes, "num_decoding_sequences")))
    row.update(flatten("batch_waiting_before_scheduling", numeric(batches, "waiting_queue_depth_before_scheduling")))
    row.update(flatten("batch_running_before_scheduling", numeric(batches, "running_request_count_before_scheduling")))
    row.update(flatten("batch_used_kv_blocks_before_scheduling", numeric(batches, "used_kv_blocks_before_scheduling")))
    row.update(flatten("batch_allocated_kv_blocks_after_forward", numeric(batches, "allocated_gpu_blocks_after_forward")))

    by_m = []
    for effective_m in sorted({int(item["effective_gemm_m"]) for item in prefills}):
        members = [item for item in prefills if int(item["effective_gemm_m"]) == effective_m]
        durations = numeric(members, "prefill_execution_duration_s")
        by_m.append(
            {
                "run_id": metadata["run_id"],
                "condition": condition,
                "scale": scale,
                "repeat": repeat,
                "effective_gemm_m": effective_m,
                "prefill_sequence_count": effective_m // 1024,
                **flatten("duration_s", durations),
            }
        )
    histograms = {
        "prefill_effective_gemm_m": dict(sorted(Counter(int(item["effective_gemm_m"]) for item in prefills).items())),
        "prefill_sequence_count": dict(sorted(Counter(int(item["num_prefill_sequences"]) for item in prefills).items())),
        "decode_sequence_count": dict(sorted(Counter(int(item["num_decoding_sequences"]) for item in decodes).items())),
    }
    signal_rows = []
    for name, result in evaluate_signals(telemetry, duration).items():
        signal_rows.append(
            {
                "run_id": metadata["run_id"],
                "condition": condition,
                "scale": scale,
                "nominal_offered_rps": metadata["nominal_offered_rps"],
                "repeat": repeat,
                "candidate": name,
                **result,
            }
        )
    return row, by_m, signal_rows, {"checks": validation, "histograms": histograms}


def microbenchmark_analysis(root: Path, output_dir: Path) -> list[dict[str, Any]]:
    files = {
        "fp16_0": root / "microbenchmark/prefill-fp16_0.json",
        "awq_w4_16": root / "microbenchmark/prefill-awq_w4_16.json",
    }
    data = {condition: read_json(path) for condition, path in files.items()}
    fp16 = {int(row["effective_gemm_m"]): row for row in data["fp16_0"]["results"]}
    awq = {int(row["effective_gemm_m"]): row for row in data["awq_w4_16"]["results"]}
    supplements = {
        condition: read_json(root / f"microbenchmark/supplemental-prefill-m2048-{condition}.json")["results"][0]
        for condition in CONDITIONS
    }
    fp16[2048] = supplements["fp16_0"]
    awq[2048] = supplements["awq_w4_16"]
    if set(fp16) != set(awq):
        raise ValueError("FP16/AWQ microbenchmark M grids differ")
    rows = []
    for effective_m in sorted(fp16):
        left, right = fp16[effective_m], awq[effective_m]
        rows.append(
            {
                "effective_gemm_m": effective_m,
                "sequence_count": left["sequence_count"],
                "fp16_median_ms": left["median_ms"],
                "fp16_p95_ms": left["p95_ms"],
                "awq_median_ms": right["median_ms"],
                "awq_p95_ms": right["p95_ms"],
                "awq_over_fp16_median": float(right["median_ms"]) / float(left["median_ms"]),
                "awq_minus_fp16_median_ms": float(right["median_ms"]) - float(left["median_ms"]),
                "role": "supplemental observed M" if effective_m == 2048 else "pre-registered",
            }
        )
    write_csv(output_dir / "prefill_microbenchmark.csv", rows)
    return rows


def attach_matched_release_times(root: Path, signal_rows: list[dict[str, Any]]) -> None:
    entries = {
        (float(row["scale"]), int(row["repeat"]), str(row["candidate"])): row["first_entry_s"]
        for row in signal_rows
        if row["condition"] == "fp16_0"
    }
    telemetry_cache: dict[str, tuple[list[dict[str, Any]], float]] = {}
    for row in signal_rows:
        row["matched_fp16_entry_s"] = None
        row["release_after_matched_fp16_entry_s"] = None
        row["release_delay_after_matched_entry_s"] = None
        if row["condition"] != "awq_w4_16":
            continue
        entry = entries[(float(row["scale"]), int(row["repeat"]), str(row["candidate"]))]
        row["matched_fp16_entry_s"] = entry
        if entry is None:
            continue
        run_id = str(row["run_id"])
        if run_id not in telemetry_cache:
            run_dir = root / "runs" / run_id
            metadata = read_json(run_dir / "metadata.json")
            telemetry_cache[run_id] = (
                read_jsonl(run_dir / "telemetry.jsonl"),
                float(metadata["measurement_duration_s"]),
            )
        telemetry, duration = telemetry_cache[run_id]
        release = release_time(telemetry, duration, str(row["candidate"]), float(entry))
        row["release_after_matched_fp16_entry_s"] = release
        row["release_delay_after_matched_entry_s"] = (
            float(release) - float(entry) if release is not None else None
        )


def serving_variability(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "ttft_s_p50",
        "ttft_s_p95",
        "ttft_s_p99",
        "strict_ttft_gt_2s_percent",
        "queueing_delay_s_p50",
        "queueing_delay_s_p95",
        "queueing_delay_s_p99",
        "first_prefill_to_first_output_s_p50",
        "first_prefill_to_first_output_s_p95",
        "first_prefill_to_first_output_s_p99",
        "tpot_s_p50",
        "tpot_s_p95",
        "tpot_s_p99",
        "completed_throughput_rps",
        "waiting_queue_area_request_s",
        "kv_ge095_duration_s",
        "preemption_count",
        "prefill_sequence_count_p50",
        "prefill_sequence_count_p95",
        "prefill_effective_gemm_m_p50",
        "prefill_effective_gemm_m_p95",
        "prefill_execution_duration_s_p50",
        "prefill_execution_duration_s_p95",
    )
    grouped = []
    for condition in CONDITIONS:
        for scale in sorted({float(row["scale"]) for row in run_rows}, reverse=True):
            members = [
                row
                for row in run_rows
                if row["condition"] == condition and float(row["scale"]) == scale
            ]
            result: dict[str, Any] = {
                "condition": condition,
                "scale": scale,
                "nominal_offered_rps": members[0]["nominal_offered_rps"],
                "run_count": len(members),
                "run_ids": ";".join(str(row["run_id"]) for row in members),
            }
            for metric in metrics:
                values = [float(row[metric]) for row in members]
                result[f"{metric}_mean"] = sum(values) / len(values)
                result[f"{metric}_min"] = min(values)
                result[f"{metric}_max"] = max(values)
            grouped.append(result)
    return grouped


def mechanism_by_load(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "prefill_batch_count",
        "prefill_sequence_count_mean",
        "prefill_sequence_count_p95",
        "prefill_effective_gemm_m_mean",
        "prefill_effective_gemm_m_p95",
        "prefill_execution_duration_s_p50",
        "prefill_execution_duration_s_p95",
        "queueing_delay_s_p95",
        "waiting_queue_area_request_s",
        "kv_ge095_duration_s",
        "preemption_count",
        "completed_throughput_rps",
    )
    output = []
    for scale in sorted({float(row["scale"]) for row in run_rows}, reverse=True):
        state_rows = {
            condition: [
                row
                for row in run_rows
                if row["condition"] == condition and float(row["scale"]) == scale
            ]
            for condition in CONDITIONS
        }
        row: dict[str, Any] = {
            "scale": scale,
            "nominal_offered_rps": state_rows["fp16_0"][0]["nominal_offered_rps"],
        }
        for condition, members in state_rows.items():
            for field in fields:
                row[f"{condition}_{field}_mean"] = sum(float(item[field]) for item in members) / len(members)
        row["awq_minus_fp16_prefill_sequence_count_mean"] = (
            row["awq_w4_16_prefill_sequence_count_mean_mean"]
            - row["fp16_0_prefill_sequence_count_mean_mean"]
        )
        row["awq_minus_fp16_prefill_batch_count"] = (
            row["awq_w4_16_prefill_batch_count_mean"]
            - row["fp16_0_prefill_batch_count_mean"]
        )
        row["awq_over_fp16_waiting_area"] = (
            row["awq_w4_16_waiting_queue_area_request_s_mean"]
            / row["fp16_0_waiting_queue_area_request_s_mean"]
            if row["fp16_0_waiting_queue_area_request_s_mean"] else None
        )
        output.append(row)
    return output


def serving_prefill_by_m(by_m_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    effective_ms = sorted({int(row["effective_gemm_m"]) for row in by_m_rows})
    for effective_m in effective_ms:
        state_values: dict[str, list[float]] = {}
        state_counts: dict[str, int] = {}
        for condition in CONDITIONS:
            members = [
                row
                for row in by_m_rows
                if row["condition"] == condition and int(row["effective_gemm_m"]) == effective_m
            ]
            state_values[condition] = [float(row["duration_s_p50"]) for row in members]
            state_counts[condition] = sum(int(row["duration_s_count"]) for row in members)
        row: dict[str, Any] = {
            "effective_gemm_m": effective_m,
            "sequence_count": effective_m // 1024,
            "fp16_event_count": state_counts["fp16_0"],
            "awq_event_count": state_counts["awq_w4_16"],
            "fp16_run_median_count": len(state_values["fp16_0"]),
            "awq_run_median_count": len(state_values["awq_w4_16"]),
            "fp16_median_of_run_medians_s": distribution(state_values["fp16_0"])["p50"],
            "awq_median_of_run_medians_s": distribution(state_values["awq_w4_16"])["p50"],
        }
        if row["fp16_median_of_run_medians_s"] and row["awq_median_of_run_medians_s"]:
            row["awq_over_fp16_serving_duration"] = (
                float(row["awq_median_of_run_medians_s"])
                / float(row["fp16_median_of_run_medians_s"])
            )
        else:
            row["awq_over_fp16_serving_duration"] = None
        output.append(row)
    return output


def classify_points(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points = []
    scales = sorted({float(row["scale"]) for row in run_rows}, reverse=True)
    for scale in scales:
        by_condition = {
            condition: sorted(
                [row for row in run_rows if row["condition"] == condition and float(row["scale"]) == scale],
                key=lambda row: int(row["repeat"]),
            )
            for condition in CONDITIONS
        }
        if any(len(rows) != 2 for rows in by_condition.values()):
            raise ValueError(f"scale {scale} does not have two repeats per state")
        fp16, awq = by_condition["fp16_0"], by_condition["awq_w4_16"]
        fp_values = [float(row["ttft_s_p95"]) for row in fp16]
        awq_values = [float(row["ttft_s_p95"]) for row in awq]
        differences = [left - right for left, right in zip(fp_values, awq_values)]
        epsilon = max(
            0.111,
            max(fp_values) - min(fp_values),
            max(awq_values) - min(awq_values),
            0.05 * sum(fp_values) / 2,
        )
        queue_gate = all(
            float(right["queueing_delay_s_p95"]) < float(left["queueing_delay_s_p95"])
            for left, right in zip(fp16, awq)
        )
        queue_area_gate = all(
            float(right["waiting_queue_area_request_s"]) < float(left["waiting_queue_area_request_s"])
            for left, right in zip(fp16, awq)
        )
        kv_dwell_gate = all(
            float(right["kv_ge095_duration_s"]) < float(left["kv_ge095_duration_s"])
            for left, right in zip(fp16, awq)
        )
        preemption_gate = all(
            int(right["preemption_count"]) <= int(left["preemption_count"])
            for left, right in zip(fp16, awq)
        ) and any(
            int(right["preemption_count"]) < int(left["preemption_count"])
            for left, right in zip(fp16, awq)
        )
        throughput_each_gate = all(
            float(right["completed_throughput_rps"]) >= 0.98 * float(left["completed_throughput_rps"])
            for left, right in zip(fp16, awq)
        )
        throughput_mean_gate = sum(float(row["completed_throughput_rps"]) for row in awq) >= sum(
            float(row["completed_throughput_rps"]) for row in fp16
        )
        valid_gate = all(bool(row["protocol_valid"]) for row in fp16 + awq)
        awq_preferred = all(value > epsilon for value in differences) and all(
            (
                queue_gate,
                queue_area_gate,
                kv_dwell_gate,
                preemption_gate,
                throughput_each_gate,
                throughput_mean_gate,
                valid_gate,
            )
        )
        both_fp16_faster = all(value < 0 for value in differences)
        both_within_noise = all(abs(value) <= epsilon for value in differences)
        no_awq_beyond_noise = all(value <= epsilon for value in differences)
        fp16_default = not awq_preferred and no_awq_beyond_noise and (
            both_fp16_faster or both_within_noise
        )
        classification = (
            "AWQ-preferred" if awq_preferred else "FP16-default/non-inferior" if fp16_default else "ambiguous"
        )
        points.append(
            {
                "scale": scale,
                "nominal_offered_rps": fp16[0]["nominal_offered_rps"],
                "classification": classification,
                "epsilon_s": epsilon,
                "fp16_p95_ttft_rep0_s": fp_values[0],
                "fp16_p95_ttft_rep1_s": fp_values[1],
                "awq_p95_ttft_rep0_s": awq_values[0],
                "awq_p95_ttft_rep1_s": awq_values[1],
                "p95_advantage_rep0_s": differences[0],
                "p95_advantage_rep1_s": differences[1],
                "both_advantages_exceed_noise": all(value > epsilon for value in differences),
                "queue_p95_lower_both": queue_gate,
                "waiting_area_lower_both": queue_area_gate,
                "kv_ge095_dwell_lower_both": kv_dwell_gate,
                "preemption_relief_gate": preemption_gate,
                "throughput_each_at_least_98pct": throughput_each_gate,
                "throughput_mean_no_lower": throughput_mean_gate,
                "all_runs_protocol_valid": valid_gate,
            }
        )
    return points


def decision_analysis(
    run_rows: list[dict[str, Any]], point_rows: list[dict[str, Any]], signal_rows: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    classifications = {float(row["scale"]): row["classification"] for row in point_rows}
    awq_scales = [
        float(row["scale"]) for row in point_rows if row["classification"] == "AWQ-preferred"
    ]
    default_scales = [
        float(row["scale"])
        for row in point_rows
        if row["classification"] == "FP16-default/non-inferior"
    ]
    ordered_by_load = sorted(point_rows, key=lambda row: float(row["nominal_offered_rps"]))
    first_awq_index = next(
        (index for index, row in enumerate(ordered_by_load) if row["classification"] == "AWQ-preferred"),
        None,
    )
    no_fp16_reversal = first_awq_index is not None and all(
        row["classification"] != "FP16-default/non-inferior"
        for row in ordered_by_load[first_awq_index + 1 :]
    )
    last_default_index = max(
        (index for index, row in enumerate(ordered_by_load) if row["classification"] == "FP16-default/non-inferior"),
        default=-1,
    )
    no_awq_below_last_default = all(
        row["classification"] != "AWQ-preferred"
        for row in ordered_by_load[:last_default_index]
    )

    point_class_by_scale = {float(row["scale"]): row["classification"] for row in point_rows}
    signal_qualification = []
    for candidate in sorted({str(row["candidate"]) for row in signal_rows}):
        fp16_rows = [
            row for row in signal_rows if row["condition"] == "fp16_0" and row["candidate"] == candidate
        ]
        high_rows = [
            row
            for row in fp16_rows
            if point_class_by_scale[float(row["scale"])] == "AWQ-preferred"
        ]
        low_rows = [
            row
            for row in fp16_rows
            if point_class_by_scale[float(row["scale"])] == "FP16-default/non-inferior"
        ]
        high_fires_all = bool(high_rows) and all(bool(row["entry_fired"]) for row in high_rows)
        low_fires_none = all(not bool(row["entry_fired"]) for row in low_rows)
        matched_awq_rows = [
            row
            for row in signal_rows
            if row["condition"] == "awq_w4_16"
            and row["candidate"] == candidate
            and point_class_by_scale[float(row["scale"])] == "AWQ-preferred"
            and row.get("matched_fp16_entry_s") is not None
        ]
        release_delays = [
            float(row["release_delay_after_matched_entry_s"])
            for row in matched_awq_rows
            if row.get("release_delay_after_matched_entry_s") is not None
        ]
        signal_qualification.append(
            {
                "candidate": candidate,
                "awq_preferred_fp16_run_count": len(high_rows),
                "awq_preferred_fire_count": sum(bool(row["entry_fired"]) for row in high_rows),
                "fp16_default_run_count": len(low_rows),
                "fp16_default_false_fire_count": sum(bool(row["entry_fired"]) for row in low_rows),
                "fires_all_awq_preferred_runs": high_fires_all,
                "fires_no_fp16_default_runs": low_fires_none,
                "matched_awq_release_eligible_count": len(release_delays),
                "matched_awq_min_release_delay_s": min(release_delays) if release_delays else None,
                "matched_awq_max_release_delay_s": max(release_delays) if release_delays else None,
                "qualifies": high_fires_all and low_fires_none,
            }
        )
    qualifying_signals = [row["candidate"] for row in signal_qualification if row["qualifies"]]
    all_runs_valid = len(run_rows) == 36 and all(bool(row["protocol_valid"]) for row in run_rows)
    gates = {
        "scale_6_fp16_default": classifications.get(6.0) == "FP16-default/non-inferior",
        "scale_4_awq_preferred": classifications.get(4.0) == "AWQ-preferred",
        "scale_4p25_awq_preferred": classifications.get(4.25) == "AWQ-preferred",
        "no_fp16_reversal_after_first_awq_win": no_fp16_reversal,
        "no_awq_win_below_last_fp16_default": no_awq_below_last_default,
        "at_least_one_online_signal_qualifies": bool(qualifying_signals),
        "all_36_runs_protocol_valid": all_runs_valid,
    }
    go = all(gates.values())
    last_default = max(
        (row for row in point_rows if row["classification"] == "FP16-default/non-inferior"),
        key=lambda row: float(row["nominal_offered_rps"]),
        default=None,
    )
    first_awq = min(
        (row for row in point_rows if row["classification"] == "AWQ-preferred"),
        key=lambda row: float(row["nominal_offered_rps"]),
        default=None,
    )
    decision = {
        "schema_version": 1,
        "decision": "GO" if go else "NO-GO",
        "validated_state_pair": ["FP16", "AWQ-W4-16"],
        "gates": gates,
        "qualifying_online_signals": qualifying_signals,
        "awq_preferred_scales": awq_scales,
        "fp16_default_scales": default_scales,
        "ambiguous_scales": [
            float(row["scale"]) for row in point_rows if row["classification"] == "ambiguous"
        ],
        "crossover_region": (
            {
                "fp16_side_scale": last_default["scale"],
                "fp16_side_nominal_rps": last_default["nominal_offered_rps"],
                "awq_side_scale": first_awq["scale"],
                "awq_side_nominal_rps": first_awq["nominal_offered_rps"],
            }
            if last_default and first_awq
            else None
        ),
        "controller_implemented": False,
    }
    return decision, signal_qualification


def plot_results(run_rows: list[dict[str, Any]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=180)
    colors = {"fp16_0": "#1f77b4", "awq_w4_16": "#d62728"}
    for condition in CONDITIONS:
        grouped = []
        for load in sorted({float(row["nominal_offered_rps"]) for row in run_rows}):
            members = [
                row for row in run_rows if row["condition"] == condition and float(row["nominal_offered_rps"]) == load
            ]
            values = [float(row["ttft_s_p95"]) for row in members]
            grouped.append((load, sum(values) / len(values), min(values), max(values)))
        x = [item[0] for item in grouped]
        y = [item[1] for item in grouped]
        axes[0].errorbar(
            x,
            y,
            yerr=[[mean - low for _, mean, low, _ in grouped], [high - mean for _, mean, _, high in grouped]],
            marker="o",
            capsize=3,
            label=condition,
            color=colors[condition],
        )
        queue = []
        for load in x:
            members = [
                row for row in run_rows if row["condition"] == condition and float(row["nominal_offered_rps"]) == load
            ]
            queue.append(sum(float(row["waiting_queue_area_request_s"]) for row in members) / len(members))
        axes[1].plot(x, queue, marker="o", label=condition, color=colors[condition])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("P95 TTFT (s; repeat range)")
    axes[1].set_yscale("symlog", linthresh=1)
    axes[1].set_ylabel("Waiting queue area (request-s)")
    for axis in axes:
        axis.set_xlabel("Nominal offered requests/s")
        axis.grid(True, alpha=0.25)
    axes[0].legend()
    fig.suptitle("Pre-registered FP16 / AWQ-W4-16 crossover grid")
    fig.tight_layout()
    fig.savefig(output_dir / "crossover_latency_queue.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = read_json(args.root / "run-plan.json")
    if len(plan["runs"]) != 36:
        raise ValueError("analysis requires the complete frozen 36-run plan")

    run_rows = []
    by_m_rows = []
    signal_rows = []
    run_audit: dict[str, Any] = {}
    for plan_row in plan["runs"]:
        run_dir = args.root / "runs" / plan_row["run_id"]
        row, by_m, signals, audit = analyze_run(run_dir, plan_row)
        run_rows.append(row)
        by_m_rows.extend(by_m)
        signal_rows.extend(signals)
        run_audit[plan_row["run_id"]] = audit
    run_rows.sort(key=lambda row: (float(row["nominal_offered_rps"]), CONDITIONS.index(str(row["condition"])), int(row["repeat"])))
    write_csv(args.output_dir / "serving_runs.csv", run_rows)
    write_csv(args.output_dir / "prefill_duration_by_m.csv", by_m_rows)
    mechanism_rows = mechanism_by_load(run_rows)
    write_csv(args.output_dir / "mechanism_by_load.csv", mechanism_rows)
    serving_m_rows = serving_prefill_by_m(by_m_rows)
    write_csv(args.output_dir / "serving_prefill_by_m.csv", serving_m_rows)

    point_rows = classify_points(run_rows)
    write_csv(args.output_dir / "crossover_points.csv", point_rows)
    attach_matched_release_times(args.root, signal_rows)
    decision, signal_qualification = decision_analysis(run_rows, point_rows, signal_rows)
    point_class = {float(row["scale"]): row["classification"] for row in point_rows}
    for row in signal_rows:
        row["offline_point_classification"] = point_class[float(row["scale"])]
    write_csv(args.output_dir / "online_signals.csv", signal_rows)
    write_csv(args.output_dir / "online_signal_qualification.csv", signal_qualification)
    variability_rows = serving_variability(run_rows)
    write_csv(args.output_dir / "serving_variability.csv", variability_rows)
    micro_rows = microbenchmark_analysis(args.root, args.output_dir)
    plot_results(run_rows, args.output_dir)

    (args.output_dir / "batch_histograms.json").write_text(
        json.dumps(run_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "crossover_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    aggregate = {
        "schema_version": 1,
        "root": str(args.root),
        "protocol_manifest_sha256": hashlib.sha256((args.root / "protocol_manifest.json").read_bytes()).hexdigest(),
        "run_plan_sha256": hashlib.sha256((args.root / "run-plan.json").read_bytes()).hexdigest(),
        "run_count": len(run_rows),
        "point_count": len(point_rows),
        "mechanism_load_row_count": len(mechanism_rows),
        "serving_prefill_m_row_count": len(serving_m_rows),
        "variability_row_count": len(variability_rows),
        "microbenchmark_point_count": len(micro_rows),
        "all_runs_protocol_valid": all(bool(row["protocol_valid"]) for row in run_rows),
        "decision": decision,
        "source_files": ["metadata.json", "requests.jsonl", "telemetry.jsonl", "batches.jsonl"],
    }
    (args.output_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
