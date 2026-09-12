"""Regenerate v5 quality, serving, mechanism, and eligibility artifacts from raw runs."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Iterable

import matplotlib.pyplot as plt

from .metrics import derive_request_metrics, distribution, percentile

CONDITIONS = ("fp16_0", "w4_8", "w4_16", "w4_32")
COLORS = {
    "fp16_0": "#1f77b4",
    "w4_8": "#ff7f0e",
    "w4_16": "#2ca02c",
    "w4_32": "#d62728",
}


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
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def normalize_chars(text: str) -> list[str]:
    return [char for char in text if char.strip()]


def local_f1(pred: list[str], ref: list[str]) -> float:
    if not pred or not ref:
        return 0.0
    same = sum((Counter(pred) & Counter(ref)).values())
    if not same:
        return 0.0
    precision = same / len(pred)
    recall = same / len(ref)
    return 2 * precision * recall / (precision + recall)


def best_reference_f1(answer: str, references: list[str]) -> float:
    pred = normalize_chars(answer or "")
    return max((local_f1(pred, normalize_chars(ref)) for ref in references), default=0.0)


def bootstrap_mean_ci(
    values: list[float], *, repeats: int = 10_000, seed: int = 2025
) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    generator = random.Random(seed)
    count = len(values)
    means = [
        sum(values[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(repeats)
    ]
    return percentile(means, 2.5), percentile(means, 97.5)


def longest_common_prefix(first: list[int], second: list[int]) -> int:
    length = 0
    for left, right in zip(first, second):
        if left != right:
            break
        length += 1
    return length


def token_agreement(first: list[int], second: list[int]) -> float:
    denominator = max(len(first), len(second))
    if denominator == 0:
        return 1.0
    matches = sum(left == right for left, right in zip(first, second))
    return matches / denominator


def condition_from_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    requests = read_jsonl(run_dir / "requests.jsonl")
    telemetry = read_jsonl(run_dir / "telemetry.jsonl")
    return metadata, requests, telemetry


def quality_analysis(run_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    runs: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    workload_hashes: set[str] = set()
    for run_dir in run_dirs:
        metadata, requests, _ = condition_from_run(run_dir)
        condition = str(metadata["condition"])
        if condition in runs:
            raise ValueError(f"duplicate quality condition: {condition}")
        if len(requests) < 100:
            raise ValueError(f"quality run {run_dir} has only {len(requests)} requests")
        if any(row.get("status") != "completed" for row in requests):
            raise ValueError(f"quality run {run_dir} is incomplete")
        runs[condition] = (metadata, requests)
        workload_hashes.add(str(metadata["workload_sha256"]))
    if set(runs) != set(CONDITIONS) or len(workload_hashes) != 1:
        raise ValueError("quality runs must contain all four conditions on one workload hash")

    by_condition: dict[str, dict[str, dict[str, Any]]] = {}
    for condition, (_, requests) in runs.items():
        by_condition[condition] = {}
        for row in requests:
            request_id = str(row["benchmark_request_id"])
            references = [str(value) for value in row.get("dataset_reference_answers", [])]
            by_condition[condition][request_id] = {
                "request_id": request_id,
                "sequence": row.get("sequence"),
                "dataset_question_id": row.get("dataset_question_id"),
                "f1": best_reference_f1(str(row.get("generated_answer") or ""), references),
                "output_token_ids": [int(value) for value in row.get("output_token_ids", [])],
            }
    request_ids = list(by_condition["fp16_0"])
    if any(set(rows) != set(request_ids) for rows in by_condition.values()):
        raise ValueError("quality request IDs are not paired across conditions")

    paired_rows: list[dict[str, Any]] = []
    for request_id in request_ids:
        base = by_condition["fp16_0"][request_id]
        row: dict[str, Any] = {
            "request_id": request_id,
            "sequence": base["sequence"],
            "dataset_question_id": base["dataset_question_id"],
            "fp16_f1": base["f1"],
        }
        for condition in CONDITIONS[1:]:
            current = by_condition[condition][request_id]
            row[f"{condition}_f1"] = current["f1"]
            row[f"{condition}_minus_fp16_f1"] = current["f1"] - base["f1"]
            row[f"{condition}_token_agreement"] = token_agreement(
                base["output_token_ids"], current["output_token_ids"]
            )
            row[f"{condition}_longest_common_prefix_tokens"] = longest_common_prefix(
                base["output_token_ids"], current["output_token_ids"]
            )
            row[f"{condition}_exact_output_match"] = int(
                base["output_token_ids"] == current["output_token_ids"]
            )
        paired_rows.append(row)
    paired_rows.sort(key=lambda row: int(row["sequence"]))
    write_csv(output_dir / "quality_paired_differences.csv", paired_rows)

    table: list[dict[str, Any]] = []
    agreement_table: list[dict[str, Any]] = []
    base_values = [float(row["fp16_f1"]) for row in paired_rows]
    for condition_index, condition in enumerate(CONDITIONS):
        key = "fp16_f1" if condition == "fp16_0" else f"{condition}_f1"
        values = [float(row[key]) for row in paired_rows]
        mean_ci = bootstrap_mean_ci(values, seed=2025 + condition_index)
        deltas = [current - base for current, base in zip(values, base_values)]
        delta_ci = bootstrap_mean_ci(deltas, seed=2125 + condition_index)
        table.append(
            {
                "condition": condition,
                "quantized_layer_count": int(runs[condition][0]["quantized_layer_count"]),
                "request_count": len(values),
                "f1_percent": 100 * sum(values) / len(values),
                "f1_percent_ci95_low": 100 * float(mean_ci[0]),
                "f1_percent_ci95_high": 100 * float(mean_ci[1]),
                "paired_delta_vs_fp16_percentage_points": 100 * sum(deltas) / len(deltas),
                "paired_delta_ci95_low": 100 * float(delta_ci[0]),
                "paired_delta_ci95_high": 100 * float(delta_ci[1]),
                "paired_delta_positive_count": sum(value > 0 for value in deltas),
                "paired_delta_zero_count": sum(value == 0 for value in deltas),
                "paired_delta_negative_count": sum(value < 0 for value in deltas),
            }
        )
        if condition == "fp16_0":
            continue
        agreements = [float(row[f"{condition}_token_agreement"]) for row in paired_rows]
        prefixes = [float(row[f"{condition}_longest_common_prefix_tokens"]) for row in paired_rows]
        agreement_ci = bootstrap_mean_ci(agreements, seed=2225 + condition_index)
        prefix_ci = bootstrap_mean_ci(prefixes, seed=2325 + condition_index)
        agreement_table.append(
            {
                "condition": condition,
                "request_count": len(agreements),
                "mean_position_aligned_token_agreement": sum(agreements) / len(agreements),
                "token_agreement_ci95_low": agreement_ci[0],
                "token_agreement_ci95_high": agreement_ci[1],
                "exact_output_match_count": sum(
                    int(row[f"{condition}_exact_output_match"]) for row in paired_rows
                ),
                "exact_output_match_percent": 100
                * sum(int(row[f"{condition}_exact_output_match"]) for row in paired_rows)
                / len(paired_rows),
                "mean_longest_common_prefix_tokens": sum(prefixes) / len(prefixes),
                "longest_common_prefix_ci95_low": prefix_ci[0],
                "longest_common_prefix_ci95_high": prefix_ci[1],
                "metric_role": "secondary quantization-distortion evidence; not primary QA quality",
            }
        )
    write_csv(output_dir / "quality_table.csv", table)
    write_csv(output_dir / "quality_output_agreement.csv", agreement_table)
    result = {
        "schema_version": 1,
        "bootstrap": {"method": "paired request resampling percentile CI", "repeats": 10_000, "seed_family": 2025},
        "workload_sha256": next(iter(workload_hashes)),
        "quality_table": table,
        "output_agreement": agreement_table,
        "raw_runs": [str(path) for path in run_dirs],
    }
    (output_dir / "quality_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def numeric_values(rows: Iterable[dict[str, Any]], field: str) -> list[float]:
    return [float(row[field]) for row in rows if isinstance(row.get(field), (int, float))]


def stage_delta(row: dict[str, Any], end: str, start: str) -> float | None:
    if row.get(end) is None or row.get(start) is None:
        return None
    return (int(row[end]) - int(row[start])) / 1e9


def flatten_distribution(prefix: str, values: list[float]) -> dict[str, Any]:
    stats = distribution(values)
    return {f"{prefix}_{key}": value for key, value in stats.items()}


def serving_analysis(run_dirs: list[Path], output_dir: Path, near_scale: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_rows: list[dict[str, Any]] = []
    raw_by_key: dict[tuple[str, float, int], list[dict[str, Any]]] = {}
    for run_dir in run_dirs:
        metadata, requests, telemetry = condition_from_run(run_dir)
        condition = str(metadata["condition"])
        scale = float(metadata["workload_time_scale"])
        match = re.search(r"-rep(\d+)$", str(metadata["run_id"]))
        repeat = int(match.group(1)) if match else 0
        completed = [row for row in requests if row.get("status") == "completed"]
        derived = [derive_request_metrics(row) for row in completed]
        ttft = [float(row["ttft_s"]) for row in derived if row["ttft_s"] is not None]
        queue = [float(row["queueing_delay_s"]) for row in derived if row["queueing_delay_s"] is not None]
        tpot = [float(row["tpot_s"]) for row in derived if row["tpot_s"] is not None]
        arrival_to_eligible = [
            value for value in (stage_delta(row, "scheduler_eligible_time_ns", "arrival_time_ns") for row in completed)
            if value is not None
        ]
        prefill = [
            value for value in (stage_delta(row, "first_output_token_time_ns", "first_prefill_time_ns") for row in completed)
            if value is not None
        ]
        violations = sum(value > 2.0 for value in ttft)
        duration = float(metadata["measurement_duration_s"])
        row: dict[str, Any] = {
            "run_id": metadata["run_id"],
            "condition": condition,
            "quantized_layer_count": metadata["quantized_layer_count"],
            "time_scale": scale,
            "nominal_offered_rps": metadata["nominal_offered_rps"],
            "repeat": repeat,
            "request_count": len(requests),
            "completed_request_count": len(completed),
            "failed_or_incomplete_request_count": len(requests) - len(completed),
            "slo_violation_count": violations,
            "slo_violation_percent": 100 * violations / len(requests),
            "measurement_duration_s": duration,
            "completed_throughput_rps": len(completed) / duration,
            "generated_token_throughput_tps": sum(int(item.get("output_token_count", 0)) for item in completed) / duration,
            "num_gpu_blocks": metadata.get("num_gpu_blocks"),
            "gpu_kv_token_slots": metadata.get("gpu_kv_token_slots"),
            "swap_in_count": max((int(item.get("swap_in_count", 0)) for item in telemetry), default=0),
            "swap_out_count": max((int(item.get("swap_out_count", 0)) for item in telemetry), default=0),
            "preemption_count": max((int(item.get("preemption_count", 0)) for item in telemetry), default=0),
            "peak_waiting_q_depth": max(numeric_values(telemetry, "waiting_q_depth"), default=None),
            "peak_running_q_count": max(numeric_values(telemetry, "running_q_count"), default=None),
            "peak_swapped_q_count": max(numeric_values(telemetry, "swapped_q_count"), default=None),
            "peak_logical_kv_utilization": max(numeric_values(telemetry, "logical_kv_utilization"), default=None),
            "peak_hbm_used_gib": max(numeric_values(telemetry, "gpu_memory_used_bytes"), default=math.nan) / 2**30,
            "min_hbm_free_gib": min(numeric_values(telemetry, "gpu_memory_free_bytes"), default=math.nan) / 2**30,
        }
        row.update(flatten_distribution("ttft_s", ttft))
        row.update(flatten_distribution("queueing_delay_s", queue))
        row.update(flatten_distribution("prefill_to_first_output_s", prefill))
        row.update(flatten_distribution("decode_tpot_s", tpot))
        row.update(flatten_distribution("arrival_to_scheduler_eligible_s", arrival_to_eligible))
        row.update(flatten_distribution("waiting_q_depth", numeric_values(telemetry, "waiting_q_depth")))
        row.update(flatten_distribution("running_q_count", numeric_values(telemetry, "running_q_count")))
        row.update(flatten_distribution("swapped_q_count", numeric_values(telemetry, "swapped_q_count")))
        row.update(flatten_distribution("logical_kv_utilization", numeric_values(telemetry, "logical_kv_utilization")))
        run_rows.append(row)
        raw_by_key[(condition, scale, repeat)] = completed
    run_rows.sort(key=lambda row: (float(row["nominal_offered_rps"]), CONDITIONS.index(str(row["condition"])), int(row["repeat"])))
    write_csv(output_dir / "serving_runs.csv", run_rows)

    grouped_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        scales = sorted({float(row["time_scale"]) for row in run_rows if row["condition"] == condition}, reverse=True)
        for scale in scales:
            members = [row for row in run_rows if row["condition"] == condition and float(row["time_scale"]) == scale]
            for metric in ("ttft_s_p95", "ttft_s_p99", "slo_violation_percent", "queueing_delay_s_p95", "completed_throughput_rps", "peak_logical_kv_utilization"):
                if any(row.get(metric) is None for row in members):
                    raise ValueError(f"missing {metric} in {condition} scale {scale}")
            grouped_rows.append(
                {
                    "condition": condition,
                    "quantized_layer_count": members[0]["quantized_layer_count"],
                    "time_scale": scale,
                    "nominal_offered_rps": members[0]["nominal_offered_rps"],
                    "run_count": len(members),
                    "run_ids": ";".join(str(row["run_id"]) for row in members),
                    **{
                        f"{metric}_{suffix}": function(float(row[metric]) for row in members)
                        for metric in ("ttft_s_p95", "ttft_s_p99", "slo_violation_percent", "queueing_delay_s_p95", "completed_throughput_rps", "peak_logical_kv_utilization")
                        for suffix, function in (("mean", lambda values: sum(values) / len(members)), ("min", min), ("max", max))
                    },
                }
            )
    write_csv(output_dir / "serving_variability.csv", grouped_rows)

    near_rows: list[dict[str, Any]] = []
    fp16 = {int(row["repeat"]): row for row in run_rows if row["condition"] == "fp16_0" and float(row["time_scale"]) == near_scale}
    for condition in CONDITIONS[1:]:
        current = {int(row["repeat"]): row for row in run_rows if row["condition"] == condition and float(row["time_scale"]) == near_scale}
        for repeat in sorted(set(fp16) & set(current)):
            base_row, current_row = fp16[repeat], current[repeat]
            near_rows.append(
                {
                    "condition": condition,
                    "repeat": repeat,
                    "time_scale": near_scale,
                    "fp16_p95_ttft_s": base_row["ttft_s_p95"],
                    "w4_p95_ttft_s": current_row["ttft_s_p95"],
                    "p95_reduction_s": float(base_row["ttft_s_p95"]) - float(current_row["ttft_s_p95"]),
                    "fp16_slo_violation_percent": base_row["slo_violation_percent"],
                    "w4_slo_violation_percent": current_row["slo_violation_percent"],
                    "slo_reduction_percentage_points": float(base_row["slo_violation_percent"]) - float(current_row["slo_violation_percent"]),
                    "fp16_p95_queueing_s": base_row["queueing_delay_s_p95"],
                    "w4_p95_queueing_s": current_row["queueing_delay_s_p95"],
                    "fp16_peak_kv": base_row["peak_logical_kv_utilization"],
                    "w4_peak_kv": current_row["peak_logical_kv_utilization"],
                }
            )
    write_csv(output_dir / "near_knee_paired_runs.csv", near_rows)
    (output_dir / "serving_summary.json").write_text(
        json.dumps({"schema_version": 1, "runs": run_rows, "variability": grouped_rows, "near_knee_pairs": near_rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_serving(grouped_rows, output_dir / "latency_slo_vs_load.png")
    plot_pressure(grouped_rows, output_dir / "queue_kv_vs_load.png")
    return run_rows, grouped_rows


def plot_serving(rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=180)
    for condition in CONDITIONS:
        members = sorted((row for row in rows if row["condition"] == condition), key=lambda row: float(row["nominal_offered_rps"]))
        x = [float(row["nominal_offered_rps"]) for row in members]
        for ax, metric, ylabel in (
            (axes[0], "ttft_s_p95", "P95 TTFT (s)"),
            (axes[1], "slo_violation_percent", "TTFT >2 s (%)"),
        ):
            y = [float(row[f"{metric}_mean"]) for row in members]
            low = [value - float(row[f"{metric}_min"]) for value, row in zip(y, members)]
            high = [float(row[f"{metric}_max"]) - value for value, row in zip(y, members)]
            ax.errorbar(x, y, yerr=[low, high], marker="o", capsize=3, label=condition, color=COLORS[condition])
            ax.set_xlabel("Nominal offered requests/s")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
    axes[0].axhline(2.0, color="0.4", linestyle="--", linewidth=1)
    axes[0].legend(fontsize=8)
    fig.suptitle("Static serving latency/SLO versus frozen load")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_pressure(rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=180)
    for condition in CONDITIONS:
        members = sorted((row for row in rows if row["condition"] == condition), key=lambda row: float(row["nominal_offered_rps"]))
        x = [float(row["nominal_offered_rps"]) for row in members]
        axes[0].plot(x, [float(row["queueing_delay_s_p95_mean"]) for row in members], marker="o", label=condition, color=COLORS[condition])
        axes[1].plot(x, [float(row["peak_logical_kv_utilization_mean"]) for row in members], marker="o", label=condition, color=COLORS[condition])
    axes[0].set_ylabel("P95 queueing delay (s)")
    axes[1].set_ylabel("Peak logical KV utilization")
    for ax in axes:
        ax.set_xlabel("Nominal offered requests/s")
        ax.grid(True, alpha=0.25)
    axes[1].axhline(0.85, color="0.4", linestyle="--", linewidth=1)
    axes[0].legend(fontsize=8)
    fig.suptitle("Queue and KV pressure versus frozen load")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def coefficient_of_variation(values: list[float]) -> float:
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / mean


def mechanism_analysis(profile_files: list[Path], micro_files: list[Path], serving_rows: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    profiles: dict[int, list[dict[str, Any]]] = {}
    for path in profile_files:
        row = json.loads(path.read_text(encoding="utf-8"))
        profiles.setdefault(int(row["quantized_layer_count"]), []).append(row)
    micros: dict[int, dict[str, Any]] = {}
    for path in micro_files:
        row = json.loads(path.read_text(encoding="utf-8"))
        micros[int(row["quantized_layer_count"])] = row
    if set(profiles) != {0, 8, 16, 32} or set(micros) != {0, 8, 16, 32}:
        raise ValueError("mechanism inputs must cover 0/8/16/32")

    rows: list[dict[str, Any]] = []
    for condition, layers in zip(CONDITIONS, (0, 8, 16, 32)):
        if len(profiles[layers]) < 2:
            raise ValueError(f"need two profile repeats for {layers} layers")
        resident = [float(row["resident_weight_bytes_after_load"]) for row in profiles[layers]]
        workspace = [float(row["profile_peak_allocated_bytes"] - row["resident_weight_bytes_after_load"]) for row in profiles[layers]]
        blocks = [float(row["num_gpu_blocks"]) for row in profiles[layers]]
        condition_serving = sorted((row for row in serving_rows if row["condition"] == condition), key=lambda row: float(row["nominal_offered_rps"]))
        saturated = [
            row for row in condition_serving
            if float(row["queueing_delay_s_p95"]) >= 2.0 or float(row["peak_logical_kv_utilization"]) >= 0.85
        ]
        micro = micros[layers]
        rows.append(
            {
                "condition": condition,
                "quantized_layer_count": layers,
                "profile_repeat_count": len(profiles[layers]),
                "persistent_allocation_gib_mean": sum(resident) / len(resident) / 2**30,
                "persistent_allocation_gib_min": min(resident) / 2**30,
                "persistent_allocation_gib_max": max(resident) / 2**30,
                "peak_temporary_prefill_workspace_gib_mean": sum(workspace) / len(workspace) / 2**30,
                "peak_temporary_prefill_workspace_gib_min": min(workspace) / 2**30,
                "peak_temporary_prefill_workspace_gib_max": max(workspace) / 2**30,
                "workspace_cv": coefficient_of_variation(workspace),
                "safe_gpu_blocks_mean": sum(blocks) / len(blocks),
                "safe_gpu_blocks_min": min(blocks),
                "safe_gpu_blocks_max": max(blocks),
                "safe_gpu_blocks_cv": coefficient_of_variation(blocks),
                "safe_kv_token_slots_mean": 16 * sum(blocks) / len(blocks),
                "prefill_1024_median_ms": micro["prefill_median_ms"],
                "prefill_1024_p95_ms": micro["prefill_p95_ms"],
                "decode_one_token_median_ms": micro["decode_one_token_median_ms"],
                "decode_one_token_p95_ms": micro["decode_one_token_p95_ms"],
                "serving_saturation_nominal_rps": float(saturated[0]["nominal_offered_rps"]) if saturated else None,
                "serving_saturation_observation": "at_or_below_lowest_tested" if saturated and saturated[0] == condition_serving[0] else ("observed" if saturated else "above_highest_tested"),
            }
        )
    write_csv(output_dir / "resource_mechanism_table.csv", rows)
    return rows


def eligibility_analysis(
    quality: dict[str, Any], serving_rows: list[dict[str, Any]], mechanism_rows: list[dict[str, Any]], near_scale: float, output_dir: Path
) -> dict[str, Any]:
    resource = {row["condition"]: row for row in mechanism_rows}
    quality_table = {row["condition"]: row for row in quality["quality_table"]}
    agreements = {row["condition"]: row for row in quality["output_agreement"]}
    fp16_blocks = float(resource["fp16_0"]["safe_gpu_blocks_mean"])
    near = {
        condition: sorted(
            (row for row in serving_rows if row["condition"] == condition and float(row["time_scale"]) == near_scale),
            key=lambda row: int(row["repeat"]),
        )
        for condition in CONDITIONS
    }
    rows: list[dict[str, Any]] = [
        {
            "condition": "fp16_0",
            "classification": "REFERENCE",
            "eligible": True,
            "reason": "full-precision reference state",
        }
    ]
    eligible_quantized: list[str] = []
    for condition in CONDITIONS[1:]:
        current_resource = resource[condition]
        profiles_stable = (
            float(current_resource["safe_gpu_blocks_cv"]) <= 0.05
            and float(current_resource["workspace_cv"]) <= 0.05
        )
        complete = len(near[condition]) >= 2 and all(
            int(row["completed_request_count"]) == 64 and int(row["failed_or_incomplete_request_count"]) == 0
            for row in near[condition]
        )
        resource_relief = float(current_resource["safe_gpu_blocks_mean"]) > 1.05 * fp16_blocks
        paired_count = min(len(near["fp16_0"]), len(near[condition]))
        p95_better_each = paired_count >= 2 and all(
            float(near[condition][index]["ttft_s_p95"]) < float(near["fp16_0"][index]["ttft_s_p95"])
            for index in range(paired_count)
        )
        slo_better_each = paired_count >= 2 and all(
            float(near[condition][index]["slo_violation_percent"]) < float(near["fp16_0"][index]["slo_violation_percent"])
            for index in range(paired_count)
        )
        fp16_p95 = [float(row["ttft_s_p95"]) for row in near["fp16_0"]]
        current_p95 = [float(row["ttft_s_p95"]) for row in near[condition]]
        mean_improvement = (
            sum(fp16_p95) / len(fp16_p95) - sum(current_p95) / len(current_p95)
            if fp16_p95 and current_p95 else -math.inf
        )
        fp16_range = max(fp16_p95) - min(fp16_p95) if len(fp16_p95) >= 2 else math.inf
        current_range = max(current_p95) - min(current_p95) if len(current_p95) >= 2 else math.inf
        noise_floor = max(fp16_range, current_range)
        if noise_floor == 0 and fp16_p95:
            noise_floor = 0.05 * sum(fp16_p95) / len(fp16_p95)
        exceeds_noise = mean_improvement > noise_floor
        mean_fp16_kv = sum(float(row["peak_logical_kv_utilization"]) for row in near["fp16_0"]) / len(near["fp16_0"])
        mean_current_kv = sum(float(row["peak_logical_kv_utilization"]) for row in near[condition]) / len(near[condition])
        mechanism_alignment = resource_relief and mean_current_kv < mean_fp16_kv
        eligible = all((profiles_stable, complete, resource_relief, p95_better_each, slo_better_each, exceeds_noise, mechanism_alignment))
        failures = [
            label for label, passed in (
                ("unstable profile/workspace", profiles_stable),
                ("incomplete near-knee evidence", complete),
                ("no >5% safe-KV relief", resource_relief),
                ("P95 not better in both repeats", p95_better_each),
                ("SLO rate not better in both repeats", slo_better_each),
                ("P95 advantage does not exceed run noise", exceeds_noise),
                ("queue/KV mechanism does not align", mechanism_alignment),
            ) if not passed
        ]
        if eligible:
            eligible_quantized.append(condition)
        rows.append(
            {
                "condition": condition,
                "classification": "ELIGIBLE" if eligible else "INELIGIBLE",
                "eligible": eligible,
                "profiles_stable": profiles_stable,
                "all_near_knee_runs_complete": complete,
                "resource_relief": resource_relief,
                "p95_better_in_both_repeats": p95_better_each,
                "slo_better_in_both_repeats": slo_better_each,
                "mean_p95_improvement_s": mean_improvement,
                "p95_noise_floor_s": noise_floor,
                "advantage_exceeds_noise": exceeds_noise,
                "mechanism_alignment": mechanism_alignment,
                "safe_gpu_blocks_mean": current_resource["safe_gpu_blocks_mean"],
                "fp16_safe_gpu_blocks_mean": fp16_blocks,
                "reason": "; ".join(failures) if failures else "all predeclared gates pass",
            }
        )
    write_csv(output_dir / "state_eligibility.csv", rows)

    quality_signal = any(
        float(quality_table[condition]["paired_delta_ci95_low"]) > 0
        or float(quality_table[condition]["paired_delta_ci95_high"]) < 0
        or float(agreements[condition]["token_agreement_ci95_high"]) < 0.999
        for condition in CONDITIONS[1:]
    )
    go = bool(eligible_quantized and quality_signal)
    result = {
        "schema_version": 1,
        "decision": "GO" if go else "NO-GO",
        "recommended_runtime_state_set": ["fp16_0", *eligible_quantized] if go else [],
        "eligible_quantized_states": eligible_quantized,
        "quality_distinction_or_distortion_signal": quality_signal,
        "state_eligibility": rows,
        "decision_rule": "GO iff at least one W4 state passes every predeclared eligibility gate and quality has a paired distinction or output-agreement distortion signal",
    }
    (output_dir / "dynamic_decision.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--quality-runs", type=Path, nargs=4, required=True)
    parser.add_argument("--serving-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--profile-files", type=Path, nargs="+", required=True)
    parser.add_argument("--microbenchmark-files", type=Path, nargs=4, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    near_scale = float(manifest["serving"]["near_knee_time_scale"])
    quality = quality_analysis(args.quality_runs, args.output_dir)
    serving_rows, _ = serving_analysis(args.serving_runs, args.output_dir, near_scale)
    mechanism = mechanism_analysis(args.profile_files, args.microbenchmark_files, serving_rows, args.output_dir)
    decision = eligibility_analysis(quality, serving_rows, mechanism, near_scale, args.output_dir)
    aggregate = {
        "schema_version": 1,
        "manifest": str(args.manifest),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "quality_summary": "quality_summary.json",
        "serving_summary": "serving_summary.json",
        "resource_mechanism_table": "resource_mechanism_table.csv",
        "state_eligibility": "state_eligibility.csv",
        "dynamic_decision": decision,
        "raw_quality_runs": [str(path) for path in args.quality_runs],
        "raw_serving_runs": [str(path) for path in args.serving_runs],
        "profile_files": [str(path) for path in args.profile_files],
        "microbenchmark_files": [str(path) for path in args.microbenchmark_files],
    }
    (args.output_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
