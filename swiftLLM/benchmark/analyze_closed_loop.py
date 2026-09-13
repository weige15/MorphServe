"""Regenerate all v9 closed-loop performance, quality, controller, and decision artifacts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import re
from typing import Any, Callable

import matplotlib.pyplot as plt

from .analyze_crossover import integrate, telemetry_segments, window_stats
from .analyze_static_frontier import (
    best_reference_f1,
    bootstrap_mean_ci,
    longest_common_prefix,
    token_agreement,
)
from .metrics import derive_request_metrics, distribution, percentile


CONDITIONS = (
    "runtime_static_fp16",
    "runtime_static_awq_w4_16",
    "closed_loop_dynamic",
)
DISPLAY = {
    "runtime_static_fp16": "RUNTIME-STATIC-FP16",
    "runtime_static_awq_w4_16": "RUNTIME-STATIC-AWQ-W4-16",
    "closed_loop_dynamic": "CLOSED-LOOP-DYNAMIC",
}
COLORS = {
    "runtime_static_fp16": "#1f77b4",
    "runtime_static_awq_w4_16": "#ff7f0e",
    "closed_loop_dynamic": "#2ca02c",
}
FP16_REFERENCE_BLOCKS = 1768
RUNTIME_FP16_BLOCKS = 1759
RUNTIME_AWQ_BLOCKS = 4170


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def flatten_distribution(prefix: str, values: list[float]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in distribution(values).items()}


def numeric(rows: list[dict[str, Any]], field: str) -> list[float]:
    return [float(row[field]) for row in rows if isinstance(row.get(field), (int, float))]


def overlap_seconds(left_start: int, left_end: int, right_start: int, right_end: int) -> float:
    return max(0, min(left_end, right_end) - max(left_start, right_start)) / 1_000_000_000


def selected_run_dir(root: Path, plan_row: dict[str, Any]) -> Path:
    return root / "runs" / str(plan_row.get("actual_run_id", plan_row["run_id"]))


def transition_integrity_ok(event: dict[str, Any]) -> bool:
    trace = event.get("engine_trace") or {}
    resize = trace.get("kv_resize") or {}
    before_record = resize.get("integrity_before") or {}
    after_record = resize.get("integrity_after") or {}
    before = before_record.get("logical_digest")
    after = after_record.get("logical_digest")
    direction = event.get("direction")
    expected_precision = (
        "AWQ_MARLIN_W4_16"
        if direction == "FP16_TO_AWQ_MARLIN_W4_16"
        else "FP16"
        if direction == "AWQ_MARLIN_W4_16_TO_FP16"
        else None
    )
    expected_blocks = RUNTIME_AWQ_BLOCKS if expected_precision == "AWQ_MARLIN_W4_16" else RUNTIME_FP16_BLOCKS
    context = trace.get("engine_context_after") or {}
    active = int(trace.get("active_request_count", 0))
    digest_ok = before is not None and before == after if active > 0 else (before is None and after is None) or before == after
    return bool(
        expected_precision is not None
        and event.get("result") == "success"
        and not event.get("error")
        and trace.get("status") == "success"
        and trace.get("precision_after") == expected_precision
        and int(trace.get("physical_blocks_after", -1)) == expected_blocks
        and int(context.get("physical_blocks", -1)) == expected_blocks
        and int(context.get("scheduler_visible_blocks", -1)) == expected_blocks
        and int(resize.get("actual_total_blocks", expected_blocks)) == expected_blocks
        and digest_ok
    )


def analyze_run(run_dir: Path, plan_row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = read_json(run_dir / "metadata.json")
    requests = read_jsonl(run_dir / "requests.jsonl")
    telemetry = read_jsonl(run_dir / "telemetry.jsonl")
    batches = read_jsonl(run_dir / "batches.jsonl")
    controller = read_jsonl(run_dir / "controller.jsonl")
    transitions = read_jsonl(run_dir / "transitions.jsonl")
    initialization = read_jsonl(run_dir / "initialization_transitions.jsonl")
    repeat_match = re.search(r"-rep(\d+)$", str(metadata["run_id"]))
    repeat = int(repeat_match.group(1)) if repeat_match else -1
    expected = int(plan_row["request_count"])
    completed = [row for row in requests if row.get("status") == "completed"]
    duration = float(metadata["measurement_duration_s"])

    derived = [derive_request_metrics(row) for row in completed]
    ttft = [float(row["ttft_s"]) for row in derived if row["ttft_s"] is not None]
    queue = [float(row["queueing_delay_s"]) for row in derived if row["queueing_delay_s"] is not None]
    tpot = [float(row["tpot_s"]) for row in derived if row["tpot_s"] is not None]
    prefill_output = numeric(completed, "first_prefill_to_first_output_s")
    segments = telemetry_segments(telemetry, duration)
    waiting_area = integrate(segments, lambda value: float(value["waiting_q_depth"]))
    waiting_ge4_s = integrate(segments, lambda value: float(int(value["waiting_q_depth"]) >= 4))
    kv_ge095_s = integrate(
        segments,
        lambda value: float(float(value["fp16_equivalent_kv_utilization"]) >= 0.95),
    )
    admissions_paused_s = integrate(
        segments, lambda value: float(bool(value.get("admissions_paused")))
    )
    fp16_wall_s = integrate(
        segments, lambda value: float(value.get("runtime_precision_state") == "FP16")
    )
    awq_wall_s = integrate(
        segments,
        lambda value: float(value.get("runtime_precision_state") == "AWQ_MARLIN_W4_16"),
    )
    transition_wall_s = max(0.0, duration - fp16_wall_s - awq_wall_s)

    prefill_batches = [row for row in batches if int(row.get("num_prefill_sequences", 0)) > 0]
    forward_durations = [float(row["forward_execution_duration_s"]) for row in batches]
    arrival_jitter = numeric(requests, "arrival_jitter_s")
    hot_intersections = []
    for request in completed:
        arrival = int(request["arrival_time_ns"])
        first = int(request["first_stream_token_received_time_ns"])
        overlap = 0.0
        for event in transitions:
            start = event.get("model_transition_started_ns")
            end = event.get("model_transition_ended_ns")
            if start is not None and end is not None:
                overlap += overlap_seconds(arrival, first, int(start), int(end))
        hot_intersections.append(overlap)

    fp16_tokens = sum(int(row.get("fp16_output_step_count", 0)) for row in completed)
    awq_tokens = sum(int(row.get("awq_output_step_count", 0)) for row in completed)
    output_tokens = fp16_tokens + awq_tokens
    exposure = Counter()
    for request in completed:
        prefill = request.get("prefill_precision_state")
        awq_steps = int(request.get("awq_output_step_count", 0))
        count = int(request.get("output_token_count", 0))
        if prefill != "AWQ_MARLIN_W4_16" and awq_steps == 0:
            exposure["never"] += 1
        elif prefill == "AWQ_MARLIN_W4_16" and awq_steps == count:
            exposure["fully"] += 1
        else:
            exposure["partially"] += 1

    row: dict[str, Any] = {
        "run_id": metadata["run_id"],
        "planned_run_id": metadata["planned_run_id"],
        "attempt_index": metadata["attempt_index"],
        "workload_class": metadata["workload_class"],
        "condition": metadata["condition"],
        "display_condition": DISPLAY[str(metadata["condition"])],
        "repeat": repeat,
        "request_count": len(requests),
        "completed_request_count": len(completed),
        "measurement_duration_s": duration,
        "strict_ttft_gt_2_count": sum(value > 2.0 for value in ttft),
        "strict_ttft_gt_2_rate": sum(value > 2.0 for value in ttft) / len(ttft) if ttft else None,
        "completed_request_throughput_rps": len(completed) / duration,
        "generated_token_throughput_tps": output_tokens / duration,
        "waiting_queue_area_request_s": waiting_area,
        "waiting_ge4_dwell_s": waiting_ge4_s,
        "waiting_ge4_dwell_fraction": waiting_ge4_s / duration,
        "fp16_equiv_kv_ge095_dwell_s": kv_ge095_s,
        "fp16_equiv_kv_ge095_dwell_fraction": kv_ge095_s / duration,
        "preemption_count": max((int(value.get("preemption_count", 0)) for value in telemetry), default=0),
        "swap_in_count": max((int(value.get("swap_in_count", 0)) for value in telemetry), default=0),
        "swap_out_count": max((int(value.get("swap_out_count", 0)) for value in telemetry), default=0),
        "fp16_wall_time_fraction": fp16_wall_s / duration,
        "awq_wall_time_fraction": awq_wall_s / duration,
        "transition_or_transient_wall_time_fraction": transition_wall_s / duration,
        "fp16_forward_count": sum(row.get("precision_state") == "FP16" for row in batches),
        "awq_forward_count": sum(row.get("precision_state") == "AWQ_MARLIN_W4_16" for row in batches),
        "fp16_prefill_request_count": sum(row.get("prefill_precision_state") == "FP16" for row in completed),
        "awq_prefill_request_count": sum(row.get("prefill_precision_state") == "AWQ_MARLIN_W4_16" for row in completed),
        "fp16_output_token_count": fp16_tokens,
        "awq_output_token_count": awq_tokens,
        "awq_output_token_fraction": awq_tokens / output_tokens if output_tokens else None,
        "requests_never_exposed_to_awq": exposure["never"],
        "requests_partially_exposed_to_awq": exposure["partially"],
        "requests_fully_exposed_to_awq": exposure["fully"],
        "request_arrivals_during_hot_transition": sum(bool(value.get("arrival_during_hot_transition")) for value in completed),
        "requests_spanning_hot_transition": sum(bool(value.get("spans_hot_transition")) for value in completed),
        "transition_count": len(transitions),
        "fp16_to_awq_transition_count": sum(value.get("direction") == "FP16_TO_AWQ_MARLIN_W4_16" for value in transitions),
        "awq_to_fp16_transition_count": sum(value.get("direction") == "AWQ_MARLIN_W4_16_TO_FP16" for value in transitions),
        "hot_transition_total_s": sum(float(value.get("hot_transition_duration_s") or 0) for value in transitions),
        "transition_pending_or_drain_total_s": sum(float(value.get("pending_or_drain_duration_s") or 0) for value in transitions),
        "admissions_paused_time_s": admissions_paused_s,
        "initialization_transition_count": len(initialization),
        "controller_sample_count": len(controller),
        "controller_first_entry_s": next((float(value["elapsed_s"]) for value in controller if value.get("entry_boolean")), None),
        "controller_first_release_s": next((float(value["elapsed_s"]) for value in controller if value.get("release_boolean")), None),
        "transition_directions": "|".join(str(value.get("direction")) for value in transitions),
        "transition_requested_times_s": "|".join(f"{float(value['requested_elapsed_s']):.9f}" for value in transitions),
        "transition_started_times_s": "|".join(
            f"{(int(value['model_transition_started_ns']) - int(metadata['measurement_start_time_ns'])) / 1e9:.9f}"
            for value in transitions if value.get("model_transition_started_ns") is not None
        ),
        "transition_ended_times_s": "|".join(
            f"{(int(value['model_transition_ended_ns']) - int(metadata['measurement_start_time_ns'])) / 1e9:.9f}"
            for value in transitions if value.get("model_transition_ended_ns") is not None
        ),
    }
    row.update(flatten_distribution("ttft_s", ttft))
    row.update(flatten_distribution("arrival_jitter_s", arrival_jitter))
    row["arrival_jitter_abs_max_s"] = max((abs(value) for value in arrival_jitter), default=None)
    row.update(flatten_distribution("queueing_delay_s", queue))
    row.update(flatten_distribution("tpot_s", tpot))
    row.update(flatten_distribution("first_prefill_to_first_output_s", prefill_output))
    row.update(flatten_distribution("waiting_q_depth", numeric(telemetry, "waiting_q_depth")))
    row.update(flatten_distribution("running_q_count", numeric(telemetry, "running_q_count")))
    row.update(flatten_distribution("swapped_q_count", numeric(telemetry, "swapped_q_count")))
    row.update(flatten_distribution("physical_used_kv_blocks", numeric(telemetry, "physical_used_kv_blocks")))
    row.update(flatten_distribution("fp16_equiv_kv_utilization", numeric(telemetry, "fp16_equivalent_kv_utilization")))
    row.update(flatten_distribution("native_kv_utilization", numeric(telemetry, "native_current_state_kv_utilization")))
    row.update(flatten_distribution("forward_duration_s", forward_durations))
    row.update(flatten_distribution("prefill_sequence_count", numeric(prefill_batches, "num_prefill_sequences")))
    row.update(flatten_distribution("prefill_tokens", numeric(prefill_batches, "total_prefill_tokens")))
    row.update(flatten_distribution("prefill_effective_m", numeric(prefill_batches, "effective_gemm_m")))
    row.update(flatten_distribution("prefill_duration_s", numeric(prefill_batches, "prefill_execution_duration_s")))
    row.update(flatten_distribution("decoding_sequence_count", numeric(batches, "num_decoding_sequences")))
    row.update(flatten_distribution("ttft_hot_transition_overlap_s", hot_intersections))

    stable_capacity_ok = all(
        int(value["physical_total_kv_blocks"])
        == (RUNTIME_FP16_BLOCKS if value["runtime_precision_state"] == "FP16" else RUNTIME_AWQ_BLOCKS)
        for value in telemetry
        if value.get("runtime_precision_state") in ("FP16", "AWQ_MARLIN_W4_16")
    )

    def precision_stream_ok(request: dict[str, Any]) -> bool:
        states = list(request.get("step_precision_states", []))
        positions = list(request.get("step_input_positions", []))
        engine_ids = list(request.get("engine_request_ids", []))
        received = [int(value) for value in request.get("step_received_time_ns", [])]
        expected_transitions = [
            {
                "output_step_index": index,
                "input_position": positions[index],
                "precision_state": state,
            }
            for index, state in enumerate(states)
            if index == 0 or states[index - 1] != state
        ] if len(positions) == len(states) else []
        return bool(
            len(states) == len(positions) == len(engine_ids) == len(received) == 512
            and set(states) <= {"FP16", "AWQ_MARLIN_W4_16"}
            and all(left <= right for left, right in zip(received, received[1:]))
            and states[0] == request.get("prefill_precision_state")
            and states.count("FP16") == int(request.get("fp16_output_step_count", -1))
            and states.count("AWQ_MARLIN_W4_16") == int(request.get("awq_output_step_count", -1))
            and int(request.get("fp16_output_step_count", -1)) + int(request.get("awq_output_step_count", -1)) == 512
            and request.get("precision_state_transitions") == expected_transitions
        )

    ordered_requests = sorted(requests, key=lambda value: int(value["sequence"]))
    arrival_integrity = all(
        math.isfinite(float(value["arrival_jitter_s"]))
        and abs(float(value["arrival_jitter_s"])) <= 0.25
        and abs(
            float(value["actual_arrival_offset_s"])
            - (int(value["arrival_time_ns"]) - int(metadata["measurement_start_time_ns"])) / 1e9
        ) <= 1e-9
        and abs(
            float(value["arrival_jitter_s"])
            - (int(value["arrival_time_ns"]) - int(value["planned_arrival_time_ns"])) / 1e9
        ) <= 1e-9
        for value in ordered_requests
    ) and all(
        int(left["arrival_time_ns"]) <= int(right["arrival_time_ns"])
        for left, right in zip(ordered_requests, ordered_requests[1:])
    )
    batch_occurrences = Counter(
        request_id
        for batch in batches
        for request_id in batch.get("benchmark_request_ids", [])
        if request_id is not None
    )
    checks = {
        "plan_identity": metadata.get("run_id") == plan_row.get("actual_run_id", plan_row["run_id"])
        and metadata.get("planned_run_id") == plan_row["run_id"]
        and metadata.get("condition") == plan_row["condition"]
        and metadata.get("workload_class") == plan_row["workload_class"]
        and metadata.get("workload_sha256") == plan_row["workload_sha256"],
        "request_count": len(requests) == expected,
        "all_requests_completed": len(completed) == expected,
        "exact_lengths": all(
            int(value.get("prompt_token_count", -1)) == 1024
            and int(value.get("output_token_count", -1)) == 512
            for value in completed
        ),
        "finite_valid_token_ids": all(
            len(value.get("output_token_ids", [])) == 512
            and all(
                isinstance(token, int) and 0 <= token < int(metadata["vocab_size"])
                for token in value.get("output_token_ids", [])
            )
            for value in completed
        ),
        "arrival_schedule_integrity_and_abs_jitter_le_0p25s": arrival_integrity
        and metadata.get("arrival_lateness_tolerance_s") == 0.25,
        "engine_request_ids_stable": all(value.get("engine_request_id_stable") for value in completed),
        "positions_contiguous": all(value.get("input_positions_contiguous") for value in completed),
        "exact_per_step_precision_evidence": all(precision_stream_ok(value) for value in completed),
        "one_forward_per_output_step": all(batch_occurrences[value["benchmark_request_id"]] == 512 for value in completed),
        "telemetry_nonempty_monotonic": bool(telemetry)
        and all(float(left["elapsed_s"]) < float(right["elapsed_s"]) for left, right in zip(telemetry, telemetry[1:])),
        "controller_trace_presence": bool(controller) == (metadata["condition"] == "closed_loop_dynamic"),
        "static_controller_never_invoked": metadata["condition"] == "closed_loop_dynamic" or not transitions,
        "static_awq_initialized_before_measurement": (
            len(initialization) == 1 if metadata["condition"] == "runtime_static_awq_w4_16" else len(initialization) == 0
        ),
        "runtime_capacity_stable_states": stable_capacity_ok,
        "allocator_bounds": all(
            0 <= int(value["physical_used_kv_blocks"]) <= int(value["physical_total_kv_blocks"])
            and int(value["base_physical_kv_blocks"]) == RUNTIME_FP16_BLOCKS
            for value in telemetry
        ),
        "policy_reference_fixed_1768": all(int(value["fp16_policy_reference_blocks"]) == FP16_REFERENCE_BLOCKS for value in telemetry),
        "all_transitions_successful_and_kv_preserved": all(transition_integrity_ok(value) for value in transitions),
        "no_recorded_errors": not metadata.get("run_exception") and all(not value.get("error") for value in requests),
    }
    checks["all_pass"] = all(checks.values())
    return row, {"run_id": metadata["run_id"], "checks": checks}


def noise_threshold(rows: list[dict[str, Any]], workload: str) -> float:
    fp16 = sorted(
        [row for row in rows if row["workload_class"] == workload and row["condition"] == "runtime_static_fp16"],
        key=lambda row: int(row["repeat"]),
    )
    dynamic = sorted(
        [row for row in rows if row["workload_class"] == workload and row["condition"] == "closed_loop_dynamic"],
        key=lambda row: int(row["repeat"]),
    )
    fp16_values = [float(row["ttft_s_p95"]) for row in fp16]
    dynamic_values = [float(row["ttft_s_p95"]) for row in dynamic]
    return max(
        0.111,
        max(fp16_values) - min(fp16_values),
        max(dynamic_values) - min(dynamic_values),
        0.05 * sum(fp16_values) / len(fp16_values),
    )


def paired_comparisons(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lookup = {
        (row["workload_class"], row["condition"], int(row["repeat"])): row for row in rows
    }
    output = []
    gates: dict[str, Any] = {}
    for workload in ("low", "high", "low_high_low"):
        epsilon = noise_threshold(rows, workload)
        for repeat in (0, 1):
            fp16 = lookup[(workload, "runtime_static_fp16", repeat)]
            awq = lookup[(workload, "runtime_static_awq_w4_16", repeat)]
            dynamic = lookup[(workload, "closed_loop_dynamic", repeat)]
            output.append(
                {
                    "workload_class": workload,
                    "repeat": repeat,
                    "noise_threshold_s": epsilon,
                    "fp16_p95_ttft_s": fp16["ttft_s_p95"],
                    "awq_p95_ttft_s": awq["ttft_s_p95"],
                    "dynamic_p95_ttft_s": dynamic["ttft_s_p95"],
                    "dynamic_improvement_vs_fp16_s": float(fp16["ttft_s_p95"]) - float(dynamic["ttft_s_p95"]),
                    "dynamic_gap_vs_awq_s": float(dynamic["ttft_s_p95"]) - float(awq["ttft_s_p95"]),
                    "dynamic_within_noise_of_fp16": abs(float(dynamic["ttft_s_p95"]) - float(fp16["ttft_s_p95"])) <= epsilon,
                    "dynamic_queue_p95_lower_than_fp16": float(dynamic["queueing_delay_s_p95"]) < float(fp16["queueing_delay_s_p95"]),
                    "dynamic_waiting_area_lower_than_fp16": float(dynamic["waiting_queue_area_request_s"]) < float(fp16["waiting_queue_area_request_s"]),
                    "dynamic_kv_dwell_lower_than_fp16": float(dynamic["fp16_equiv_kv_ge095_dwell_s"]) < float(fp16["fp16_equiv_kv_ge095_dwell_s"]),
                    "dynamic_preemptions_no_greater_than_fp16": int(dynamic["preemption_count"]) <= int(fp16["preemption_count"]),
                    "dynamic_throughput_at_least_98pct_fp16": float(dynamic["completed_request_throughput_rps"]) >= 0.98 * float(fp16["completed_request_throughput_rps"]),
                }
            )
    low_rows = [row for row in output if row["workload_class"] == "low"]
    high_rows = [row for row in output if row["workload_class"] == "high"]
    high_dynamic = [lookup[("high", "closed_loop_dynamic", repeat)] for repeat in (0, 1)]
    high_fp16 = [lookup[("high", "runtime_static_fp16", repeat)] for repeat in (0, 1)]
    gates["low_no_false_entry_both"] = all(
        lookup[("low", "closed_loop_dynamic", repeat)]["transition_count"] == 0 for repeat in (0, 1)
    )
    gates["low_fp16_like_within_noise_both"] = all(row["dynamic_within_noise_of_fp16"] for row in low_rows)
    gates["high_enters_awq_both"] = all(row["fp16_to_awq_transition_count"] >= 1 for row in high_dynamic)
    gates["high_improvement_exceeds_noise_both"] = all(
        float(row["dynamic_improvement_vs_fp16_s"]) > float(row["noise_threshold_s"])
        for row in high_rows
    )
    gates["high_queue_area_kv_aligned_both"] = all(
        row["dynamic_queue_p95_lower_than_fp16"]
        and row["dynamic_waiting_area_lower_than_fp16"]
        and row["dynamic_kv_dwell_lower_than_fp16"]
        for row in high_rows
    )
    gates["high_preemption_aligned"] = all(row["dynamic_preemptions_no_greater_than_fp16"] for row in high_rows) and any(
        int(dynamic["preemption_count"]) < int(fp16["preemption_count"])
        for dynamic, fp16 in zip(high_dynamic, high_fp16)
    )
    gates["high_throughput_gate"] = all(row["dynamic_throughput_at_least_98pct_fp16"] for row in high_rows) and (
        sum(float(row["completed_request_throughput_rps"]) for row in high_dynamic)
        >= sum(float(row["completed_request_throughput_rps"]) for row in high_fp16)
    )
    return output, gates


def quality_analysis(root: Path, plan: list[dict[str, Any]], output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    repeat_rows: list[dict[str, Any]] = []
    by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    for plan_row in plan:
        if plan_row["workload_class"] != "high":
            continue
        requests = read_jsonl(selected_run_dir(root, plan_row) / "requests.jsonl")
        values = []
        for request in requests:
            f1 = best_reference_f1(
                str(request.get("generated_answer") or ""),
                [str(value) for value in request.get("dataset_reference_answers", [])],
            )
            request["f1"] = f1
            values.append(f1)
            by_key[(str(plan_row["condition"]), int(plan_row["repeat"]), str(request["benchmark_request_id"]))] = request
        repeat_rows.append(
            {
                "condition": plan_row["condition"],
                "display_condition": DISPLAY[plan_row["condition"]],
                "repeat": plan_row["repeat"],
                "request_count": len(values),
                "f1_percent": 100 * sum(values) / len(values),
                "awq_output_token_fraction": sum(int(value.get("awq_output_step_count", 0)) for value in requests) / (len(requests) * 512),
            }
        )

    paired_rows: list[dict[str, Any]] = []
    request_ids = sorted({key[2] for key in by_key})
    for repeat in (0, 1):
        for request_id in request_ids:
            fp16 = by_key[("runtime_static_fp16", repeat, request_id)]
            awq = by_key[("runtime_static_awq_w4_16", repeat, request_id)]
            dynamic = by_key[("closed_loop_dynamic", repeat, request_id)]
            fp16_ids = [int(value) for value in fp16["output_token_ids"]]
            row = {
                "repeat": repeat,
                "request_id": request_id,
                "dataset_question_id": fp16.get("dataset_question_id"),
                "fp16_f1": fp16["f1"],
                "awq_f1": awq["f1"],
                "dynamic_f1": dynamic["f1"],
                "dynamic_minus_awq_f1": float(dynamic["f1"]) - float(awq["f1"]),
                "dynamic_minus_fp16_f1": float(dynamic["f1"]) - float(fp16["f1"]),
                "awq_token_agreement_vs_fp16": token_agreement(fp16_ids, awq["output_token_ids"]),
                "dynamic_token_agreement_vs_fp16": token_agreement(fp16_ids, dynamic["output_token_ids"]),
                "awq_common_prefix_vs_fp16": longest_common_prefix(fp16_ids, awq["output_token_ids"]),
                "dynamic_common_prefix_vs_fp16": longest_common_prefix(fp16_ids, dynamic["output_token_ids"]),
                "awq_exact_match_vs_fp16": int(fp16_ids == awq["output_token_ids"]),
                "dynamic_exact_match_vs_fp16": int(fp16_ids == dynamic["output_token_ids"]),
                "dynamic_awq_token_fraction": dynamic["awq_output_step_fraction"],
                "dynamic_prefill_precision": dynamic["prefill_precision_state"],
            }
            paired_rows.append(row)

    # The 64 questions, not 128 repeated executions, are the independent units.
    unique_rows: list[dict[str, Any]] = []
    for request_id in request_ids:
        matched = [row for row in paired_rows if row["request_id"] == request_id]
        aggregate = {"request_id": request_id, "repeat_count": len(matched)}
        for field in (
            "fp16_f1",
            "awq_f1",
            "dynamic_f1",
            "dynamic_minus_awq_f1",
            "dynamic_minus_fp16_f1",
            "awq_token_agreement_vs_fp16",
            "dynamic_token_agreement_vs_fp16",
            "awq_common_prefix_vs_fp16",
            "dynamic_common_prefix_vs_fp16",
            "awq_exact_match_vs_fp16",
            "dynamic_exact_match_vs_fp16",
            "dynamic_awq_token_fraction",
        ):
            aggregate[field] = sum(float(row[field]) for row in matched) / len(matched)
        unique_rows.append(aggregate)

    summary_rows = []
    field_for = {
        "runtime_static_fp16": "fp16_f1",
        "runtime_static_awq_w4_16": "awq_f1",
        "closed_loop_dynamic": "dynamic_f1",
    }
    for index, condition in enumerate(CONDITIONS):
        values = [float(row[field_for[condition]]) for row in unique_rows]
        ci = bootstrap_mean_ci(values, seed=9000 + index)
        delta_awq = [float(row["dynamic_minus_awq_f1"]) for row in unique_rows] if condition == "closed_loop_dynamic" else []
        delta_fp16 = [float(row["dynamic_minus_fp16_f1"]) for row in unique_rows] if condition == "closed_loop_dynamic" else []
        delta_awq_ci = bootstrap_mean_ci(delta_awq, seed=9100) if delta_awq else (None, None)
        delta_fp16_ci = bootstrap_mean_ci(delta_fp16, seed=9200) if delta_fp16 else (None, None)
        matching_pairs = [row for row in paired_rows if condition != "runtime_static_fp16"]
        prefix_field = "dynamic_common_prefix_vs_fp16" if condition == "closed_loop_dynamic" else "awq_common_prefix_vs_fp16"
        agreement_field = "dynamic_token_agreement_vs_fp16" if condition == "closed_loop_dynamic" else "awq_token_agreement_vs_fp16"
        exact_field = "dynamic_exact_match_vs_fp16" if condition == "closed_loop_dynamic" else "awq_exact_match_vs_fp16"
        summary_rows.append(
            {
                "condition": condition,
                "display_condition": DISPLAY[condition],
                "independent_question_count": len(values),
                "serving_repeat_count": 2,
                "aggregate_definition": "per-question mean across two serving repeats, then macro mean across 64 unique questions",
                "f1_percent": 100 * sum(values) / len(values),
                "f1_ci95_low_percent": 100 * float(ci[0]),
                "f1_ci95_high_percent": 100 * float(ci[1]),
                "dynamic_minus_awq_f1_percentage_points": 100 * sum(delta_awq) / len(delta_awq) if delta_awq else None,
                "dynamic_minus_awq_ci95_low_pp": 100 * float(delta_awq_ci[0]) if delta_awq else None,
                "dynamic_minus_awq_ci95_high_pp": 100 * float(delta_awq_ci[1]) if delta_awq else None,
                "dynamic_minus_fp16_f1_percentage_points": 100 * sum(delta_fp16) / len(delta_fp16) if delta_fp16 else None,
                "dynamic_minus_fp16_ci95_low_pp": 100 * float(delta_fp16_ci[0]) if delta_fp16 else None,
                "dynamic_minus_fp16_ci95_high_pp": 100 * float(delta_fp16_ci[1]) if delta_fp16 else None,
                "mean_position_aligned_token_agreement_vs_fp16": (
                    1.0 if condition == "runtime_static_fp16" else sum(float(row[agreement_field]) for row in matching_pairs) / len(matching_pairs)
                ),
                "mean_common_prefix_tokens_vs_fp16": (
                    512.0 if condition == "runtime_static_fp16" else sum(float(row[prefix_field]) for row in matching_pairs) / len(matching_pairs)
                ),
                "exact_output_match_rate_vs_fp16": (
                    1.0 if condition == "runtime_static_fp16" else sum(float(row[exact_field]) for row in matching_pairs) / len(matching_pairs)
                ),
                "metric_role": "DuReader character-overlap F1 is semantic quality; token agreement/prefix/exact match are fidelity mechanism evidence only",
            }
        )
    quality_decision = {
        "dynamic_minus_awq_paired_unique_question_ci95_low_pp": next(
            row["dynamic_minus_awq_ci95_low_pp"] for row in summary_rows if row["condition"] == "closed_loop_dynamic"
        ),
    }
    quality_decision["dynamic_semantically_better_than_awq"] = (
        float(quality_decision["dynamic_minus_awq_paired_unique_question_ci95_low_pp"]) > 0
    )
    write_csv(output_dir / "quality_repeat_values.csv", repeat_rows)
    write_csv(output_dir / "quality_request_pairs.csv", paired_rows)
    write_csv(output_dir / "quality_unique_question_aggregate.csv", unique_rows)
    write_csv(output_dir / "quality_fidelity_summary.csv", summary_rows)
    return repeat_rows, summary_rows, quality_decision


def controller_audit(root: Path, plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for plan_row in plan:
        run_dir = selected_run_dir(root, plan_row)
        telemetry = read_jsonl(run_dir / "telemetry.jsonl")
        controller = read_jsonl(run_dir / "controller.jsonl")
        transitions = read_jsonl(run_dir / "transitions.jsonl")
        first_logged = next((float(row["elapsed_s"]) for row in controller if row.get("entry_boolean")), None)
        first_release = next((float(row["elapsed_s"]) for row in controller if row.get("release_boolean")), None)
        output.append(
            {
                "run_id": plan_row["run_id"],
                "workload_class": plan_row["workload_class"],
                "condition": plan_row["condition"],
                "repeat": plan_row["repeat"],
                "controller_enabled": plan_row["condition"] == "closed_loop_dynamic",
                "telemetry_samples": len(telemetry),
                "controller_samples": len(controller),
                "sample_count_parity": len(controller) == len(telemetry) if plan_row["condition"] == "closed_loop_dynamic" else len(controller) == 0,
                "first_entry_condition_s": first_logged,
                "first_release_condition_s": first_release,
                "transition_count": len(transitions),
                "transition_directions": "|".join(str(row.get("direction")) for row in transitions),
                "first_transition_request_s": transitions[0].get("requested_elapsed_s") if transitions else None,
                "entry_log_request_same_tick": (
                    abs(float(transitions[0]["requested_elapsed_s"]) - first_logged) <= 0.25
                    if transitions and first_logged is not None else first_logged is None
                ),
                "all_transition_results_success": all(row.get("result") == "success" for row in transitions),
                "all_transition_kv_integrity": all(transition_integrity_ok(row) for row in transitions),
                "max_sampling_jitter_s": max((abs(float(row.get("sampling_jitter_s", 0))) for row in telemetry), default=0),
                "policy_reference_values": "|".join(map(str, sorted({row.get("fp16_policy_reference_blocks") for row in telemetry}))),
            }
        )
    return output


def phased_gates(root: Path, plan: list[dict[str, Any]]) -> dict[str, Any]:
    metadata = read_json(root / "input" / "workload_metadata.json")
    high_start = float(metadata["phased_boundaries"]["high"]["start_offset_s"])
    recovery_start = float(metadata["phased_boundaries"]["low_recovery"]["start_offset_s"])
    recovery_last = float(metadata["phased_boundaries"]["low_recovery"]["last_arrival_offset_s"])
    repeats = []
    for repeat in (0, 1):
        plan_row = next(
            row for row in plan
            if row["workload_class"] == "low_high_low"
            and row["condition"] == "closed_loop_dynamic"
            and int(row["repeat"]) == repeat
        )
        run_dir = selected_run_dir(root, plan_row)
        metadata_run = read_json(run_dir / "metadata.json")
        transitions = read_jsonl(run_dir / "transitions.jsonl")
        requests = read_jsonl(run_dir / "requests.jsonl")
        directions = [row.get("direction") for row in transitions]
        requested = [float(row["requested_elapsed_s"]) for row in transitions]
        ended = [
            (int(row["model_transition_ended_ns"]) - int(metadata_run["measurement_start_time_ns"])) / 1e9
            for row in transitions if row.get("model_transition_ended_ns") is not None
        ]
        restore_end = ended[1] if len(ended) >= 2 else None
        post_restore = [
            row for row in requests
            if restore_end is not None
            and row.get("phase") == "low_recovery"
            and float(row.get("actual_arrival_offset_s", -1)) > restore_end
            and row.get("status") == "completed"
        ]
        repeats.append(
            {
                "repeat": repeat,
                "directions": directions,
                "requested_s": requested,
                "ended_s": ended,
                "low_phase_remained_fp16": not requested or requested[0] >= high_start,
                "entry_during_or_after_high_start": len(requested) >= 1 and requested[0] >= high_start,
                "restore_requested_after_recovery_start": len(requested) >= 2 and requested[1] >= recovery_start,
                "restore_completed_before_last_recovery_arrival": restore_end is not None and restore_end < recovery_last,
                "completed_recovery_requests_arriving_after_restore": len(post_restore),
                "exactly_one_ordered_roundtrip": directions == [
                    "FP16_TO_AWQ_MARLIN_W4_16",
                    "AWQ_MARLIN_W4_16_TO_FP16",
                ],
                "all_192_completed": len([row for row in requests if row.get("status") == "completed"]) == 192,
                "all_positions_contiguous": all(row.get("input_positions_contiguous") for row in requests),
            }
        )
    return {
        "phase_boundaries": {
            "high_start_s": high_start,
            "recovery_start_s": recovery_start,
            "recovery_last_arrival_s": recovery_last,
        },
        "repeats": repeats,
        "both_repeats_pass": all(
            row["low_phase_remained_fp16"]
            and row["entry_during_or_after_high_start"]
            and row["restore_requested_after_recovery_start"]
            and row["restore_completed_before_last_recovery_arrival"]
            and row["completed_recovery_requests_arriving_after_restore"] > 0
            and row["exactly_one_ordered_roundtrip"]
            and row["all_192_completed"]
            and row["all_positions_contiguous"]
            for row in repeats
        ),
    }


def transition_rows(root: Path, plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for plan_row in plan:
        run_dir = selected_run_dir(root, plan_row)
        metadata = read_json(run_dir / "metadata.json")
        requests = read_jsonl(run_dir / "requests.jsonl")
        for event in read_jsonl(run_dir / "transitions.jsonl"):
            trace = event.get("engine_trace") or {}
            start = event.get("model_transition_started_ns")
            end = event.get("model_transition_ended_ns")
            affected = 0
            arrivals = 0
            for request in requests:
                if start is None or end is None:
                    continue
                arrival = int(request["arrival_time_ns"])
                completion = int(request["completion_time_ns"])
                arrivals += int(int(start) <= arrival <= int(end))
                affected += int(max(arrival, int(start)) <= min(completion, int(end)))
            resize = trace.get("kv_resize") or {}
            output.append(
                {
                    "run_id": plan_row["run_id"],
                    "workload_class": plan_row["workload_class"],
                    "repeat": plan_row["repeat"],
                    "controller_transition_id": event["controller_transition_id"],
                    "direction": event["direction"],
                    "requested_elapsed_s": event["requested_elapsed_s"],
                    "started_elapsed_s": ((int(start) - int(metadata["measurement_start_time_ns"])) / 1e9 if start is not None else None),
                    "ended_elapsed_s": ((int(end) - int(metadata["measurement_start_time_ns"])) / 1e9 if end is not None else None),
                    "pending_or_drain_duration_s": event.get("pending_or_drain_duration_s"),
                    "hot_transition_duration_s": event.get("hot_transition_duration_s"),
                    "request_to_completion_duration_s": event.get("request_to_completion_duration_s"),
                    "boundary_cuda_sync_s": float(trace.get("boundary_cuda_sync_ns", 0)) / 1e9,
                    "kv_resize_s": float(resize.get("elapsed_ns", 0)) / 1e9,
                    "kv_remapped_blocks": resize.get("remapped_blocks", 0),
                    "kv_d2d_bytes": resize.get("d2d_bytes", 0),
                    "h2d_bytes": (trace.get("weight_transition") or {}).get("h2d_bytes"),
                    "requests_whose_service_interval_spans_transition": affected,
                    "arrivals_during_hot_transition": arrivals,
                    "transition_peak_allocated_bytes": trace.get("transition_peak_allocated_bytes"),
                    "memory_after_allocated_bytes": (trace.get("memory_after") or {}).get("allocated_bytes"),
                    "status": event.get("result"),
                    "kv_integrity": transition_integrity_ok(event),
                }
            )
    return output


def headline_table(run_rows: list[dict[str, Any]], quality_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    quality = {row["condition"]: row for row in quality_rows}
    table = []
    for condition in CONDITIONS:
        rows = sorted(
            [row for row in run_rows if row["workload_class"] == "high" and row["condition"] == condition],
            key=lambda row: int(row["repeat"]),
        )
        mean = lambda field: sum(float(row[field]) for row in rows) / len(rows)
        table.append(
            {
                "condition": DISPLAY[condition],
                "f1_percent": quality[condition]["f1_percent"],
                "p95_ttft_s_mean": mean("ttft_s_p95"),
                "p95_ttft_s_repeats": "|".join(f"{float(row['ttft_s_p95']):.6f}" for row in rows),
                "strict_ttft_gt_2_rate_mean": mean("strict_ttft_gt_2_rate"),
                "p99_ttft_s_mean": mean("ttft_s_p99"),
                "p95_queueing_s_mean": mean("queueing_delay_s_p95"),
                "preemptions_mean": mean("preemption_count"),
                "fp16_wall_time_fraction_mean": mean("fp16_wall_time_fraction"),
                "awq_wall_time_fraction_mean": mean("awq_wall_time_fraction"),
                "awq_output_token_fraction_mean": mean("awq_output_token_fraction"),
                "transitions_mean": mean("transition_count"),
            }
        )
    return table


def plot_timeline(root: Path, plan: list[dict[str, Any]], workload: str, output: Path) -> None:
    plan_row = next(
        row for row in plan
        if row["workload_class"] == workload
        and row["condition"] == "closed_loop_dynamic"
        and int(row["repeat"]) == 0
    )
    run_dir = selected_run_dir(root, plan_row)
    telemetry = read_jsonl(run_dir / "telemetry.jsonl")
    requests = read_jsonl(run_dir / "requests.jsonl")
    transitions = read_jsonl(run_dir / "transitions.jsonl")
    metadata = read_json(run_dir / "metadata.json")
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].scatter(
        [row["actual_arrival_offset_s"] for row in requests],
        [row["ttft_s"] for row in requests],
        s=13,
        alpha=0.75,
    )
    axes[0].axhline(2.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("TTFT (s)")
    axes[1].step([row["elapsed_s"] for row in telemetry], [row["waiting_q_depth"] for row in telemetry], where="post")
    axes[1].set_ylabel("Waiting")
    axes[2].step([row["elapsed_s"] for row in telemetry], [row["fp16_equivalent_kv_utilization"] for row in telemetry], where="post", label="FP16-equivalent /1768")
    axes[2].step([row["elapsed_s"] for row in telemetry], [row["native_current_state_kv_utilization"] for row in telemetry], where="post", label="native", alpha=0.75)
    axes[2].axhline(0.95, color="black", linestyle="--", linewidth=1)
    axes[2].set_ylabel("KV utilization")
    axes[2].legend(loc="upper right")
    state_value = {"FP16": 0, "MORPHING_TO_AWQ": 1, "AWQ_MARLIN_W4_16": 2, "RESTORING_TO_FP16": 1}
    axes[3].step([row["elapsed_s"] for row in telemetry], [state_value.get(row["runtime_precision_state"], 1) for row in telemetry], where="post")
    axes[3].set_yticks([0, 1, 2], ["FP16", "transition", "AWQ"])
    axes[3].set_ylabel("State")
    axes[3].set_xlabel("Measured elapsed time (s)")
    for event in transitions:
        start = (int(event["model_transition_started_ns"]) - int(metadata["measurement_start_time_ns"])) / 1e9
        end = (int(event["model_transition_ended_ns"]) - int(metadata["measurement_start_time_ns"])) / 1e9
        for axis in axes:
            axis.axvspan(start, end, color="red", alpha=0.14)
            axis.axvline(float(event["requested_elapsed_s"]), color="red", linestyle=":", linewidth=1)
    fig.suptitle(f"Closed-loop dynamic {workload} repeat 0 (red: request/hot transition)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    output_dir = root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = read_json(root / "run-plan.json")["runs"]
    status = read_json(root / "execution_status.json")
    selected = status.get("selected_runs", {})
    if set(selected) != {row["run_id"] for row in plan}:
        raise ValueError("execution status does not select exactly one valid attempt per planned cell")
    plan = [dict(row, actual_run_id=selected[row["run_id"]]) for row in plan]
    run_rows = []
    validations = []
    for plan_row in plan:
        row, validation = analyze_run(selected_run_dir(root, plan_row), plan_row)
        run_rows.append(row)
        validations.append(validation)
    run_rows.sort(key=lambda row: (str(row["workload_class"]), str(row["condition"]), int(row["repeat"])))
    write_csv(output_dir / "serving_runs.csv", run_rows)
    (output_dir / "run_validation.json").write_text(
        json.dumps({"runs": validations, "all_pass": all(row["checks"]["all_pass"] for row in validations)}, indent=2, sort_keys=True) + "\n"
    )

    comparisons, performance_gates = paired_comparisons(run_rows)
    write_csv(output_dir / "paired_performance.csv", comparisons)
    repeat_quality, quality_rows, quality_gate = quality_analysis(root, plan, output_dir)
    controller_rows = controller_audit(root, plan)
    write_csv(output_dir / "controller_trigger_audit.csv", controller_rows)
    transitions = transition_rows(root, plan)
    write_csv(output_dir / "transition_cost_contribution.csv", transitions)
    phased = phased_gates(root, plan)
    (output_dir / "phased_reversibility.json").write_text(
        json.dumps(phased, indent=2, sort_keys=True) + "\n"
    )

    headline = headline_table(run_rows, quality_rows)
    # Attach the already-generated repeat-level F1 values without treating them as independent CIs.
    repeat_lookup = defaultdict(list)
    for row in repeat_quality:
        repeat_lookup[row["condition"]].append((int(row["repeat"]), float(row["f1_percent"])))
    for row in headline:
        condition = next(key for key, value in DISPLAY.items() if value == row["condition"])
        row["f1_repeat_values_percent"] = "|".join(
            f"{value:.6f}" for _, value in sorted(repeat_lookup[condition])
        )
    write_csv(output_dir / "headline_table5.csv", headline)

    dynamic_runs = [row for row in run_rows if row["condition"] == "closed_loop_dynamic"]
    all_runs_valid = all(row["checks"]["all_pass"] for row in validations)
    all_dynamic_transitions_valid = all(
        row["all_transition_results_success"] and row["all_transition_kv_integrity"]
        for row in controller_rows if row["controller_enabled"]
    )
    no_oscillation = all(int(row["transition_count"]) <= 2 for row in dynamic_runs)
    # Equal-state restored FP16 allocated memory across phased fresh-process repeats.
    phased_restore_memory = [
        int(row["memory_after_allocated_bytes"])
        for row in transitions
        if row["workload_class"] == "low_high_low"
        and row["direction"] == "AWQ_MARLIN_W4_16_TO_FP16"
        and row.get("memory_after_allocated_bytes") is not None
    ]
    hbm_spread = max(phased_restore_memory) - min(phased_restore_memory) if len(phased_restore_memory) == 2 else None
    no_meaningful_hbm_drift = hbm_spread is not None and hbm_spread <= 64 * 1024 * 1024
    systems_checks = {
        "all_18_runs_valid": all_runs_valid,
        **performance_gates,
        "all_dynamic_transitions_successful_and_kv_preserved": all_dynamic_transitions_valid,
        "phased_reversibility_both_repeats": phased["both_repeats_pass"],
        "no_closed_loop_oscillation": no_oscillation,
        "restored_fp16_hbm_repeat_spread_le_64mib": no_meaningful_hbm_drift,
    }
    systems_go = all(systems_checks.values())
    quality_decision = (
        "STRONG GO"
        if systems_go and quality_gate["dynamic_semantically_better_than_awq"]
        else "SUPPORTED / UNCERTAIN"
        if systems_go
        else "NO-GO"
    )
    decision = {
        "schema_version": 1,
        "predeclared_noise_formula": "max(0.111 s, static-FP16 two-repeat P95 range, Dynamic two-repeat P95 range, 5% of static-FP16 P95 mean), separately per workload",
        "performance_gates": performance_gates,
        "systems_checks": systems_checks,
        "restored_fp16_allocated_hbm_repeat_spread_bytes": hbm_spread,
        "quality_gate": quality_gate,
        "systems_adaptation_decision": "GO" if systems_go else "NO-GO",
        "quality_latency_tradeoff_decision": quality_decision,
    }
    (output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    plot_timeline(root, plan, "high", output_dir / "headline_latency_queue_kv_state_timeline.png")
    plot_timeline(root, plan, "low_high_low", output_dir / "phased_controller_timeline.png")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
