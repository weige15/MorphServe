"""Regenerate release-side v10 metrics, transition timelines, gates, and decision."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
from typing import Any

from .analyze_closed_loop import analyze_run
from .analyze_crossover import counter_value_at, integrate, telemetry_segments
from .analyze_static_frontier import best_reference_f1
from .metrics import derive_request_metrics, percentile


FP16_BASE_BLOCKS = 1759
CONDITIONS = (
    "runtime_static_fp16",
    "runtime_static_awq_w4_16",
    "closed_loop_dynamic",
)
ALTERNATING = ("heldout_one_cycle", "heldout_two_cycle")
ENTRY = "FP16_TO_AWQ_MARLIN_W4_16"
RESTORE = "AWQ_MARLIN_W4_16_TO_FP16"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_csv(
    path: Path, rows: list[dict[str, Any]], empty_columns: tuple[str, ...] = ()
) -> None:
    if not rows:
        if not empty_columns:
            raise ValueError(f"cannot write empty table without a schema: {path}")
        path.write_text(",".join(empty_columns) + "\n", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def selected_plan(root: Path) -> list[dict[str, Any]]:
    plan = read_json(root / "run-plan.json")["runs"]
    selected = read_json(root / "execution_status.json")["selected_runs"]
    return [dict(row, actual_run_id=selected[row["run_id"]]) for row in plan]


def run_dir(root: Path, row: dict[str, Any]) -> Path:
    return root / "runs" / str(row["actual_run_id"])


def boundaries_for(
    workload_metadata: dict[str, Any], workload: str
) -> list[dict[str, Any]]:
    return list(workload_metadata["phase_boundaries"][workload])


def phase_at(boundaries: list[dict[str, Any]], timestamp_s: float) -> dict[str, Any]:
    selected = boundaries[0]
    for boundary in boundaries:
        if float(boundary["start_offset_s"]) > timestamp_s:
            break
        selected = boundary
    return selected


def ns_elapsed(timestamp_ns: int | None, measurement_start_ns: int) -> float | None:
    return (
        None
        if timestamp_ns is None
        else (int(timestamp_ns) - measurement_start_ns) / 1_000_000_000
    )


def numeric_summary(prefix: str, rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [float(row[field]) for row in rows if isinstance(row.get(field), (int, float))]
    return {
        f"{prefix}_count": len(values),
        f"{prefix}_mean": sum(values) / len(values) if values else None,
        f"{prefix}_min": min(values) if values else None,
        f"{prefix}_max": max(values) if values else None,
    }


def phase_metrics(
    root: Path,
    plan: list[dict[str, Any]],
    workload_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for plan_row in plan:
        directory = run_dir(root, plan_row)
        metadata = read_json(directory / "metadata.json")
        requests = read_jsonl(directory / "requests.jsonl")
        telemetry = read_jsonl(directory / "telemetry.jsonl")
        transitions = read_jsonl(directory / "transitions.jsonl")
        duration = float(metadata["measurement_duration_s"])
        segments = telemetry_segments(telemetry, duration)
        boundaries = boundaries_for(workload_metadata, str(plan_row["workload_class"]))
        for index, boundary in enumerate(boundaries):
            phase_index = int(boundary["phase_index"])
            phase_requests = [
                row for row in requests if int(row.get("phase_index", -1)) == phase_index
            ]
            derived = [derive_request_metrics(row) for row in phase_requests]
            ttft = [float(row["ttft_s"]) for row in derived if row["ttft_s"] is not None]
            queue = [
                float(row["queueing_delay_s"])
                for row in derived
                if row["queueing_delay_s"] is not None
            ]
            start = float(boundary["start_offset_s"])
            end = (
                float(boundaries[index + 1]["start_offset_s"])
                if index + 1 < len(boundaries)
                else duration
            )
            transition_subset = [
                event
                for event in transitions
                if start <= float(event["requested_elapsed_s"]) < end
            ]
            output.append(
                {
                    "run_id": plan_row["run_id"],
                    "actual_run_id": plan_row["actual_run_id"],
                    "workload_class": plan_row["workload_class"],
                    "condition": plan_row["condition"],
                    "repeat": plan_row["repeat"],
                    "phase_index": phase_index,
                    "phase": boundary["phase"],
                    "regime": boundary["regime"],
                    "phase_start_s": start,
                    "phase_interval_end_s": end,
                    "phase_last_arrival_s": boundary["last_arrival_offset_s"],
                    "request_count": len(phase_requests),
                    "completed_request_count": sum(
                        row.get("status") == "completed" for row in phase_requests
                    ),
                    "ttft_p50_s": percentile(ttft, 0.50),
                    "ttft_p95_s": percentile(ttft, 0.95),
                    "ttft_p99_s": percentile(ttft, 0.99),
                    "strict_ttft_gt_2_rate": (
                        sum(value > 2.0 for value in ttft) / len(ttft) if ttft else None
                    ),
                    "queue_p95_s": percentile(queue, 0.95),
                    "waiting_queue_area_request_s": integrate(
                        segments,
                        lambda row: float(row["waiting_q_depth"]),
                        start=start,
                        end=end,
                    ),
                    "preemption_count": counter_value_at(
                        telemetry, "preemption_count", end
                    )
                    - counter_value_at(telemetry, "preemption_count", start),
                    "entry_request_count": sum(
                        event.get("direction") == ENTRY for event in transition_subset
                    ),
                    "release_request_count": sum(
                        event.get("direction") == RESTORE for event in transition_subset
                    ),
                    "fp16_prefill_request_count": sum(
                        row.get("prefill_precision_state") == "FP16"
                        for row in phase_requests
                    ),
                    "awq_prefill_request_count": sum(
                        row.get("prefill_precision_state") == "AWQ_MARLIN_W4_16"
                        for row in phase_requests
                    ),
                    "fp16_output_token_count": sum(
                        int(row.get("fp16_output_step_count", 0))
                        for row in phase_requests
                    ),
                    "awq_output_token_count": sum(
                        int(row.get("awq_output_step_count", 0))
                        for row in phase_requests
                    ),
                }
            )
    return output


def phase_noise(rows: list[dict[str, Any]], workload: str, phase_index: int) -> float:
    values: dict[str, list[float]] = {}
    for condition in ("runtime_static_fp16", "closed_loop_dynamic"):
        values[condition] = [
            float(row["ttft_p95_s"])
            for row in rows
            if row["workload_class"] == workload
            and int(row["phase_index"]) == phase_index
            and row["condition"] == condition
        ]
    fp16 = values["runtime_static_fp16"]
    dynamic = values["closed_loop_dynamic"]
    if len(fp16) != 2 or len(dynamic) != 2:
        raise ValueError(f"missing two-repeat phase noise inputs: {workload} p{phase_index}")
    return max(
        0.111,
        max(fp16) - min(fp16),
        max(dynamic) - min(dynamic),
        0.05 * sum(fp16) / len(fp16),
    )


def matched_phase_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (
            row["workload_class"],
            row["condition"],
            int(row["repeat"]),
            int(row["phase_index"]),
        ): row
        for row in rows
    }
    output: list[dict[str, Any]] = []
    for dynamic in rows:
        if dynamic["condition"] != "closed_loop_dynamic":
            continue
        workload = str(dynamic["workload_class"])
        repeat = int(dynamic["repeat"])
        phase_index = int(dynamic["phase_index"])
        fp16 = lookup[(workload, "runtime_static_fp16", repeat, phase_index)]
        awq = lookup.get(
            (workload, "runtime_static_awq_w4_16", repeat, phase_index)
        )
        epsilon = phase_noise(rows, workload, phase_index)
        output.append(
            {
                "workload_class": workload,
                "repeat": repeat,
                "phase_index": phase_index,
                "phase": dynamic["phase"],
                "regime": dynamic["regime"],
                "noise_threshold_s": epsilon,
                "fp16_p95_ttft_s": fp16["ttft_p95_s"],
                "dynamic_p95_ttft_s": dynamic["ttft_p95_s"],
                "awq_p95_ttft_s": None if awq is None else awq["ttft_p95_s"],
                "dynamic_improvement_vs_fp16_s": float(fp16["ttft_p95_s"])
                - float(dynamic["ttft_p95_s"]),
                "dynamic_gap_vs_awq_s": (
                    None
                    if awq is None
                    else float(dynamic["ttft_p95_s"]) - float(awq["ttft_p95_s"])
                ),
                "dynamic_within_noise_of_fp16": abs(
                    float(dynamic["ttft_p95_s"]) - float(fp16["ttft_p95_s"])
                )
                <= epsilon,
                "dynamic_queue_p95_s": dynamic["queue_p95_s"],
                "fp16_queue_p95_s": fp16["queue_p95_s"],
                "awq_queue_p95_s": None if awq is None else awq["queue_p95_s"],
                "dynamic_waiting_area_request_s": dynamic[
                    "waiting_queue_area_request_s"
                ],
                "fp16_waiting_area_request_s": fp16[
                    "waiting_queue_area_request_s"
                ],
                "awq_waiting_area_request_s": (
                    None if awq is None else awq["waiting_queue_area_request_s"]
                ),
                "dynamic_preemptions": dynamic["preemption_count"],
                "fp16_preemptions": fp16["preemption_count"],
                "awq_preemptions": None if awq is None else awq["preemption_count"],
                "dynamic_entry_requests": dynamic["entry_request_count"],
                "dynamic_release_requests": dynamic["release_request_count"],
                "high_mechanism_gate": (
                    dynamic["regime"] == "high"
                    and float(fp16["ttft_p95_s"]) - float(dynamic["ttft_p95_s"])
                    > epsilon
                    and float(dynamic["waiting_queue_area_request_s"])
                    < float(fp16["waiting_queue_area_request_s"])
                    and int(dynamic["preemption_count"])
                    <= int(fp16["preemption_count"])
                    and int(dynamic["entry_request_count"]) == 1
                    and int(dynamic["release_request_count"]) == 0
                ),
                "obviously_worse_than_both_controls": (
                    False
                    if awq is None
                    else float(dynamic["ttft_p95_s"])
                    > float(fp16["ttft_p95_s"]) + epsilon
                    and float(dynamic["ttft_p95_s"])
                    > float(awq["ttft_p95_s"]) + epsilon
                ),
            }
        )
    return output


def catch_up_end(
    telemetry: list[dict[str, Any]], duration: float, resume_s: float
) -> float | None:
    contiguous_start: float | None = None
    for left, right, row in telemetry_segments(telemetry, duration):
        left = max(left, resume_s)
        if right <= left:
            continue
        if int(row["waiting_q_depth"]) == 0:
            if contiguous_start is None:
                contiguous_start = left
            if right - contiguous_start >= 3.0:
                return contiguous_start + 3.0
        else:
            contiguous_start = None
    return None


def transition_timelines(
    root: Path,
    plan: list[dict[str, Any]],
    workload_metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    timelines: list[dict[str, Any]] = []
    post_restore_requests: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    for plan_row in plan:
        if plan_row["condition"] != "closed_loop_dynamic":
            continue
        directory = run_dir(root, plan_row)
        metadata = read_json(directory / "metadata.json")
        requests = read_jsonl(directory / "requests.jsonl")
        telemetry = read_jsonl(directory / "telemetry.jsonl")
        batches = read_jsonl(directory / "batches.jsonl")
        transitions = read_jsonl(directory / "transitions.jsonl")
        start_ns = int(metadata["measurement_start_time_ns"])
        duration = float(metadata["measurement_duration_s"])
        segments = telemetry_segments(telemetry, duration)
        boundaries = boundaries_for(workload_metadata, str(plan_row["workload_class"]))
        directions = [event.get("direction") for event in transitions]
        expected = (
            []
            if plan_row["workload_class"] == "low_only"
            else [ENTRY, RESTORE]
            if plan_row["workload_class"] == "heldout_one_cycle"
            else [ENTRY, RESTORE, ENTRY, RESTORE]
        )
        state_rows.append(
            {
                "run_id": plan_row["run_id"],
                "workload_class": plan_row["workload_class"],
                "repeat": plan_row["repeat"],
                "transition_directions": "|".join(str(value) for value in directions),
                "expected_directions": "|".join(expected),
                "exact_expected_sequence": directions == expected,
                "final_controller_state": metadata.get("final_controller_state"),
                "final_runtime_precision": (
                    metadata.get("final_engine_snapshot") or {}
                ).get("precision_state"),
            }
        )
        for restore_index, event in enumerate(
            [event for event in transitions if event.get("direction") == RESTORE]
        ):
            trace = event.get("engine_trace") or {}
            lifecycle = trace.get("restore_lifecycle") or {}
            intent_s = ns_elapsed(
                event.get("release_intent_condition_first_true_ns"), start_ns
            )
            request_s = float(event["requested_elapsed_s"])
            pause_s = ns_elapsed(lifecycle.get("admission_pause_ns"), start_ns)
            drain_start_s = ns_elapsed(lifecycle.get("drain_start_ns"), start_ns)
            feasible_s = ns_elapsed(
                lifecycle.get("first_physical_shrink_legal_ns"), start_ns
            )
            drain_end_s = ns_elapsed(lifecycle.get("drain_end_ns"), start_ns)
            hot_start_s = ns_elapsed(event.get("model_transition_started_ns"), start_ns)
            hot_end_s = ns_elapsed(event.get("model_transition_ended_ns"), start_ns)
            capacity_published_s = ns_elapsed(
                lifecycle.get("capacity_published_ns"), start_ns
            )
            engine_receipt_s = ns_elapsed(
                lifecycle.get("engine_restore_request_received_ns"), start_ns
            )
            resume_s = ns_elapsed(lifecycle.get("admission_resume_ns"), start_ns)
            if None in (
                intent_s,
                engine_receipt_s,
                pause_s,
                drain_start_s,
                feasible_s,
                drain_end_s,
                hot_start_s,
                hot_end_s,
                capacity_published_s,
                resume_s,
            ):
                phase = phase_at(boundaries, request_s)
                timeline = {
                    "run_id": plan_row["run_id"],
                    "workload_class": plan_row["workload_class"],
                    "repeat": plan_row["repeat"],
                    "restore_index": restore_index,
                    "phase_index": phase["phase_index"],
                    "phase": phase["phase"],
                    "lifecycle_complete": False,
                }
                timelines.append(timeline)
                continue
            assert intent_s is not None and engine_receipt_s is not None
            assert pause_s is not None and drain_start_s is not None
            assert feasible_s is not None and drain_end_s is not None
            assert hot_start_s is not None and hot_end_s is not None
            assert capacity_published_s is not None and resume_s is not None
            phase = phase_at(boundaries, request_s)
            phase_last_arrival = float(phase["last_arrival_offset_s"])
            eligible_post = [
                row
                for row in requests
                if int(row.get("phase_index", -1)) == int(phase["phase_index"])
                and float(row["actual_arrival_offset_s"]) > resume_s
                and float(row["planned_arrival_offset_s"]) <= phase_last_arrival
            ]
            for request in eligible_post:
                post_restore_requests.append(
                    {
                        "run_id": plan_row["run_id"],
                        "repeat": plan_row["repeat"],
                        "restore_index": restore_index,
                        "phase_index": phase["phase_index"],
                        "benchmark_request_id": request["benchmark_request_id"],
                        "actual_arrival_offset_s": request["actual_arrival_offset_s"],
                        "prefill_precision_state": request["prefill_precision_state"],
                        "fp16_output_step_count": request["fp16_output_step_count"],
                        "awq_output_step_count": request["awq_output_step_count"],
                        "status": request["status"],
                        "engine_request_id_stable": request["engine_request_id_stable"],
                        "input_positions_contiguous": request["input_positions_contiguous"],
                        "executed_fp16": request["prefill_precision_state"] == "FP16"
                        and int(request["fp16_output_step_count"]) > 0,
                    }
                )
            pause_arrivals = [
                row
                for row in requests
                if pause_s <= float(row["actual_arrival_offset_s"]) <= resume_s
            ]
            drain_batches = [
                row
                for row in batches
                if drain_start_s
                <= (int(row["timestamp_ns"]) - start_ns) / 1e9
                <= feasible_s
                and int(row.get("num_decoding_sequences", 0)) > 0
            ]
            draining_request_ids = {
                request_id
                for row in drain_batches
                for request_id in row.get("benchmark_request_ids", [])
            }
            catch_end = catch_up_end(telemetry, duration, resume_s)
            next_entry = next(
                (
                    later
                    for later in transitions
                    if later.get("direction") == ENTRY
                    and float(later["requested_elapsed_s"]) > request_s
                ),
                None,
            )
            observation = event.get("release_intent_observation") or {}
            context_pause = lifecycle.get("context_at_admission_pause") or {}
            context_safe = lifecycle.get("context_at_first_physical_shrink_legal") or {}
            timeline = {
                "run_id": plan_row["run_id"],
                "workload_class": plan_row["workload_class"],
                "repeat": plan_row["repeat"],
                "restore_index": restore_index,
                "phase_index": phase["phase_index"],
                "phase": phase["phase"],
                "phase_regime": phase["regime"],
                "phase_start_s": phase["start_offset_s"],
                "phase_last_arrival_s": phase_last_arrival,
                "lifecycle_complete": True,
                "release_intent_first_true_s": intent_s,
                "restore_request_s": request_s,
                "engine_restore_request_received_s": engine_receipt_s,
                "admission_pause_s": pause_s,
                "allocated_blocks_at_release_intent": observation.get(
                    "physical_used_kv_blocks"
                ),
                "scheduler_used_blocks_at_release_intent": observation.get(
                    "scheduler_used_kv_blocks"
                ),
                "blocks_above_fp16_base_at_release_intent": observation.get(
                    "blocks_above_fp16_base"
                ),
                "allocated_blocks_at_admission_pause": context_pause.get(
                    "allocated_gpu_kv_blocks"
                ),
                "drain_required": lifecycle.get("drain_required"),
                "drain_start_s": drain_start_s,
                "first_physical_shrink_legal_s": feasible_s,
                "allocated_blocks_at_first_legal": context_safe.get(
                    "allocated_gpu_kv_blocks"
                ),
                "drain_end_s": drain_end_s,
                "drain_duration_s": feasible_s - drain_start_s,
                "hot_restore_start_s": hot_start_s,
                "hot_restore_end_s": hot_end_s,
                "hot_restore_duration_s": hot_end_s - hot_start_s,
                "capacity_published_s": capacity_published_s,
                "admission_resume_s": resume_s,
                "pause_to_resume_s": resume_s - pause_s,
                "arrivals_during_admission_pause": len(pause_arrivals),
                "decode_forwards_during_drain": len(drain_batches),
                "requests_continuing_decode_during_drain": len(draining_request_ids),
                "queue_area_during_drain_request_s": integrate(
                    segments,
                    lambda row: float(row["waiting_q_depth"]),
                    start=drain_start_s,
                    end=feasible_s,
                ),
                "queue_area_during_pause_request_s": integrate(
                    segments,
                    lambda row: float(row["waiting_q_depth"]),
                    start=pause_s,
                    end=resume_s,
                ),
                "queue_peak_during_pause": max(
                    (
                        int(row["waiting_q_depth"])
                        for row in telemetry
                        if pause_s <= float(row["elapsed_s"]) <= resume_s
                    ),
                    default=0,
                ),
                "catch_up_end_s": catch_end,
                "catch_up_duration_s": (
                    None if catch_end is None else catch_end - resume_s
                ),
                "queue_area_until_catch_up_request_s": (
                    None
                    if catch_end is None
                    else integrate(
                        segments,
                        lambda row: float(row["waiting_q_depth"]),
                        start=resume_s,
                        end=catch_end,
                    )
                ),
                "post_restore_arrivals_in_low_phase": len(eligible_post),
                "post_restore_requests_executing_fp16": sum(
                    row["prefill_precision_state"] == "FP16"
                    and int(row["fp16_output_step_count"]) > 0
                    for row in eligible_post
                ),
                "seconds_until_phase_last_arrival": phase_last_arrival - resume_s,
                "next_entry_request_s": (
                    None if next_entry is None else next_entry["requested_elapsed_s"]
                ),
                "next_entry_delay_after_resume_s": (
                    None
                    if next_entry is None
                    else float(next_entry["requested_elapsed_s"]) - resume_s
                ),
                "model_memory_before_allocated_bytes": (
                    trace.get("memory_before") or {}
                ).get("allocated_bytes"),
                "model_memory_after_allocated_bytes": (
                    trace.get("memory_after") or {}
                ).get("allocated_bytes"),
                "transition_peak_allocated_bytes": trace.get(
                    "transition_peak_allocated_bytes"
                ),
                "restored_driver_hbm_used_bytes": next(
                    (
                        row.get("gpu_memory_used_bytes")
                        for row in telemetry
                        if float(row["elapsed_s"]) >= resume_s
                    ),
                    None,
                ),
                "kv_integrity_preserved": (
                    ((trace.get("kv_resize") or {}).get("integrity_before") or {}).get(
                        "logical_digest"
                    )
                    == ((trace.get("kv_resize") or {}).get("integrity_after") or {}).get(
                        "logical_digest"
                    )
                ),
            }
            timeline["timeline_ordered"] = all(
                left <= right
                for left, right in zip(
                    (
                        intent_s,
                        request_s,
                        engine_receipt_s,
                        pause_s,
                        drain_start_s,
                        feasible_s,
                        drain_end_s,
                        hot_start_s,
                        hot_end_s,
                        capacity_published_s,
                        resume_s,
                    ),
                    (
                        request_s,
                        engine_receipt_s,
                        pause_s,
                        drain_start_s,
                        feasible_s,
                        drain_end_s,
                        hot_start_s,
                        hot_end_s,
                        capacity_published_s,
                        resume_s,
                        math.inf,
                    ),
                )
            )
            timeline["useful_service_gate"] = bool(
                phase["regime"] == "low"
                and request_s >= float(phase["start_offset_s"])
                and len(eligible_post) >= 8
                and timeline["post_restore_requests_executing_fp16"] >= 8
                and phase_last_arrival - resume_s >= 20.0
            )
            timeline["drain_cost_gate"] = bool(
                resume_s - pause_s <= 5.0
                and catch_end is not None
                and catch_end - resume_s <= 10.0
            )
            next_phase = next(
                (
                    boundary
                    for boundary in boundaries
                    if float(boundary["start_offset_s"])
                    > float(phase["start_offset_s"])
                ),
                None,
            )
            timeline["no_restore_induced_reentry"] = bool(
                next_entry is None
                or (
                    float(next_entry["requested_elapsed_s"]) - resume_s >= 15.0
                    and next_phase is not None
                    and next_phase["regime"] == "high"
                    and float(next_entry["requested_elapsed_s"])
                    >= float(next_phase["start_offset_s"])
                )
            )
            timelines.append(timeline)
    return timelines, post_restore_requests, state_rows


def environment_rows(root: Path, plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for plan_row in plan:
        rows = read_jsonl(run_dir(root, plan_row) / "gpu_environment.jsonl")
        result: dict[str, Any] = {
            "run_id": plan_row["run_id"],
            "workload_class": plan_row["workload_class"],
            "condition": plan_row["condition"],
            "repeat": plan_row["repeat"],
            "sample_count": len(rows),
            "available_sample_count": sum(
                row.get("gpu_environment_source") == "pynvml" for row in rows
            ),
            "error_sample_count": sum(bool(row.get("gpu_environment_error")) for row in rows),
            "throttle_reason_masks": "|".join(
                str(value)
                for value in sorted(
                    {
                        int(row["gpu_clock_throttle_reasons_mask"])
                        for row in rows
                        if row.get("gpu_clock_throttle_reasons_mask") is not None
                    }
                )
            ),
        }
        for prefix, field in (
            ("sm_clock_mhz", "gpu_sm_clock_mhz"),
            ("memory_clock_mhz", "gpu_memory_clock_mhz"),
            ("temperature_c", "gpu_temperature_c"),
            ("power_w", "gpu_power_w"),
            ("utilization_percent", "gpu_utilization_percent"),
        ):
            result.update(numeric_summary(prefix, rows, field))
        output.append(result)
    return output


def quality_rows(root: Path, plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for plan_row in plan:
        requests = read_jsonl(run_dir(root, plan_row) / "requests.jsonl")
        scores = [
            best_reference_f1(
                str(row.get("generated_answer") or ""),
                [str(answer) for answer in row.get("dataset_reference_answers", [])],
            )
            for row in requests
        ]
        output.append(
            {
                "run_id": plan_row["run_id"],
                "workload_class": plan_row["workload_class"],
                "condition": plan_row["condition"],
                "repeat": plan_row["repeat"],
                "request_count": len(scores),
                "descriptive_dureader_f1_percent": 100 * sum(scores) / len(scores),
                "decision_role": "descriptive_only_not_a_release_side_gate",
            }
        )
    return output


def decision_artifact(
    serving: list[dict[str, Any]],
    validations: list[dict[str, Any]],
    matched: list[dict[str, Any]],
    timelines: list[dict[str, Any]],
    states: list[dict[str, Any]],
    environment: list[dict[str, Any]],
) -> dict[str, Any]:
    serving_lookup = {
        (row["workload_class"], row["condition"], int(row["repeat"])): row
        for row in serving
    }
    throughput: list[dict[str, Any]] = []
    for workload in ("low_only", *ALTERNATING):
        fp16 = [
            serving_lookup[(workload, "runtime_static_fp16", repeat)]
            for repeat in (0, 1)
        ]
        dynamic = [
            serving_lookup[(workload, "closed_loop_dynamic", repeat)]
            for repeat in (0, 1)
        ]
        repeat_ratios = [
            float(dynamic[index]["completed_request_throughput_rps"])
            / float(fp16[index]["completed_request_throughput_rps"])
            for index in range(2)
        ]
        mean_ratio = (
            sum(float(row["completed_request_throughput_rps"]) for row in dynamic)
            / sum(float(row["completed_request_throughput_rps"]) for row in fp16)
        )
        throughput.append(
            {
                "workload_class": workload,
                "repeat_ratios": repeat_ratios,
                "two_repeat_mean_ratio": mean_ratio,
                "passes_0p98_noninferiority": min(repeat_ratios) >= 0.98
                and mean_ratio >= 0.98,
            }
        )
    low = [
        row
        for row in matched
        if row["workload_class"] == "low_only" and int(row["phase_index"]) == 0
    ]
    high = [row for row in matched if row["regime"] == "high"]
    recovery_comparison = {
        (row["workload_class"], int(row["repeat"]), int(row["phase_index"])): row
        for row in matched
        if row["regime"] == "low" and int(row["phase_index"]) > 0
    }
    for timeline in timelines:
        comparison = recovery_comparison.get(
            (
                str(timeline["workload_class"]),
                int(timeline["repeat"]),
                int(timeline["phase_index"]),
            )
        )
        timeline["recovery_not_obviously_worse_than_both_controls"] = bool(
            comparison is not None and not comparison["obviously_worse_than_both_controls"]
        )
        timeline["transition_pass"] = bool(
            timeline.get("lifecycle_complete")
            and timeline.get("timeline_ordered")
            and timeline.get("kv_integrity_preserved")
            and timeline.get("useful_service_gate")
            and timeline.get("drain_cost_gate")
            and timeline.get("no_restore_induced_reentry")
            and timeline["recovery_not_obviously_worse_than_both_controls"]
        )
    checks = {
        "all_16_runs_raw_valid": len(validations) == 16
        and all(row["checks"]["all_pass"] for row in validations),
        "low_only_zero_false_entries_both": len(low) == 2
        and all(int(row["dynamic_entry_requests"]) == 0 for row in low),
        "low_only_fp16_like_within_noise_both": len(low) == 2
        and all(row["dynamic_within_noise_of_fp16"] for row in low),
        "every_high_phase_preserves_entry_benefit": len(high) == 6
        and all(row["high_mechanism_gate"] for row in high),
        "throughput_noninferior_with_0p98_margin": all(
            row["passes_0p98_noninferiority"] for row in throughput
        ),
        "every_release_has_complete_useful_bounded_recovery": len(timelines) == 6
        and all(row.get("transition_pass") for row in timelines),
        "ordered_one_and_two_cycle_sequences_both_repeats": len(states) == 6
        and all(row["exact_expected_sequence"] for row in states)
        and all(
            row["final_runtime_precision"] == "FP16"
            for row in states
            if row["workload_class"] in ALTERNATING
        ),
        "no_chatter_or_release_during_high": len(high) == 6
        and all(int(row["dynamic_release_requests"]) == 0 for row in high)
        and all(row.get("no_restore_induced_reentry") for row in timelines),
        "gpu_clock_temperature_power_available": len(environment) == 16
        and all(int(row["available_sample_count"]) > 0 for row in environment),
    }
    restored_hbm: list[dict[str, Any]] = []
    for workload in ALTERNATING:
        restore_count = 1 if workload == "heldout_one_cycle" else 2
        for restore_index in range(restore_count):
            values = [
                int(row["restored_driver_hbm_used_bytes"])
                for row in timelines
                if row["workload_class"] == workload
                and int(row["restore_index"]) == restore_index
                and row.get("restored_driver_hbm_used_bytes") is not None
            ]
            restored_hbm.append(
                {
                    "workload_class": workload,
                    "restore_index": restore_index,
                    "repeat_values_bytes": values,
                    "repeat_spread_bytes": max(values) - min(values)
                    if len(values) == 2
                    else None,
                }
            )
    scientific_checks = {
        key: value
        for key, value in checks.items()
        if key != "gpu_clock_temperature_power_available"
    }
    return {
        "schema_version": 1,
        "throughput_noninferiority": throughput,
        "restored_driver_hbm": restored_hbm,
        "checks": checks,
        "release_side_systems_decision": (
            "GO" if all(scientific_checks.values()) else "NO-GO"
        ),
        "decision_scope": (
            "Release-side systems only. Descriptive DuReader F1 is not a completion gate."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    plan = selected_plan(root)
    workload_metadata = read_json(root / "input" / "workload_metadata.json")

    serving: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    for plan_row in plan:
        row, validation = analyze_run(run_dir(root, plan_row), plan_row)
        row["repeat"] = int(plan_row["repeat"])
        serving.append(row)
        validations.append(validation)
    phases = phase_metrics(root, plan, workload_metadata)
    matched = matched_phase_rows(phases)
    timelines, post_restore, states = transition_timelines(
        root, plan, workload_metadata
    )
    environment = environment_rows(root, plan)
    quality = quality_rows(root, plan)
    decision = decision_artifact(
        serving, validations, matched, timelines, states, environment
    )

    write_csv(output / "serving_runs.csv", serving)
    write_csv(output / "phase_metrics.csv", phases)
    write_csv(output / "matched_phase_comparison.csv", matched)
    write_csv(
        output / "transition_timeline.csv",
        timelines,
        ("run_id", "workload_class", "repeat", "restore_index", "lifecycle_complete"),
    )
    write_csv(
        output / "post_restore_requests.csv",
        post_restore,
        ("run_id", "repeat", "restore_index", "benchmark_request_id", "executed_fp16"),
    )
    write_csv(output / "state_timeline.csv", states)
    write_csv(output / "gpu_environment_summary.csv", environment)
    write_csv(output / "quality_descriptive.csv", quality)
    (output / "run_validation.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": (
                    "PASS"
                    if all(row["checks"]["all_pass"] for row in validations)
                    else "FAIL"
                ),
                "runs": validations,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "multi_cycle_state_timeline.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "expected": "FP16 -> AWQ -> FP16 -> AWQ -> FP16",
                "runs": [
                    row
                    for row in states
                    if row["workload_class"] == "heldout_two_cycle"
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
