"""Analyze preregistered paired v11 throughput-confirmation runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

from scipy.stats import t as student_t

from .metrics import percentile
from .run_throughput_confirmation_plan import valid_completed_run


MARGIN = 0.98
ONE_SIDED_ALPHA = 0.025
CONDITIONS = ("runtime_static_fp16", "closed_loop_dynamic")
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def distribution(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": mean(values) if values else None,
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values) if values else None,
    }


def exact_expected_directions(workload: str) -> list[str]:
    if workload == "low_only":
        return []
    if workload == "heldout_one_cycle":
        return [ENTRY, RESTORE]
    if workload == "heldout_two_cycle":
        return [ENTRY, RESTORE, ENTRY, RESTORE]
    raise ValueError(f"unknown workload: {workload}")


def request_integrity(requests: list[dict[str, Any]], expected: int) -> bool:
    return bool(
        len(requests) == expected
        and len({row.get("benchmark_request_id") for row in requests}) == expected
        and all(
            row.get("status") == "completed"
            and int(row.get("output_token_count", -1)) == 512
            and len(row.get("output_token_ids", [])) == 512
            and len(row.get("step_precision_states", [])) == 512
            and row.get("engine_request_id_stable") is True
            and row.get("input_positions_contiguous") is True
            and row.get("error") is None
            for row in requests
        )
    )


def run_record(run_dir: Path, plan_row: dict[str, Any]) -> dict[str, Any]:
    metadata = read_json(run_dir / "metadata.json")
    summary = read_json(run_dir / "summary.json")
    requests = read_jsonl(run_dir / "requests.jsonl")
    telemetry = read_jsonl(run_dir / "telemetry.jsonl")
    controller = read_jsonl(run_dir / "controller.jsonl")
    gpu = read_jsonl(run_dir / "gpu_environment.jsonl")
    loop_lag = read_jsonl(run_dir / "event_loop_lag.jsonl")
    transitions = read_jsonl(run_dir / "transitions.jsonl")
    initialization = read_jsonl(run_dir / "initialization_transitions.jsonl")

    start_ns = int(metadata["measurement_start_time_ns"])
    duration = float(metadata["measurement_duration_s"])
    final_planned = max(requests, key=lambda row: int(row["sequence"]))
    final_actual = max(requests, key=lambda row: int(row["arrival_time_ns"]))
    last_completion_ns = max(int(row["stream_completion_time_ns"]) for row in requests)
    final_planned_ns = int(final_planned["planned_arrival_time_ns"])
    final_actual_ns = int(final_actual["arrival_time_ns"])
    last_completion_elapsed = (last_completion_ns - start_ns) / 1e9
    final_actual_elapsed = (final_actual_ns - start_ns) / 1e9
    planned_horizon = float(metadata["planned_schedule_last_offset_s"])
    arrival_component = final_actual_elapsed - planned_horizon
    completion_tail = (last_completion_ns - final_actual_ns) / 1e9
    planned_completion_tail = (last_completion_ns - final_planned_ns) / 1e9
    post_completion_runner = duration - last_completion_elapsed

    evaluate_wall = sum(int(row.get("controller_evaluate_wall_ns", 0)) for row in controller)
    evaluate_cpu = sum(int(row.get("controller_evaluate_cpu_ns", 0)) for row in controller)
    window_wall = sum(int(row.get("controller_window_wall_ns", 0)) for row in controller)
    window_cpu = sum(int(row.get("controller_window_cpu_ns", 0)) for row in controller)
    bookkeeping_wall = sum(
        int(row.get("controller_decision_bookkeeping_wall_ns", 0))
        + int(row.get("controller_trace_fields_wall_ns", 0))
        for row in controller
    )
    bookkeeping_cpu = sum(
        int(row.get("controller_decision_bookkeeping_cpu_ns", 0))
        + int(row.get("controller_trace_fields_cpu_ns", 0))
        for row in controller
    )

    ttft = [float(row["ttft_s"]) for row in requests]
    tpot = [float(row["tpot_s"]) for row in requests]
    output_first_lag = [
        (int(row["first_stream_token_received_time_ns"]) - int(row["first_output_token_time_ns"]))
        / 1e9
        for row in requests
    ]
    output_final_lag = [
        (int(row["stream_completion_time_ns"]) - int(row["completion_time_ns"])) / 1e9
        for row in requests
    ]
    environment = [row for row in gpu if row.get("gpu_environment_source") == "pynvml"]
    hbm = [int(row["gpu_memory_used_bytes"]) for row in telemetry]
    directions = [row.get("direction") for row in transitions]
    expected_directions = exact_expected_directions(str(plan_row["workload_class"]))
    dynamic = plan_row["condition"] == "closed_loop_dynamic"
    final_snapshot = metadata.get("final_engine_snapshot") or {}

    return {
        "run_id": plan_row["run_id"],
        "actual_run_id": run_dir.name,
        "pair_id": plan_row["pair_id"],
        "ordinal": plan_row["ordinal"],
        "workload": plan_row["workload_class"],
        "condition": plan_row["condition"],
        "request_count": len(requests),
        "completed_request_count": metadata["completed_request_count"],
        "generated_output_token_count": metadata["generated_output_token_count"],
        "measurement_duration_s": duration,
        "completed_request_throughput_rps": float(summary["completed_request_throughput_rps"]),
        "generated_token_throughput_tps": float(summary["generated_token_throughput_tps"]),
        "ttft_p95_s": percentile(ttft, 95),
        "ttft_p99_s": percentile(ttft, 99),
        "tpot_mean_s": mean(tpot),
        "tpot_p95_s": percentile(tpot, 95),
        "planned_last_arrival_offset_s": planned_horizon,
        "planned_last_arrival_timestamp_ns": final_planned_ns,
        "actual_last_arrival_timestamp_ns": final_actual_ns,
        "last_request_completion_timestamp_ns": last_completion_ns,
        "arrival_timing_component_s": arrival_component,
        "post_last_actual_arrival_completion_tail_s": completion_tail,
        "post_last_planned_arrival_drain_s": planned_completion_tail,
        "post_completion_runner_s": post_completion_runner,
        "arrival_jitter_p95_s": percentile([abs(float(row["arrival_jitter_s"])) for row in requests], 95),
        "arrival_jitter_p99_s": percentile([abs(float(row["arrival_jitter_s"])) for row in requests], 99),
        "telemetry_lateness_p95_s": percentile([float(row["sampling_jitter_s"]) for row in telemetry], 95),
        "telemetry_lateness_p99_s": percentile([float(row["sampling_jitter_s"]) for row in telemetry], 99),
        "event_loop_lag_p95_s": percentile([float(row["event_loop_lag_s"]) for row in loop_lag], 95),
        "event_loop_lag_p99_s": percentile([float(row["event_loop_lag_s"]) for row in loop_lag], 99),
        "output_first_consumer_lag_p95_s": percentile(output_first_lag, 95),
        "output_first_consumer_lag_p99_s": percentile(output_first_lag, 99),
        "output_final_consumer_lag_p95_s": percentile(output_final_lag, 95),
        "controller_evaluate_wall_s": evaluate_wall / 1e9,
        "controller_evaluate_cpu_s": evaluate_cpu / 1e9,
        "controller_window_wall_s": window_wall / 1e9,
        "controller_window_cpu_s": window_cpu / 1e9,
        "controller_trace_bookkeeping_wall_s": bookkeeping_wall / 1e9,
        "controller_trace_bookkeeping_cpu_s": bookkeeping_cpu / 1e9,
        "controller_blocking_wall_fraction": (evaluate_wall + bookkeeping_wall) / 1e9 / duration,
        "gpu_environment_sample_count": len(environment),
        "gpu_sm_clock_mean_mhz": mean([float(row["gpu_sm_clock_mhz"]) for row in environment]) if environment else None,
        "gpu_temperature_mean_c": mean([float(row["gpu_temperature_c"]) for row in environment]) if environment else None,
        "gpu_temperature_max_c": max([float(row["gpu_temperature_c"]) for row in environment]) if environment else None,
        "gpu_power_mean_w": mean([float(row["gpu_power_w"]) for row in environment]) if environment else None,
        "gpu_power_max_w": max([float(row["gpu_power_w"]) for row in environment]) if environment else None,
        "hbm_used_start_bytes": hbm[0],
        "hbm_used_end_bytes": hbm[-1],
        "hbm_used_min_bytes": min(hbm),
        "hbm_used_max_bytes": max(hbm),
        "transition_directions": directions,
        "transition_count": len(transitions),
        "initialization_transition_count": len(initialization),
        "request_integrity": request_integrity(requests, int(plan_row["request_count"])),
        "transition_sequence_valid": (directions == expected_directions if dynamic else directions == []),
        "final_state_valid": (
            final_snapshot.get("precision_state") == "FP16"
            and int(final_snapshot.get("num_gpu_blocks", -1)) == 1759
        ),
        "all_transitions_success": all(row.get("result") == "success" for row in transitions),
        "nvml_available": bool(environment),
        "raw_run_valid": valid_completed_run(run_dir, plan_row),
    }


def paired_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for row in records:
        by_pair.setdefault(str(row["pair_id"]), {})[str(row["condition"])] = row
    output: list[dict[str, Any]] = []
    for pair_id, pair in sorted(by_pair.items()):
        fp16 = pair["runtime_static_fp16"]
        dynamic = pair["closed_loop_dynamic"]
        throughput_ratio = dynamic["completed_request_throughput_rps"] / fp16["completed_request_throughput_rps"]
        duration_ratio = dynamic["measurement_duration_s"] / fp16["measurement_duration_s"]
        generated_ratio = dynamic["generated_token_throughput_tps"] / fp16["generated_token_throughput_tps"]
        total_log_ratio = math.log(throughput_ratio)
        h = float(fp16["planned_last_arrival_offset_s"])
        jf = float(fp16["arrival_timing_component_s"])
        jd = float(dynamic["arrival_timing_component_s"])
        qf = float(fp16["post_last_actual_arrival_completion_tail_s"])
        qd = float(dynamic["post_last_actual_arrival_completion_tail_s"])
        rf = float(fp16["post_completion_runner_s"])
        rd = float(dynamic["post_completion_runner_s"])
        arrival_term = math.log((h + jf) / (h + jd))
        through_completion = math.log((h + jf + qf) / (h + jd + qd))
        completion_tail_term = through_completion - arrival_term
        runner_term = total_log_ratio - through_completion
        positive_duration_excess = max(
            0.0, float(dynamic["measurement_duration_s"]) - float(fp16["measurement_duration_s"])
        )
        controller_wall = float(dynamic["controller_evaluate_wall_s"]) + float(
            dynamic["controller_trace_bookkeeping_wall_s"]
        )
        output.append(
            {
                "pair_id": pair_id,
                "workload": fp16["workload"],
                "condition_order": (
                    "FP16_DYNAMIC" if int(fp16["ordinal"]) < int(dynamic["ordinal"]) else "DYNAMIC_FP16"
                ),
                "fp16_run_id": fp16["run_id"],
                "dynamic_run_id": dynamic["run_id"],
                "fp16_throughput_rps": fp16["completed_request_throughput_rps"],
                "dynamic_throughput_rps": dynamic["completed_request_throughput_rps"],
                "throughput_ratio_dynamic_over_fp16": throughput_ratio,
                "log_throughput_ratio": total_log_ratio,
                "measurement_duration_ratio_dynamic_over_fp16": duration_ratio,
                "generated_token_throughput_ratio_dynamic_over_fp16": generated_ratio,
                "fp16_ttft_p95_s": fp16["ttft_p95_s"],
                "dynamic_ttft_p95_s": dynamic["ttft_p95_s"],
                "fp16_ttft_p99_s": fp16["ttft_p99_s"],
                "dynamic_ttft_p99_s": dynamic["ttft_p99_s"],
                "fp16_tpot_mean_s": fp16["tpot_mean_s"],
                "dynamic_tpot_mean_s": dynamic["tpot_mean_s"],
                "final_planned_arrival_offset_s": h,
                "fp16_last_completion_timestamp_ns": fp16["last_request_completion_timestamp_ns"],
                "dynamic_last_completion_timestamp_ns": dynamic["last_request_completion_timestamp_ns"],
                "fp16_post_last_planned_arrival_drain_s": fp16["post_last_planned_arrival_drain_s"],
                "dynamic_post_last_planned_arrival_drain_s": dynamic["post_last_planned_arrival_drain_s"],
                "tail_arrival_timing_log_term": arrival_term,
                "tail_completion_log_term": completion_tail_term,
                "tail_post_completion_runner_log_term": runner_term,
                "tail_terms_sum_exact": arrival_term + completion_tail_term + runner_term,
                "completion_tail_fraction_of_log_throughput_difference": (
                    completion_tail_term / total_log_ratio if total_log_ratio != 0 else None
                ),
                "dynamic_controller_blocking_wall_s": controller_wall,
                "positive_duration_excess_s": positive_duration_excess,
                "controller_fraction_of_positive_duration_excess": (
                    controller_wall / positive_duration_excess if positive_duration_excess > 0 else None
                ),
            }
        )
    return output


def inference_for(workload: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row["log_throughput_ratio"]) for row in rows]
    n = len(values)
    if n < 2:
        raise ValueError(f"{workload} needs at least two complete pairs")
    avg = mean(values)
    sd = stdev(values)
    se = sd / math.sqrt(n)
    df = n - 1
    critical = float(student_t.ppf(1.0 - ONE_SIDED_ALPHA, df))
    lower_log = avg - critical * se
    test_statistic = (avg - math.log(MARGIN)) / se if se else math.inf
    p_value = float(student_t.sf(test_statistic, df)) if se else (0.0 if avg > math.log(MARGIN) else 1.0)
    two_sided_95_upper = avg + float(student_t.ppf(0.975, df)) * se
    return {
        "workload": workload,
        "pair_count": n,
        "estimand": "geometric mean Dynamic/FP16 completed-request throughput ratio",
        "margin": MARGIN,
        "null": "mean log ratio <= log(0.98)",
        "method": "paired one-sample Student t on run-level log ratios",
        "one_sided_alpha": ONE_SIDED_ALPHA,
        "one_sided_confidence_level": 1.0 - ONE_SIDED_ALPHA,
        "mean_log_ratio": avg,
        "log_ratio_sd": sd,
        "standard_error": se,
        "degrees_of_freedom": df,
        "t_statistic_vs_margin": test_statistic,
        "one_sided_p_value": p_value,
        "geometric_mean_ratio": math.exp(avg),
        "one_sided_lower_confidence_bound": math.exp(lower_log),
        "noninferior": math.exp(lower_log) > MARGIN,
        "two_sided_95_upper_ratio": math.exp(two_sided_95_upper),
        "repeatable_nonzero_deficit": math.exp(two_sided_95_upper) < 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    plan = read_json(args.plan)["runs"]
    selected = read_json(args.status)["selected_runs"]
    args.output.mkdir(parents=True, exist_ok=True)

    records = [
        run_record(root / "runs" / selected[row["run_id"]], row)
        for row in plan
    ]
    pairs = paired_rows(records)
    workloads = sorted({str(row["workload"]) for row in pairs})
    inference = [
        inference_for(workload, [row for row in pairs if row["workload"] == workload])
        for workload in workloads
    ]
    guardrails = {
        "all_raw_runs_valid": all(row["raw_run_valid"] for row in records),
        "all_requests_integral": all(row["request_integrity"] for row in records),
        "all_transition_sequences_valid": all(row["transition_sequence_valid"] for row in records),
        "all_final_states_fp16_1759": all(row["final_state_valid"] for row in records),
        "all_transitions_success": all(row["all_transitions_success"] for row in records),
        "nvml_available_all_runs": all(row["nvml_available"] for row in records),
        "low_only_dynamic_zero_transitions": all(
            row["transition_count"] == 0
            for row in records
            if row["workload"] == "low_only" and row["condition"] == "closed_loop_dynamic"
        ),
    }
    accounted = [
        float(row["controller_fraction_of_positive_duration_excess"])
        for row in pairs
        if row["controller_fraction_of_positive_duration_excess"] is not None
    ]
    controller_overhead = {
        "measurement_boundary": (
            "evaluate includes all frozen v10 window construction/integration and decision work; "
            "trace bookkeeping includes decision construction and trace_fields only"
        ),
        "instrumentation_effect": "timings are output-only and exact policy parity is a prerequisite",
        "dynamic_runs": [
            {
                key: row[key]
                for key in (
                    "run_id",
                    "workload",
                    "controller_evaluate_wall_s",
                    "controller_evaluate_cpu_s",
                    "controller_window_wall_s",
                    "controller_window_cpu_s",
                    "controller_trace_bookkeeping_wall_s",
                    "controller_trace_bookkeeping_cpu_s",
                    "controller_blocking_wall_fraction",
                )
            }
            for row in records
            if row["condition"] == "closed_loop_dynamic"
        ],
        "median_fraction_of_positive_matched_duration_excess": median(accounted) if accounted else None,
        "materiality_rule": (
            "eligible for one optimization only if the two-sided 95% upper ratio is below 1.0 "
            "and median measured blocking wall time is >=25% of positive matched duration excess"
        ),
    }
    phase_a_optimization_eligible = bool(
        len(inference) == 1
        and inference[0]["repeatable_nonzero_deficit"]
        and accounted
        and median(accounted) >= 0.25
    )
    decision = {
        "scope": "phase_a" if workloads == ["low_only"] else "phase_c",
        "workload_noninferiority": {
            row["workload"]: row["noninferior"] for row in inference
        },
        "all_workloads_noninferior": all(row["noninferior"] for row in inference),
        "all_guardrails_pass": all(guardrails.values()),
        "phase_a_control_plane_optimization_eligible": phase_a_optimization_eligible,
        "phase_a_implementation_decision": (
            "ONE_SEMANTICS_PRESERVING_OPTIMIZATION_ALLOWED"
            if phase_a_optimization_eligible
            else "KEEP_CONTROLLER_UNCHANGED"
        ),
    }

    write_csv(args.output / "serving_runs.csv", records)
    write_csv(args.output / "paired_throughput.csv", pairs)
    write_csv(args.output / "tail_decomposition.csv", pairs)
    write_csv(
        args.output / "gpu_environment_summary.csv",
        [
            {key: value for key, value in row.items() if key.startswith("gpu_") or key.startswith("hbm_") or key in {"run_id", "pair_id", "workload", "condition", "ordinal"}}
            for row in records
        ],
    )
    (args.output / "inference.json").write_text(json.dumps(inference, indent=2, sort_keys=True) + "\n")
    (args.output / "controller_overhead.json").write_text(json.dumps(controller_overhead, indent=2, sort_keys=True) + "\n")
    (args.output / "guardrails.json").write_text(json.dumps(guardrails, indent=2, sort_keys=True) + "\n")
    (args.output / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"inference": inference, "decision": decision}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
