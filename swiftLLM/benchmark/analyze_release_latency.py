"""Replay the frozen v9 traces to diagnose release latency and freeze v10 candidates.

This is development evidence only: an earlier release would change the later
trajectory, so none of the replayed candidate outcomes is a v10 serving result.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable

from .analyze_crossover import counter_value_at, integrate, telemetry_segments


FP16_POLICY_REFERENCE_BLOCKS = 1768
RUNTIME_FP16_BLOCKS = 1759
DYNAMIC_RUN_IDS = (
    "v9-high-closed_loop_dynamic-rep0",
    "v9-high-closed_loop_dynamic-rep1",
    "v9-low_high_low-closed_loop_dynamic-rep0",
    "v9-low_high_low-closed_loop_dynamic-rep1",
)
CANDIDATES = {
    "primary_quiet_pressure_drop": {
        "history_window_s": 15.0,
        "recent_window_s": 5.0,
        "prior_window_s": 10.0,
        "waiting_mean_lte": 0.5,
        "current_waiting_eq": 0,
        "new_preemptions_eq": 0,
        "current_fp16_reference_utilization_lte": 1.25,
        "recent_to_prior_utilization_ratio_lte": 0.80,
    }
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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


def first_row(
    rows: list[dict[str, Any]],
    not_before: float,
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in rows
            if float(row["elapsed_s"]) >= not_before and predicate(row)
        ),
        None,
    )


def candidate_stats(
    rows: list[dict[str, Any]],
    segments: list[tuple[float, float, dict[str, Any]]],
    row: dict[str, Any],
    spec: dict[str, float | int],
) -> dict[str, Any]:
    now = float(row["elapsed_s"])
    recent_window = float(spec["recent_window_s"])
    prior_window = float(spec["prior_window_s"])
    recent_start = now - recent_window
    prior_start = recent_start - prior_window
    waiting_mean = integrate(
        segments,
        lambda value: float(value["waiting_q_depth"]),
        start=recent_start,
        end=now,
    ) / recent_window
    recent_utilization_mean = integrate(
        segments,
        lambda value: float(value["scheduler_used_kv_blocks"])
        / FP16_POLICY_REFERENCE_BLOCKS,
        start=recent_start,
        end=now,
    ) / recent_window
    prior_utilization_mean = integrate(
        segments,
        lambda value: float(value["scheduler_used_kv_blocks"])
        / FP16_POLICY_REFERENCE_BLOCKS,
        start=prior_start,
        end=recent_start,
    ) / prior_window
    utilization_ratio = (
        recent_utilization_mean / prior_utilization_mean
        if prior_utilization_mean > 0
        else None
    )
    recent_preemptions = int(row["preemption_count"]) - counter_value_at(
        rows, "preemption_count", recent_start
    )
    current_utilization = (
        float(row["scheduler_used_kv_blocks"]) / FP16_POLICY_REFERENCE_BLOCKS
    )
    satisfied = bool(
        int(row["waiting_q_depth"]) == int(spec["current_waiting_eq"])
        and waiting_mean <= float(spec["waiting_mean_lte"])
        and recent_preemptions == int(spec["new_preemptions_eq"])
        and current_utilization
        <= float(spec["current_fp16_reference_utilization_lte"])
        and utilization_ratio is not None
        and utilization_ratio
        <= float(spec["recent_to_prior_utilization_ratio_lte"])
    )
    return {
        "waiting_mean": waiting_mean,
        "recent_utilization_mean": recent_utilization_mean,
        "prior_utilization_mean": prior_utilization_mean,
        "utilization_ratio": utilization_ratio,
        "recent_preemptions": recent_preemptions,
        "current_utilization": current_utilization,
        "satisfied": satisfied,
    }


def old_stats(
    rows: list[dict[str, Any]],
    segments: list[tuple[float, float, dict[str, Any]]],
    row: dict[str, Any],
    entry_request_s: float,
) -> dict[str, Any]:
    now = float(row["elapsed_s"])
    start = now - 10.0
    waiting_mean = integrate(
        segments,
        lambda value: float(value["waiting_q_depth"]),
        start=start,
        end=now,
    ) / 10.0
    le070_fraction = integrate(
        segments,
        lambda value: float(
            float(value["scheduler_used_kv_blocks"])
            / FP16_POLICY_REFERENCE_BLOCKS
            <= 0.70
        ),
        start=start,
        end=now,
    ) / 10.0
    recent_preemptions = int(row["preemption_count"]) - counter_value_at(
        rows, "preemption_count", start
    )
    return {
        "persistence_ready": now >= entry_request_s + 10.0,
        "waiting_ready": waiting_mean <= 0.5,
        "preemption_ready": recent_preemptions == 0,
        "current_le070": float(row["scheduler_used_kv_blocks"]) / FP16_POLICY_REFERENCE_BLOCKS
        <= 0.70,
        "dwell_ready": le070_fraction >= 0.90,
        "waiting_mean": waiting_mean,
        "le070_fraction": le070_fraction,
        "recent_preemptions": recent_preemptions,
    }


def elapsed(row: dict[str, Any] | None) -> float | None:
    return None if row is None else float(row["elapsed_s"])


def nonnegative_delta(later: float | None, earlier: float | None) -> float | None:
    if later is None or earlier is None:
        return None
    return max(0.0, later - earlier)


def episode_start(
    rows: list[dict[str, Any]], target: dict[str, Any], predicate: Callable[[dict[str, Any]], bool]
) -> float | None:
    target_index = rows.index(target)
    if not predicate(rows[target_index]):
        return None
    start_index = target_index
    while start_index > 0 and predicate(rows[start_index - 1]):
        start_index -= 1
    return float(rows[start_index]["elapsed_s"])


def replay_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = read_json(run_dir / "metadata.json")
    rows = read_jsonl(run_dir / "controller.jsonl")
    transitions = read_jsonl(run_dir / "transitions.jsonl")
    if not rows or not transitions or transitions[0]["direction"] != "FP16_TO_AWQ_MARLIN_W4_16":
        raise ValueError(f"{run_dir.name} lacks the required dynamic entry evidence")

    duration = float(metadata["measurement_duration_s"])
    start_ns = int(metadata["measurement_start_time_ns"])
    entry = transitions[0]
    entry_request_s = float(entry["requested_elapsed_s"])
    awq_stable_s = (int(entry["model_transition_ended_ns"]) - start_ns) / 1e9
    post_awq = [row for row in rows if float(row["elapsed_s"]) >= awq_stable_s]
    first_awq_sample_s = float(post_awq[0]["elapsed_s"])
    segments = telemetry_segments(rows, duration)

    trajectory: list[dict[str, Any]] = []
    candidate_first: dict[str, dict[str, Any] | None] = {
        name: None for name in CANDIDATES
    }
    old_first: dict[str, dict[str, Any] | None] = {
        name: None
        for name in (
            "persistence",
            "waiting",
            "preemption",
            "current_le070",
            "dwell",
            "all",
        )
    }

    for row in post_awq:
        old = old_stats(rows, segments, row, entry_request_s)
        if old_first["persistence"] is None and old["persistence_ready"]:
            old_first["persistence"] = row
        if old_first["waiting"] is None and old["waiting_ready"]:
            old_first["waiting"] = row
        if old_first["preemption"] is None and old["preemption_ready"]:
            old_first["preemption"] = row
        if old_first["current_le070"] is None and old["current_le070"]:
            old_first["current_le070"] = row
        if old_first["dwell"] is None and old["dwell_ready"]:
            old_first["dwell"] = row
        old_all = bool(
            old["persistence_ready"]
            and old["waiting_ready"]
            and old["preemption_ready"]
            and old["dwell_ready"]
        )
        if old_first["all"] is None and old_all:
            old_first["all"] = row

        derived: dict[str, Any] = {
            "run_id": run_dir.name,
            "workload_class": metadata["workload_class"],
            "elapsed_s": float(row["elapsed_s"]),
            "time_since_awq_stable_s": float(row["elapsed_s"]) - awq_stable_s,
            "waiting_q_depth": int(row["waiting_q_depth"]),
            "running_q_count": int(row["running_q_count"]),
            "swapped_q_count": int(row["swapped_q_count"]),
            "preemption_count": int(row["preemption_count"]),
            "scheduler_used_kv_blocks": int(row["scheduler_used_kv_blocks"]),
            "physical_used_kv_blocks": int(row["physical_used_kv_blocks"]),
            "fp16_reference_utilization": float(row["scheduler_used_kv_blocks"])
            / FP16_POLICY_REFERENCE_BLOCKS,
            "physical_margin_to_1759_blocks": RUNTIME_FP16_BLOCKS
            - int(row["physical_used_kv_blocks"]),
            "physical_shrink_feasible": int(row["physical_used_kv_blocks"])
            <= RUNTIME_FP16_BLOCKS,
            "old_persistence_ready": old["persistence_ready"],
            "old_waiting_mean_10s": old["waiting_mean"],
            "old_waiting_ready": old["waiting_ready"],
            "old_recent_preemptions_10s": old["recent_preemptions"],
            "old_preemption_ready": old["preemption_ready"],
            "old_current_le070": old["current_le070"],
            "old_le070_fraction_10s": old["le070_fraction"],
            "old_dwell90_ready": old["dwell_ready"],
            "old_release_all_ready": old_all,
        }
        for name, spec in CANDIDATES.items():
            stats = candidate_stats(rows, segments, row, spec)
            full_post_awq_window = float(row["elapsed_s"]) >= first_awq_sample_s + float(
                spec["history_window_s"]
            )
            ready = bool(full_post_awq_window and stats["satisfied"])
            derived[f"{name}_waiting_mean"] = stats["waiting_mean"]
            derived[f"{name}_recent_preemptions"] = stats["recent_preemptions"]
            derived[f"{name}_recent_utilization_mean"] = stats[
                "recent_utilization_mean"
            ]
            derived[f"{name}_prior_utilization_mean"] = stats[
                "prior_utilization_mean"
            ]
            derived[f"{name}_utilization_ratio"] = stats["utilization_ratio"]
            derived[f"{name}_ready"] = ready
            if ready and candidate_first[name] is None:
                candidate_first[name] = row
        trajectory.append(derived)

    entry_context = (entry.get("engine_trace") or {}).get("engine_context_after") or {}
    physical_feasible_at_entry = (
        int(entry_context.get("allocated_gpu_kv_blocks", RUNTIME_FP16_BLOCKS + 1))
        <= RUNTIME_FP16_BLOCKS
    )
    physical_first = first_row(
        post_awq,
        awq_stable_s,
        lambda row: int(row["physical_used_kv_blocks"]) <= RUNTIME_FP16_BLOCKS,
    )
    pressure_ready_s = max(
        value
        for value in (elapsed(old_first["waiting"]), elapsed(old_first["preemption"]))
        if value is not None
    )
    old_release_s = elapsed(old_first["all"])
    physical_first_s = awq_stable_s if physical_feasible_at_entry else elapsed(physical_first)
    current_le070_s = elapsed(old_first["current_le070"])
    dwell_s = elapsed(old_first["dwell"])
    persistence_s = elapsed(old_first["persistence"])
    threshold_episode_target = old_first["dwell"] or (
        post_awq[-1] if post_awq and old_stats(rows, segments, post_awq[-1], entry_request_s)["current_le070"] else None
    )
    qualifying_threshold_episode_s = (
        None
        if threshold_episode_target is None
        else episode_start(
            post_awq,
            threshold_episode_target,
            lambda row: float(row["scheduler_used_kv_blocks"])
            / FP16_POLICY_REFERENCE_BLOCKS
            <= 0.70,
        )
    )
    restore = next(
        (
            event
            for event in transitions
            if event.get("direction") == "AWQ_MARLIN_W4_16_TO_FP16"
        ),
        None,
    )
    restore_context = ((restore or {}).get("engine_trace") or {}).get(
        "engine_context_before"
    ) or {}

    summary: dict[str, Any] = {
        "run_id": run_dir.name,
        "workload_class": metadata["workload_class"],
        "measurement_duration_s": duration,
        "entry_request_s": entry_request_s,
        "awq_stable_s": awq_stable_s,
        "first_stable_awq_controller_sample_s": first_awq_sample_s,
        "waiting_queue_area_post_awq_request_s": integrate(
            segments,
            lambda value: float(value["waiting_q_depth"]),
            start=awq_stable_s,
            end=duration,
        ),
        "preemptions_at_awq_stable": counter_value_at(
            rows, "preemption_count", awq_stable_s
        ),
        "preemptions_at_end": int(rows[-1]["preemption_count"]),
        "first_old_persistence_ready_s": persistence_s,
        "first_old_waiting_ready_s": elapsed(old_first["waiting"]),
        "first_old_preemption_ready_s": elapsed(old_first["preemption"]),
        "first_pressure_clearance_ready_s": pressure_ready_s,
        "physical_feasible_at_awq_entry_completion": physical_feasible_at_entry,
        "allocated_blocks_at_awq_entry_completion": entry_context.get(
            "allocated_gpu_kv_blocks"
        ),
        "first_physical_shrink_feasible_s": physical_first_s,
        "first_old_current_le070_s": current_le070_s,
        "qualifying_or_terminal_le070_episode_start_s": qualifying_threshold_episode_s,
        "first_old_dwell90_ready_s": dwell_s,
        "first_old_release_rule_true_s": old_release_s,
        "observed_restore_request_to_hot_start_s": (
            None if restore is None else restore.get("pending_or_drain_duration_s")
        ),
        "observed_restore_allocated_blocks_at_hot_start": restore_context.get(
            "allocated_gpu_kv_blocks"
        ),
        "terminal_old_le070_fraction": trajectory[-1]["old_le070_fraction_10s"],
        "old_release_censored_at_measurement_end": old_release_s is None,
        # Barrier lags are intentionally parallel/non-additive. They expose
        # which condition was still blocking without pretending observational
        # replay is a causal mediation experiment.
        "persistence_barrier_from_entry_request_s": nonnegative_delta(
            persistence_s, entry_request_s
        ),
        "waiting_preemption_barrier_from_awq_stable_s": nonnegative_delta(
            pressure_ready_s, awq_stable_s
        ),
        "physical_feasibility_barrier_from_awq_stable_s": nonnegative_delta(
            physical_first_s, awq_stable_s
        ),
        "le070_first_crossing_lag_after_first_physical_feasibility_s": nonnegative_delta(
            current_le070_s, physical_first_s
        ),
        "le070_qualifying_episode_lag_after_first_physical_feasibility_s": nonnegative_delta(
            qualifying_threshold_episode_s, physical_first_s
        ),
        "dwell90_lag_after_qualifying_le070_episode_s": nonnegative_delta(
            dwell_s or duration, qualifying_threshold_episode_s
        ),
        "old_release_lag_after_pressure_clearance_s": nonnegative_delta(
            old_release_s, pressure_ready_s
        ),
        "old_release_lag_after_first_physical_feasibility_s": nonnegative_delta(
            old_release_s, physical_first_s
        ),
        "persistence_incremental_block_at_old_release_s": (
            None
            if old_release_s is None
            else max(
                0.0,
                persistence_s
                - max(
                    elapsed(old_first["waiting"]) or awq_stable_s,
                    elapsed(old_first["preemption"]) or awq_stable_s,
                    dwell_s or awq_stable_s,
                ),
            )
        ),
    }

    for name, first in candidate_first.items():
        first_s = elapsed(first)
        feasible_after = (
            None
            if first is None
            else first_row(
                post_awq,
                first_s or awq_stable_s,
                lambda row: int(row["physical_used_kv_blocks"])
                <= RUNTIME_FP16_BLOCKS,
            )
        )
        summary.update(
            {
                f"{name}_first_true_s": first_s,
                f"{name}_lead_vs_old_release_s": (
                    None
                    if first_s is None or old_release_s is None
                    else old_release_s - first_s
                ),
                f"{name}_scheduler_used_blocks_at_intent": (
                    None if first is None else int(first["scheduler_used_kv_blocks"])
                ),
                f"{name}_physical_used_blocks_at_intent": (
                    None if first is None else int(first["physical_used_kv_blocks"])
                ),
                f"{name}_blocks_above_1759_at_intent": (
                    None
                    if first is None
                    else int(first["physical_used_kv_blocks"])
                    - RUNTIME_FP16_BLOCKS
                ),
                f"{name}_first_observed_feasible_after_intent_s": elapsed(feasible_after),
                f"{name}_observed_feasibility_lag_s": nonnegative_delta(
                    elapsed(feasible_after), first_s
                ),
            }
        )
    return summary, trajectory


def v7_regime_rows(v7_root: Path) -> list[dict[str, Any]]:
    serving_path = v7_root / "analysis" / "serving_runs.csv"
    crossover_path = v7_root / "analysis" / "crossover_points.csv"
    with serving_path.open(newline="", encoding="utf-8") as handle:
        serving = list(csv.DictReader(handle))
    with crossover_path.open(newline="", encoding="utf-8") as handle:
        crossover = list(csv.DictReader(handle))
    classification = {
        float(row["scale"]): row.get("classification") for row in crossover
    }
    fields = (
        "run_id",
        "condition",
        "scale",
        "nominal_offered_rps",
        "repeat",
        "measurement_duration_s",
        "ttft_s_p95",
        "queueing_delay_s_p95",
        "waiting_queue_area_request_s",
        "preemption_count",
        "peak_logical_kv_utilization",
    )
    return [
        {
            **{field: row.get(field) for field in fields},
            "classification": classification.get(float(row["scale"])),
            "development_role": (
                "validated_low_regime"
                if float(row["scale"]) == 5.75
                else "validated_high_regime"
            ),
        }
        for row in serving
        if float(row["scale"]) in (5.75, 4.25)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v9-root",
        type=Path,
        default=Path("benchmark-results/closed-loop-runtime-v9"),
    )
    parser.add_argument(
        "--v7-root",
        type=Path,
        default=Path("benchmark-results/fp16-awq-crossover-v7"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    decomposition: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    for run_id in DYNAMIC_RUN_IDS:
        summary, run_trajectory = replay_run(args.v9_root / "runs" / run_id)
        decomposition.append(summary)
        trajectory.extend(run_trajectory)

    write_csv(args.output_dir / "v9_release_delay_decomposition.csv", decomposition)
    write_csv(args.output_dir / "v9_release_trajectory.csv", trajectory)
    write_csv(args.output_dir / "v7_operating_regimes.csv", v7_regime_rows(args.v7_root))
    result = {
        "schema_version": 1,
        "status": "development_replay_not_final_validation",
        "warning": (
            "Candidate timestamps are retrospective first-fire times on frozen v9 traces. "
            "Earlier release would change execution, so held-out causal runs are mandatory."
        ),
        "capacities": {
            "policy_reference_fp16_blocks": FP16_POLICY_REFERENCE_BLOCKS,
            "physical_runtime_fp16_blocks": RUNTIME_FP16_BLOCKS,
        },
        "candidate_count": len(CANDIDATES),
        "candidates": CANDIDATES,
        "barrier_interpretation": (
            "Reported barrier lags are parallel and non-additive because threshold, dwell, "
            "queue/preemption, persistence, and feasibility overlap. The ordered threshold "
            "and dwell lags are descriptive, not randomized causal mediation."
        ),
        "runs": decomposition,
    }
    (args.output_dir / "v9_release_delay_decomposition.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
