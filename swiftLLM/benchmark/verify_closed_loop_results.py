"""Post-result independent recomputation of the decisive v9 claims from raw files."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Callable


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values: list[float], value: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * value / 100
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def segments(rows: list[dict[str, Any]], duration: float):
    ordered = sorted(rows, key=lambda row: float(row["elapsed_s"]))
    for index, row in enumerate(ordered):
        left = max(0.0, float(row["elapsed_s"]))
        right = min(duration, float(ordered[index + 1]["elapsed_s"]) if index + 1 < len(ordered) else duration)
        if right > left:
            yield left, right, row


def integrate(rows: list[dict[str, Any]], duration: float, value: Callable[[dict[str, Any]], float], start: float = 0.0, end: float | None = None) -> float:
    stop = duration if end is None else end
    return sum(
        max(0.0, min(right, stop) - max(left, start)) * value(row)
        for left, right, row in segments(rows, duration)
    )


def window(rows: list[dict[str, Any]], duration: float, now: float, width: float) -> dict[str, float]:
    start = max(0.0, now - width)
    elapsed = now - start
    return {
        "wait_mean": integrate(rows, duration, lambda row: float(row["waiting_q_depth"]), start, now) / elapsed,
        "wait_ge4": integrate(rows, duration, lambda row: float(int(row["waiting_q_depth"]) >= 4), start, now) / elapsed,
        "kv_ge095": integrate(rows, duration, lambda row: float(float(row["scheduler_used_kv_blocks"]) / 1768 >= 0.95), start, now) / elapsed,
        "kv_le070": integrate(rows, duration, lambda row: float(float(row["scheduler_used_kv_blocks"]) / 1768 <= 0.70), start, now) / elapsed,
    }


def counter_at(rows: list[dict[str, Any]], timestamp: float) -> int:
    value = 0
    for row in rows:
        if float(row["elapsed_s"]) > timestamp:
            break
        value = int(row["preemption_count"])
    return value


def recompute_triggers(telemetry: list[dict[str, Any]], duration: float, transitions: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    first_entry = None
    for row in telemetry:
        now = float(row["elapsed_s"])
        if now < 3:
            continue
        stats = window(telemetry, duration, now, 3.0)
        if stats["kv_ge095"] >= 0.80 and stats["wait_ge4"] >= 0.50 and int(row["waiting_q_depth"]) >= 4:
            first_entry = now
            break
    if not transitions:
        return first_entry, None
    entry_request = float(transitions[0]["requested_elapsed_s"])
    entry_end_ns = transitions[0].get("model_transition_ended_ns")
    first_release = None
    for row in telemetry:
        now = float(row["elapsed_s"])
        if now < entry_request + 10 or row.get("runtime_precision_state") != "AWQ_MARLIN_W4_16":
            continue
        stats = window(telemetry, duration, now, 10.0)
        new_preemptions = int(row["preemption_count"]) - counter_at(telemetry, now - 10.0)
        if stats["wait_mean"] <= 0.5 and new_preemptions == 0 and stats["kv_le070"] >= 0.90:
            first_release = now
            break
    return first_entry, first_release


def normalize_chars(text: str) -> list[str]:
    return [char for char in text if char.strip()]


def f1(answer: str, references: list[str]) -> float:
    pred = normalize_chars(answer)
    scores = []
    for reference in references:
        ref = normalize_chars(reference)
        same = sum((Counter(pred) & Counter(ref)).values()) if pred and ref else 0
        if not same:
            scores.append(0.0)
        else:
            precision = same / len(pred)
            recall = same / len(ref)
            scores.append(2 * precision * recall / (precision + recall))
    return max(scores, default=0.0)


def bootstrap(values: list[float], seed: int) -> tuple[float, float]:
    generator = random.Random(seed)
    count = len(values)
    means = [sum(values[generator.randrange(count)] for _ in range(count)) / count for _ in range(10_000)]
    return percentile(means, 2.5), percentile(means, 97.5)


def transition_ok(event: dict[str, Any]) -> bool:
    trace = event.get("engine_trace") or {}
    resize = trace.get("kv_resize") or {}
    direction = event.get("direction")
    target = "AWQ_MARLIN_W4_16" if direction == "FP16_TO_AWQ_MARLIN_W4_16" else "FP16" if direction == "AWQ_MARLIN_W4_16_TO_FP16" else None
    blocks = 4170 if target == "AWQ_MARLIN_W4_16" else 1759
    context = trace.get("engine_context_after") or {}
    before = (resize.get("integrity_before") or {}).get("logical_digest")
    after = (resize.get("integrity_after") or {}).get("logical_digest")
    active = int(trace.get("active_request_count", 0))
    digest = (before is not None and before == after) if active else ((before is None and after is None) or before == after)
    return bool(
        event.get("result") == trace.get("status") == "success"
        and trace.get("precision_after") == target
        and trace.get("physical_blocks_after") == blocks
        and context.get("physical_blocks") == context.get("scheduler_visible_blocks") == blocks
        and digest
    )


def close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return abs(left - right) <= tolerance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    plan = read_json(root / "run-plan.json")["runs"]
    status = read_json(root / "execution_status.json")
    selected = status["selected_runs"]
    serving_csv = list(csv.DictReader((root / "analysis" / "serving_runs.csv").open()))
    serving = {(row["workload_class"], row["condition"], int(row["repeat"])): row for row in serving_csv}
    raw: dict[tuple[str, str, int], dict[str, Any]] = {}
    file_hashes: dict[str, dict[str, str]] = {}
    checks: dict[str, bool] = {}

    for item in plan:
        key = (str(item["workload_class"]), str(item["condition"]), int(item["repeat"]))
        actual = selected[item["run_id"]]
        run_dir = root / "runs" / actual
        metadata = read_json(run_dir / "metadata.json")
        requests = read_jsonl(run_dir / "requests.jsonl")
        telemetry = read_jsonl(run_dir / "telemetry.jsonl")
        batches = read_jsonl(run_dir / "batches.jsonl")
        controller = read_jsonl(run_dir / "controller.jsonl")
        transitions = read_jsonl(run_dir / "transitions.jsonl")
        file_hashes[actual] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(run_dir.iterdir()) if path.is_file()
        }
        expected_count = int(item["request_count"])
        batch_counts = Counter(
            request_id
            for batch in batches
            for request_id in batch.get("benchmark_request_ids", [])
            if request_id is not None
        )
        request_ok = len(requests) == expected_count
        for request in requests:
            states = list(request["step_precision_states"])
            positions = list(request["step_input_positions"])
            engine_ids = list(request["engine_request_ids"])
            times = [int(value) for value in request["step_received_time_ns"]]
            expected_positions = list(range(1023, 1535))
            request_ok &= bool(
                request["status"] == "completed"
                and len(request["output_token_ids"]) == 512
                and positions == expected_positions
                and len(states) == len(engine_ids) == len(times) == 512
                and len(set(engine_ids)) == 1
                and set(states) <= {"FP16", "AWQ_MARLIN_W4_16"}
                and states.count("FP16") == int(request["fp16_output_step_count"])
                and states.count("AWQ_MARLIN_W4_16") == int(request["awq_output_step_count"])
                and all(left <= right for left, right in zip(times, times[1:]))
                and batch_counts[request["benchmark_request_id"]] == 512
            )
        checks[f"{actual}:request_arrays_recomputed"] = request_ok
        checks[f"{actual}:arrival_schedule_recomputed"] = all(
            abs(float(row["arrival_jitter_s"])) <= 0.25
            and close(
                float(row["actual_arrival_offset_s"]),
                (int(row["arrival_time_ns"]) - int(metadata["measurement_start_time_ns"])) / 1e9,
            )
            for row in requests
        ) and all(int(left["arrival_time_ns"]) <= int(right["arrival_time_ns"]) for left, right in zip(requests, requests[1:]))
        checks[f"{actual}:transition_endpoints_recomputed"] = all(transition_ok(event) for event in transitions)
        checks[f"{actual}:capacity_and_policy_reference_recomputed"] = all(
            int(row["fp16_policy_reference_blocks"]) == 1768
            and int(row["base_physical_kv_blocks"]) == 1759
            and 0 <= int(row["physical_used_kv_blocks"]) <= int(row["physical_total_kv_blocks"])
            and (
                row["runtime_precision_state"] not in {"FP16", "AWQ_MARLIN_W4_16"}
                or int(row["physical_total_kv_blocks"])
                == (1759 if row["runtime_precision_state"] == "FP16" else 4170)
            )
            for row in telemetry
        )

        duration = float(metadata["measurement_duration_s"])
        ttft = [(int(row["first_stream_token_received_time_ns"]) - int(row["arrival_time_ns"])) / 1e9 for row in requests]
        queue = [(int(row["first_prefill_time_ns"]) - int(row["scheduler_eligible_time_ns"])) / 1e9 for row in requests]
        recomputed = {
            "p95": percentile(ttft, 95),
            "p99": percentile(ttft, 99),
            "slo": sum(value > 2 for value in ttft) / len(ttft),
            "queue95": percentile(queue, 95),
            "waiting_area": integrate(telemetry, duration, lambda row: float(row["waiting_q_depth"])),
            "kv_dwell": integrate(telemetry, duration, lambda row: float(float(row["scheduler_used_kv_blocks"]) / 1768 >= 0.95)),
            "preemptions": max(int(row["preemption_count"]) for row in telemetry),
            "throughput": len(requests) / duration,
            "awq_fraction": sum(states == "AWQ_MARLIN_W4_16" for row in requests for states in row["step_precision_states"]) / (len(requests) * 512),
            "transitions": transitions,
            "requests": requests,
            "metadata": metadata,
        }
        saved = serving[key]
        checks[f"{actual}:serving_metrics_recomputed"] = all(
            close(recomputed_value, float(saved[saved_field]), 1e-8)
            for recomputed_value, saved_field in (
                (recomputed["p95"], "ttft_s_p95"),
                (recomputed["p99"], "ttft_s_p99"),
                (recomputed["slo"], "strict_ttft_gt_2_rate"),
                (recomputed["queue95"], "queueing_delay_s_p95"),
                (recomputed["waiting_area"], "waiting_queue_area_request_s"),
                (recomputed["kv_dwell"], "fp16_equiv_kv_ge095_dwell_s"),
                (recomputed["throughput"], "completed_request_throughput_rps"),
                (recomputed["awq_fraction"], "awq_output_token_fraction"),
            )
        ) and recomputed["preemptions"] == int(saved["preemption_count"])
        if item["condition"] == "closed_loop_dynamic":
            entry, release = recompute_triggers(telemetry, duration, transitions)
            logged_entry = next((float(row["elapsed_s"]) for row in controller if row.get("entry_boolean")), None)
            logged_release = next((float(row["elapsed_s"]) for row in controller if row.get("release_boolean")), None)
            checks[f"{actual}:controller_windows_recomputed"] = entry == logged_entry and release == logged_release
        raw[key] = recomputed

    # Recompute the predeclared performance gates.
    performance: dict[str, bool] = {}
    epsilons = {}
    for workload in ("low", "high", "low_high_low"):
        fp = [raw[(workload, "runtime_static_fp16", repeat)]["p95"] for repeat in (0, 1)]
        dy = [raw[(workload, "closed_loop_dynamic", repeat)]["p95"] for repeat in (0, 1)]
        epsilons[workload] = max(0.111, max(fp) - min(fp), max(dy) - min(dy), 0.05 * sum(fp) / 2)
    performance["low_no_false_entry_both"] = all(not raw[("low", "closed_loop_dynamic", repeat)]["transitions"] for repeat in (0, 1))
    performance["low_fp16_like_within_noise_both"] = all(
        abs(raw[("low", "closed_loop_dynamic", repeat)]["p95"] - raw[("low", "runtime_static_fp16", repeat)]["p95"]) <= epsilons["low"]
        for repeat in (0, 1)
    )
    performance["high_enters_awq_both"] = all(
        any(event["direction"] == "FP16_TO_AWQ_MARLIN_W4_16" for event in raw[("high", "closed_loop_dynamic", repeat)]["transitions"])
        for repeat in (0, 1)
    )
    performance["high_improvement_exceeds_noise_both"] = all(
        raw[("high", "runtime_static_fp16", repeat)]["p95"] - raw[("high", "closed_loop_dynamic", repeat)]["p95"] > epsilons["high"]
        for repeat in (0, 1)
    )
    performance["high_queue_area_kv_aligned_both"] = all(
        raw[("high", "closed_loop_dynamic", repeat)]["queue95"] < raw[("high", "runtime_static_fp16", repeat)]["queue95"]
        and raw[("high", "closed_loop_dynamic", repeat)]["waiting_area"] < raw[("high", "runtime_static_fp16", repeat)]["waiting_area"]
        and raw[("high", "closed_loop_dynamic", repeat)]["kv_dwell"] < raw[("high", "runtime_static_fp16", repeat)]["kv_dwell"]
        for repeat in (0, 1)
    )
    performance["high_preemption_aligned"] = all(
        raw[("high", "closed_loop_dynamic", repeat)]["preemptions"] <= raw[("high", "runtime_static_fp16", repeat)]["preemptions"]
        for repeat in (0, 1)
    ) and any(
        raw[("high", "closed_loop_dynamic", repeat)]["preemptions"] < raw[("high", "runtime_static_fp16", repeat)]["preemptions"]
        for repeat in (0, 1)
    )
    performance["high_throughput_gate"] = all(
        raw[("high", "closed_loop_dynamic", repeat)]["throughput"] >= 0.98 * raw[("high", "runtime_static_fp16", repeat)]["throughput"]
        for repeat in (0, 1)
    ) and sum(raw[("high", "closed_loop_dynamic", repeat)]["throughput"] for repeat in (0, 1)) >= sum(
        raw[("high", "runtime_static_fp16", repeat)]["throughput"] for repeat in (0, 1)
    )

    decision = read_json(root / "analysis" / "decision.json")
    checks["performance_gates_recomputed"] = performance == decision["performance_gates"]

    # Recompute F1 and fidelity using 64 question means across serving repeats.
    by_condition: dict[str, dict[str, list[dict[str, Any]]]] = {condition: {} for condition in CONDITIONS}
    for condition in CONDITIONS:
        for repeat in (0, 1):
            for request in raw[("high", condition, repeat)]["requests"]:
                by_condition[condition].setdefault(request["benchmark_request_id"], []).append(request)
    ids = sorted(by_condition["runtime_static_fp16"])
    aggregates = []
    for request_id in ids:
        values = {}
        for condition in CONDITIONS:
            values[condition] = sum(
                f1(str(row["generated_answer"] or ""), [str(value) for value in row["dataset_reference_answers"]])
                for row in by_condition[condition][request_id]
            ) / 2
        aggregates.append(values)
    delta_awq = [row["closed_loop_dynamic"] - row["runtime_static_awq_w4_16"] for row in aggregates]
    delta_ci = bootstrap(delta_awq, 9100)
    quality_csv = {row["condition"]: row for row in csv.DictReader((root / "analysis" / "quality_fidelity_summary.csv").open())}
    checks["quality_f1_and_paired_ci_recomputed"] = all(
        close(100 * sum(row[condition] for row in aggregates) / 64, float(quality_csv[condition]["f1_percent"]), 1e-10)
        for condition in CONDITIONS
    ) and close(100 * delta_ci[0], float(quality_csv["closed_loop_dynamic"]["dynamic_minus_awq_ci95_low_pp"]), 1e-10)

    agreements = {"runtime_static_awq_w4_16": [], "closed_loop_dynamic": []}
    prefixes = {"runtime_static_awq_w4_16": [], "closed_loop_dynamic": []}
    exact = {"runtime_static_awq_w4_16": [], "closed_loop_dynamic": []}
    for repeat in (0, 1):
        fp_requests = {row["benchmark_request_id"]: row for row in raw[("high", "runtime_static_fp16", repeat)]["requests"]}
        for condition in agreements:
            for row in raw[("high", condition, repeat)]["requests"]:
                base = fp_requests[row["benchmark_request_id"]]["output_token_ids"]
                current = row["output_token_ids"]
                agreements[condition].append(sum(left == right for left, right in zip(base, current)) / 512)
                prefix = 0
                for left, right in zip(base, current):
                    if left != right:
                        break
                    prefix += 1
                prefixes[condition].append(prefix)
                exact[condition].append(int(base == current))
    checks["fidelity_recomputed"] = all(
        close(sum(agreements[condition]) / len(agreements[condition]), float(quality_csv[condition]["mean_position_aligned_token_agreement_vs_fp16"]), 1e-12)
        and close(sum(prefixes[condition]) / len(prefixes[condition]), float(quality_csv[condition]["mean_common_prefix_tokens_vs_fp16"]), 1e-12)
        and close(sum(exact[condition]) / len(exact[condition]), float(quality_csv[condition]["exact_output_match_rate_vs_fp16"]), 1e-12)
        for condition in agreements
    )

    # Recompute useful phased restoration and headline table values.
    workload_meta = read_json(root / "input" / "workload_metadata.json")
    high_start = float(workload_meta["phased_boundaries"]["high"]["start_offset_s"])
    recovery_start = float(workload_meta["phased_boundaries"]["low_recovery"]["start_offset_s"])
    final_arrival = float(workload_meta["phased_boundaries"]["low_recovery"]["last_arrival_offset_s"])
    phased_pass = True
    restore_memory = []
    for repeat in (0, 1):
        item = raw[("low_high_low", "closed_loop_dynamic", repeat)]
        events = item["transitions"]
        directions = [event["direction"] for event in events]
        requested = [float(event["requested_elapsed_s"]) for event in events]
        ended = [
            (int(event["model_transition_ended_ns"]) - int(item["metadata"]["measurement_start_time_ns"])) / 1e9
            for event in events
        ]
        restore_end = ended[1] if len(ended) > 1 else None
        later = [row for row in item["requests"] if restore_end is not None and row["phase"] == "low_recovery" and float(row["actual_arrival_offset_s"]) > restore_end]
        phased_pass &= bool(
            directions == ["FP16_TO_AWQ_MARLIN_W4_16", "AWQ_MARLIN_W4_16_TO_FP16"]
            and requested[0] >= high_start
            and requested[1] >= recovery_start
            and restore_end < final_arrival
            and later
        )
        if len(events) > 1:
            restore_memory.append(int(events[1]["engine_trace"]["memory_after"]["allocated_bytes"]))
    checks["phased_reversibility_recomputed"] = phased_pass == decision["systems_checks"]["phased_reversibility_both_repeats"]

    headline = {row["condition"]: row for row in csv.DictReader((root / "analysis" / "headline_table5.csv").open())}
    checks["headline_table_recomputed"] = all(
        close(
            sum(raw[("high", condition, repeat)]["p95"] for repeat in (0, 1)) / 2,
            float(headline[DISPLAY[condition]]["p95_ttft_s_mean"]),
            1e-10,
        )
        for condition in CONDITIONS
    )

    all_raw_valid = all(value for key, value in checks.items() if ":" in key)
    all_transitions_valid = all(
        transition_ok(event)
        for key, item in raw.items() if key[1] == "closed_loop_dynamic"
        for event in item["transitions"]
    )
    no_oscillation = all(len(item["transitions"]) <= 2 for key, item in raw.items() if key[1] == "closed_loop_dynamic")
    hbm_spread = max(restore_memory) - min(restore_memory) if len(restore_memory) == 2 else None
    systems = {
        "all_18_runs_valid": all_raw_valid,
        **performance,
        "all_dynamic_transitions_successful_and_kv_preserved": all_transitions_valid,
        "phased_reversibility_both_repeats": phased_pass,
        "no_closed_loop_oscillation": no_oscillation,
        "restored_fp16_hbm_repeat_spread_le_64mib": hbm_spread is not None and hbm_spread <= 64 * 1024 * 1024,
    }
    systems_go = all(systems.values())
    quality_decision = "STRONG GO" if systems_go and delta_ci[0] > 0 else "SUPPORTED / UNCERTAIN" if systems_go else "NO-GO"
    checks["systems_decision_recomputed"] = systems == decision["systems_checks"] and ("GO" if systems_go else "NO-GO") == decision["systems_adaptation_decision"]
    checks["quality_decision_recomputed"] = quality_decision == decision["quality_latency_tradeoff_decision"]

    result = {
        "schema_version": 1,
        "role": "post-result independent verifier; it does not change the pre-registered policy, workloads, gates, or frozen analyzer",
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "check_count": len(checks),
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "checks": checks,
        "recomputed_performance_gates": performance,
        "recomputed_noise_thresholds_s": epsilons,
        "recomputed_dynamic_minus_awq_f1_ci95_pp": [100 * delta_ci[0], 100 * delta_ci[1]],
        "recomputed_systems_adaptation_decision": "GO" if systems_go else "NO-GO",
        "recomputed_quality_latency_tradeoff_decision": quality_decision,
        "raw_file_sha256": file_hashes,
    }
    output = root / "independent_verification.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "raw_file_sha256"}, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
