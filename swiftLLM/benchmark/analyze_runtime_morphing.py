"""Regenerate runtime-morphing v8 tables and invariant checks from raw JSON."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


MIB = 2 ** 20
LEAK_TOLERANCE_BYTES = 64 * MIB


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    location = (len(ordered) - 1) * percent / 100
    lower = int(location)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = location - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def step_checks(payload: dict) -> tuple[bool, list[dict]]:
    rows = []
    valid = True
    vocab_size = 128256
    for sequence in range(2):
        steps = sorted(
            (row for row in payload["steps"] if row["sequence"] == sequence),
            key=lambda row: row["step_index"],
        )
        indices = [row["step_index"] for row in steps]
        positions = [row["input_position"] for row in steps]
        ids = {row["request_id"] for row in steps}
        one_prefill = sum(bool(row["is_prefill"]) for row in steps) == 1
        row_valid = (
            indices == list(range(len(steps)))
            and all(b == a + 1 for a, b in zip(positions, positions[1:]))
            and len(ids) == 1
            and one_prefill
            and all(0 <= row["output_token_id"] < vocab_size for row in steps)
        )
        valid &= row_valid
        rows.append({
            "condition": payload["condition"],
            "sequence": sequence,
            "step_count": len(steps),
            "request_ids": sorted(ids),
            "step_indices_contiguous": indices == list(range(len(steps))),
            "positions_increase_by_one": all(
                b == a + 1 for a, b in zip(positions, positions[1:])
            ),
            "one_prefill_step": one_prefill,
            "tokens_valid": all(
                0 <= row["output_token_id"] < vocab_size for row in steps
            ),
            "valid": row_valid,
        })
    return valid, rows


def transition_row(trace: dict, source: str) -> dict:
    weights = trace.get("weight_transition", {})
    layers = weights.get("layers", [])
    kv = trace.get("kv_resize", {})
    total_ns = int(trace.get("elapsed_ns", 0))
    layer_ns = sum(int(row["elapsed_ns"]) for row in layers)
    layer_sync_ns = sum(int(row["cuda_sync_ns"]) for row in layers)
    boundary_sync_ns = int(trace.get("boundary_cuda_sync_ns", 0))
    kv_ns = int(kv.get("elapsed_ns", 0))
    cleanup_ns = int(weights.get("cleanup_ns", 0))
    return {
        "source": source,
        "transition_id": trace.get("transition_id"),
        "direction": trace["direction"],
        "status": trace["status"],
        "active_request_count": trace.get("active_request_count", 0),
        "used_kv_blocks": trace.get("used_kv_blocks", 0),
        "total_ms": total_ns / 1e6,
        "boundary_cuda_sync_ms": boundary_sync_ns / 1e6,
        "layer_total_ms": layer_ns / 1e6,
        "layer_cuda_sync_ms": layer_sync_ns / 1e6,
        "weight_cleanup_ms": cleanup_ns / 1e6,
        "kv_resize_ms": kv_ns / 1e6,
        "bookkeeping_residual_ms": max(
            0, total_ns - boundary_sync_ns - layer_ns - cleanup_ns - kv_ns
        ) / 1e6,
        "h2d_bytes": int(weights.get("h2d_bytes", 0)),
        "d2h_bytes": int(kv.get("d2h_bytes", 0)),
        "d2d_bytes": int(kv.get("d2d_bytes", 0)),
        "per_layer_count": len(layers),
        "base_blocks": trace.get("base_blocks"),
        "physical_blocks_after": trace.get("physical_blocks_after"),
        "allocated_bytes_before": trace["memory_before"]["allocated_bytes"],
        "allocated_bytes_after": trace["memory_after"]["allocated_bytes"],
        "reserved_bytes_before": trace["memory_before"]["reserved_bytes"],
        "reserved_bytes_after": trace["memory_after"]["reserved_bytes"],
        "transition_peak_allocated_bytes": trace.get("transition_peak_allocated_bytes"),
        "max_layer_coexist_allocated_bytes": max(
            (
                row.get("memory_with_both_representations", {}).get("allocated_bytes", 0)
                for row in layers
            ),
            default=0,
        ),
        "free_bytes_before": trace["memory_before"]["free_bytes"],
        "free_bytes_after": trace["memory_after"]["free_bytes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("benchmark-results/runtime-morphing-v8")
    )
    args = parser.parse_args()
    raw = args.root / "raw"
    analysis = args.root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)

    fp16 = load(raw / "state-fp16.json")
    static_awq = load(raw / "state-static-awq.json")
    morph = load(raw / "state-morph.json")
    roundtrip = load(raw / "state-roundtrip.json")
    cycles = load(raw / "capacity-cycles.json")

    checks: dict[str, bool] = {}
    payloads = (fp16, static_awq, morph, roundtrip, cycles)
    checks["all_raw_runs_same_rtx3090_device5"] = all(
        payload["gpu"]["name"] == "NVIDIA GeForce RTX 3090"
        and payload["gpu"]["cuda_visible_devices"] == "5"
        for payload in payloads
    )
    checks["raw_runs_share_source_provenance"] = all(
        payload["provenance"] == fp16["provenance"] for payload in payloads
    )
    checks["raw_source_hashes_match_current_files"] = all(
        Path(relative).is_file()
        and hashlib.sha256(Path(relative).read_bytes()).hexdigest() == digest
        for relative, digest in fp16["provenance"]["runtime_source_files_sha256"].items()
    )
    checks["identical_prompts_all_conditions"] = (
        fp16["prompt_token_ids"]
        == static_awq["prompt_token_ids"]
        == morph["prompt_token_ids"]
        == roundtrip["prompt_token_ids"]
    )
    step_rows = []
    for payload in (fp16, static_awq, morph, roundtrip):
        valid, rows = step_checks(payload)
        checks[f"{payload['condition']}_step_integrity"] = valid
        step_rows.extend(rows)

    for payload in (morph, roundtrip):
        prefill_counts = {f"runtime-r{i}": 0 for i in range(2)}
        for event in payload["batch_events"]:
            if event["num_prefill_sequences"]:
                for request_id in event["benchmark_request_ids"]:
                    prefill_counts[request_id] += 1
        checks[f"{payload['condition']}_one_prefill_per_request"] = all(
            count == 1 for count in prefill_counts.values()
        )
        checks[f"{payload['condition']}_ends_without_active_requests"] = (
            payload["final_snapshot"]["running_q_count"] == 0
            and payload["final_snapshot"]["swapped_q_count"] == 0
        )
        for index, trace in enumerate(payload["transition_boundaries"]):
            context = trace["engine_context_before"]
            checks[f"{payload['condition']}_transition_{index}_two_active"] = (
                context["active_request_count"] == 2
                and len(context["active_request_states"]) == 2
            )
            checks[f"{payload['condition']}_transition_{index}_layer_coverage"] = (
                len(trace["weight_transition"]["layers"]) == 16
            )
            before = trace["kv_resize"].get("integrity_before")
            after = trace["kv_resize"].get("integrity_after")
            if before is not None and after is not None:
                checks[f"{payload['condition']}_transition_{index}_kv_integrity"] = (
                    before["logical_digest"] == after["logical_digest"]
                )

    morph_boundary = morph["transition_boundaries"][0]["engine_context_before"][
        "active_request_states"
    ]
    round_boundary = roundtrip["transition_boundaries"][0]["engine_context_before"][
        "active_request_states"
    ]
    checks["morph_fp16_prefix_matches_static"] = all(
        morph["output_token_ids"][sequence][
            :morph_boundary[sequence]["output_len"]
        ]
        == fp16["output_token_ids"][sequence][
            :morph_boundary[sequence]["output_len"]
        ]
        for sequence in range(2)
    )
    checks["roundtrip_fp16_prefix_matches_static"] = all(
        roundtrip["output_token_ids"][sequence][
            :round_boundary[sequence]["output_len"]
        ]
        == fp16["output_token_ids"][sequence][
            :round_boundary[sequence]["output_len"]
        ]
        for sequence in range(2)
    )
    morph_prefix_lengths = {
        row["request_id"]: row["output_len"] for row in morph_boundary
    }
    checks["morph_has_fp16_then_awq_steps"] = all(
        [row["precision_state"] for row in morph["steps"] if row["sequence"] == sequence]
        == ["FP16"] * morph_prefix_lengths[sequence]
        + ["AWQ_MARLIN_W4_16"] * (24 - morph_prefix_lengths[sequence])
        for sequence in range(2)
    )
    round_restore_boundary = roundtrip["transition_boundaries"][1][
        "engine_context_before"
    ]["active_request_states"]
    round_prefix_lengths = {
        row["request_id"]: row["output_len"] for row in round_boundary
    }
    round_restore_lengths = {
        row["request_id"]: row["output_len"] for row in round_restore_boundary
    }
    checks["roundtrip_has_fp16_awq_fp16_steps"] = all(
        [row["precision_state"] for row in roundtrip["steps"] if row["sequence"] == sequence]
        == ["FP16"] * round_prefix_lengths[sequence]
        + ["AWQ_MARLIN_W4_16"]
        * (round_restore_lengths[sequence] - round_prefix_lengths[sequence])
        + ["FP16"] * (24 - round_restore_lengths[sequence])
        for sequence in range(2)
    )

    checks["static_fp16_capacity_1768"] = fp16["num_gpu_blocks"] == 1768
    checks["static_awq_capacity_4286"] = static_awq["num_gpu_blocks"] == 4286
    checks["dynamic_awq_capacity_is_physical_and_safe"] = (
        cycles["dynamic_awq_blocks"] > cycles["base_blocks"]
        and all(
            row["snapshot_after_morph"]["physical_blocks"]
            == cycles["dynamic_awq_blocks"]
            and row["snapshot_after_morph"]["scheduler_visible_blocks"]
            == cycles["dynamic_awq_blocks"]
            for row in cycles["cycle_rows"]
        )
    )
    checks["restored_capacity_is_physical_base"] = all(
        row["snapshot_after_restore"]["physical_gpu_blocks"] == cycles["base_blocks"]
        and row["snapshot_after_restore"]["num_gpu_blocks"] == cycles["base_blocks"]
        and row["snapshot_after_restore"]["extension_gpu_blocks"] == 0
        for row in cycles["cycle_rows"]
    )
    extension_execution = cycles["full_model_extension_execution"]
    checks["full_model_crosses_kv_segments"] = (
        extension_execution["block_table_ids_before_restore"][0]
        < cycles["base_blocks"]
        <= extension_execution["block_table_ids_before_restore"][1]
        and extension_execution["finite_valid_tokens_before_restore"]
        and extension_execution["output_token_ids_before_restore"]
        == static_awq["output_token_ids"][0][:5]
    )
    checks["live_extension_state_restored_and_consumed"] = (
        extension_execution["logical_kv_before_restore"]["logical_digest"]
        == extension_execution["logical_kv_after_restore"]["logical_digest"]
        and all(
            block_id < cycles["base_blocks"]
            for block_id in extension_execution["block_table_ids_after_restore"]
        )
        and extension_execution["request_id_before_after"] == [0, 0]
        and extension_execution["post_restore_token_valid"]
        and len(extension_execution["output_token_ids_after_restore"]) == 6
    )
    envelope = cycles["max_shape_capacity_validation"]
    checks["dynamic_capacity_passes_max_shape_envelope"] = (
        envelope["physical_blocks_during_probe"] == cycles["dynamic_awq_blocks"]
        and envelope["batch_size"] == 32
        and envelope["token_count"] == 49152
        and envelope["ignore_kvcache"]
        and envelope["outputs_valid"]
        and envelope["active_kv_integrity_before"]["logical_digest"]
        == envelope["active_kv_integrity_after"]["logical_digest"]
        and envelope["driver_used_bytes_at_envelope"] <= envelope["usable_budget_bytes"]
        and envelope["derived_safe_total_blocks"] >= cycles["dynamic_awq_blocks"]
    )
    checks["awq_runtime_byte_exact"] = (
        cycles["awq_runtime_equivalence"]["byte_exact"]
        and cycles["awq_runtime_equivalence"]["observed_sha256"]
        == static_awq["selected_layer_digest"]["sha256"]
        and cycles["awq_runtime_equivalence"]["compared_bytes"]
        == static_awq["selected_layer_digest"]["compared_bytes"]
    )
    checks["fp16_restore_byte_exact"] = (
        cycles["fp16_runtime_equivalence"]["byte_exact"]
        and cycles["fp16_runtime_equivalence"]["observed_sha256"]
        == fp16["selected_layer_digest"]["sha256"]
        and cycles["fp16_runtime_equivalence"]["compared_bytes"]
        == fp16["selected_layer_digest"]["compared_bytes"]
    )
    checks["every_cycle_remaps_extension_without_kv_change"] = all(
        row["kv_integrity_before"]["logical_digest"]
        == row["kv_integrity_after"]["logical_digest"]
        and any(value >= cycles["base_blocks"] for value in row["extension_ids_before_restore"])
        and all(value < cycles["base_blocks"] for value in row["base_ids_after_restore"])
        for row in cycles["cycle_rows"]
    )

    restored_allocated = [
        row["memory_after_restore"]["allocated_bytes"] for row in cycles["cycle_rows"]
    ]
    restored_reserved = [
        row["memory_after_restore"]["reserved_bytes"] for row in cycles["cycle_rows"]
    ]
    allocated_spread = max(restored_allocated) - min(restored_allocated)
    reserved_spread = max(restored_reserved) - min(restored_reserved)
    allocated_final_drift = abs(restored_allocated[-1] - restored_allocated[0])
    reserved_final_drift = abs(restored_reserved[-1] - restored_reserved[0])
    checks["repeated_cycles_have_no_monotonic_memory_leak"] = (
        allocated_final_drift <= LEAK_TOLERANCE_BYTES
        and reserved_final_drift <= LEAK_TOLERANCE_BYTES
        and not all(b > a for a, b in zip(restored_allocated, restored_allocated[1:]))
        and not all(b > a for a, b in zip(restored_reserved, restored_reserved[1:]))
    )

    transition_rows = []
    raw_transitions = []
    for source, payload, key in (
        ("state-morph", morph, "transition_boundaries"),
        ("state-roundtrip", roundtrip, "transition_boundaries"),
        ("capacity-cycles", cycles, "transition_traces"),
    ):
        for trace in payload[key]:
            transition_rows.append(transition_row(trace, source))
            raw_transitions.append(trace)
    checks["all_transitions_successful"] = all(
        row["status"] == "success" and row["per_layer_count"] == 16
        for row in transition_rows
    )

    controlled = [row for row in transition_rows if row["source"] == "capacity-cycles"]
    summaries = {}
    for direction in sorted({row["direction"] for row in controlled}):
        selected = [row for row in controlled if row["direction"] == direction]
        summaries[direction] = {
            "count": len(selected),
            "total_ms": {
                "min": min(row["total_ms"] for row in selected),
                "median": statistics.median(row["total_ms"] for row in selected),
                "p95": percentile([row["total_ms"] for row in selected], 95),
                "max": max(row["total_ms"] for row in selected),
            },
            "layer_total_ms_median": statistics.median(
                row["layer_total_ms"] for row in selected
            ),
            "layer_cuda_sync_ms_median": statistics.median(
                row["layer_cuda_sync_ms"] for row in selected
            ),
            "weight_cleanup_ms_median": statistics.median(
                row["weight_cleanup_ms"] for row in selected
            ),
            "kv_resize_ms_median": statistics.median(
                row["kv_resize_ms"] for row in selected
            ),
            "boundary_cuda_sync_ms_median": statistics.median(
                row["boundary_cuda_sync_ms"] for row in selected
            ),
            "bookkeeping_residual_ms_median": statistics.median(
                row["bookkeeping_residual_ms"] for row in selected
            ),
            "transition_peak_allocated_bytes_max": max(
                row["transition_peak_allocated_bytes"] for row in selected
            ),
            "h2d_bytes": sorted({row["h2d_bytes"] for row in selected}),
        }

    with (
        Path("benchmark-results/fp16-awq-crossover-v7/analysis/crossover_points.csv")
    ).open(newline="", encoding="utf-8") as handle:
        crossover = list(csv.DictReader(handle))
    advantages = [
        float(row[field])
        for row in crossover
        if row["classification"] == "AWQ-preferred"
        for field in ("p95_advantage_rep0_s", "p95_advantage_rep1_s")
    ]
    one_way_median_s = summaries["FP16_TO_AWQ_MARLIN_W4_16"]["total_ms"]["median"] / 1000
    restore_median_s = summaries["AWQ_MARLIN_W4_16_TO_FP16"]["total_ms"]["median"] / 1000
    summaries["v7_amortization_estimate"] = {
        "v7_awq_preferred_p95_advantage_s": {
            "min": min(advantages),
            "median": statistics.median(advantages),
            "max": max(advantages),
        },
        "one_way_transition_cost_s": one_way_median_s,
        "round_trip_transition_cost_s": one_way_median_s + restore_median_s,
        "one_way_break_even_request_equivalents": {
            "at_min_v7_advantage": one_way_median_s / min(advantages),
            "at_median_v7_advantage": one_way_median_s / statistics.median(advantages),
        },
        "round_trip_break_even_request_equivalents": {
            "at_min_v7_advantage": (one_way_median_s + restore_median_s) / min(advantages),
            "at_median_v7_advantage": (one_way_median_s + restore_median_s) / statistics.median(advantages),
        },
        "caveat": "Request-equivalent estimate compares blocking transition wall time with static per-request P95 TTFT advantage; it is not a closed-loop throughput measurement.",
    }

    capacity_rows = [
        {
            "state": "static_fp16",
            "base_blocks": fp16["num_gpu_blocks"],
            "extension_blocks": 0,
            "total_physical_blocks": fp16["num_gpu_blocks"],
            "scheduler_visible_blocks": fp16["num_gpu_blocks"],
            "physical_kv_bytes": fp16["physical_kv_bytes"],
        },
        {
            "state": "runtime_fp16",
            "base_blocks": cycles["base_blocks"],
            "extension_blocks": 0,
            "total_physical_blocks": cycles["base_blocks"],
            "scheduler_visible_blocks": cycles["base_blocks"],
            "physical_kv_bytes": cycles["base_physical_kv_bytes"],
        },
        {
            "state": "static_awq_w4_16",
            "base_blocks": static_awq["num_gpu_blocks"],
            "extension_blocks": 0,
            "total_physical_blocks": static_awq["num_gpu_blocks"],
            "scheduler_visible_blocks": static_awq["num_gpu_blocks"],
            "physical_kv_bytes": static_awq["physical_kv_bytes"],
        },
        {
            "state": "runtime_awq_w4_16",
            "base_blocks": cycles["base_blocks"],
            "extension_blocks": cycles["dynamic_awq_blocks"] - cycles["base_blocks"],
            "total_physical_blocks": cycles["dynamic_awq_blocks"],
            "scheduler_visible_blocks": cycles["dynamic_awq_blocks"],
            "physical_kv_bytes": cycles["dynamic_awq_physical_kv_bytes"],
        },
    ]
    memory_rows = [
        {
            "cycle": row["cycle"],
            "restored_allocated_bytes": row["memory_after_restore"]["allocated_bytes"],
            "restored_reserved_bytes": row["memory_after_restore"]["reserved_bytes"],
            "restored_free_bytes": row["memory_after_restore"]["free_bytes"],
            "allocated_spread_from_min_bytes": row["memory_after_restore"]["allocated_bytes"] - min(restored_allocated),
            "reserved_spread_from_min_bytes": row["memory_after_restore"]["reserved_bytes"] - min(restored_reserved),
            "within_64mib_tolerance": (
                row["memory_after_restore"]["allocated_bytes"] - min(restored_allocated) <= LEAK_TOLERANCE_BYTES
                and row["memory_after_restore"]["reserved_bytes"] - min(restored_reserved) <= LEAK_TOLERANCE_BYTES
            ),
        }
        for row in cycles["cycle_rows"]
    ]

    write_csv(analysis / "capacity.csv", capacity_rows)
    write_csv(analysis / "transition_costs.csv", transition_rows)
    write_csv(analysis / "memory_cycles.csv", memory_rows)
    write_csv(analysis / "state_steps.csv", step_rows)
    (analysis / "transition_cost_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    preservation = {
        "schema_version": 1,
        "checks": checks,
        "all_pass": all(checks.values()),
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "runtime_fp16_block_gap_from_static": fp16["num_gpu_blocks"] - cycles["base_blocks"],
        "dynamic_awq_block_gain": cycles["dynamic_awq_blocks"] - cycles["base_blocks"],
        "allocated_cycle_spread_bytes": allocated_spread,
        "reserved_cycle_spread_bytes": reserved_spread,
        "allocated_final_drift_bytes": allocated_final_drift,
        "reserved_final_drift_bytes": reserved_final_drift,
        "leak_tolerance_bytes": LEAK_TOLERANCE_BYTES,
    }
    (analysis / "state_preservation_summary.json").write_text(
        json.dumps(preservation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (raw / "transition-traces.jsonl").open("w", encoding="utf-8") as handle:
        for trace in raw_transitions:
            handle.write(json.dumps(trace, sort_keys=True) + "\n")
    print(json.dumps(preservation, indent=2, sort_keys=True))
    if not preservation["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
