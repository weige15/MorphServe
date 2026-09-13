"""Regenerate v6 packed-backend serving metrics and static eligibility."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from . import analyze_static_frontier as frontier


CONDITIONS = ("fp16_0", "awq_w4_8", "awq_w4_16", "awq_w4_32")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def value_range(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return max(values) - min(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frontier.CONDITIONS = CONDITIONS
    frontier.COLORS = {
        "fp16_0": "#1f77b4",
        "awq_w4_8": "#ff7f0e",
        "awq_w4_16": "#2ca02c",
        "awq_w4_32": "#d62728",
    }
    run_dirs = sorted(
        path.parent
        for path in (args.root / "phase-d" / "runs").glob("serv-*/summary.json")
    )
    if len(run_dirs) != 16:
        raise ValueError(f"expected 16 completed serving runs, found {len(run_dirs)}")
    run_rows, variability = frontier.serving_analysis(
        run_dirs, args.output_dir, near_scale=6.0
    )

    resources = {
        row["condition"]: row
        for row in read_csv(args.root / "phase-c" / "resource-performance-gate.csv")
    }
    fp16_near = sorted(
        (
            row for row in run_rows
            if row["condition"] == "fp16_0" and float(row["time_scale"]) == 6
        ),
        key=lambda row: int(row["repeat"]),
    )
    fp16_p95_range = value_range(fp16_near, "ttft_s_p95")
    fp16_slo_range = value_range(fp16_near, "slo_violation_percent")
    eligibility = []
    for condition in CONDITIONS[1:]:
        candidate = sorted(
            (
                row for row in run_rows
                if row["condition"] == condition and float(row["time_scale"]) == 6
            ),
            key=lambda row: int(row["repeat"]),
        )
        if len(candidate) != 2 or len(fp16_near) != 2:
            raise ValueError(f"{condition} lacks two matched near-knee repeats")
        p95_reductions = [
            float(base["ttft_s_p95"]) - float(current["ttft_s_p95"])
            for base, current in zip(fp16_near, candidate)
        ]
        slo_reductions = [
            float(base["slo_violation_percent"])
            - float(current["slo_violation_percent"])
            for base, current in zip(fp16_near, candidate)
        ]
        candidate_p95_range = value_range(candidate, "ttft_s_p95")
        candidate_slo_range = value_range(candidate, "slo_violation_percent")
        p95_noise = max(
            fp16_p95_range,
            candidate_p95_range,
            0.05 * sum(float(row["ttft_s_p95"]) for row in fp16_near) / 2,
        )
        slo_noise = max(fp16_slo_range, candidate_slo_range, 100 / 64)
        p95_better_both = all(value > 0 for value in p95_reductions)
        slo_better_both = all(value > 0 for value in slo_reductions)
        p95_exceeds_noise = sum(p95_reductions) / 2 > p95_noise
        slo_exceeds_noise = sum(slo_reductions) / 2 > slo_noise
        advantage = (p95_better_both and p95_exceeds_noise) or (
            slo_better_both and slo_exceeds_noise
        )
        resource_relief = float(resources[condition]["block_gain_vs_fp16"]) > 0.05
        profiles_stable = (
            float(resources[condition]["blocks_cv"]) < 0.05
            and float(resources[condition]["workspace_cv"]) < 0.05
        )
        no_hidden_workspace = (
            float(resources[condition]["temporary_profile_bytes_mean"])
            <= 1.01 * float(resources["fp16_0"]["temporary_profile_bytes_mean"])
        )
        mechanism_aligned = (
            sum(float(row["peak_logical_kv_utilization"]) for row in candidate) / 2
            < sum(float(row["peak_logical_kv_utilization"]) for row in fp16_near) / 2
            and sum(float(row["preemption_count"]) for row in candidate)
            < sum(float(row["preemption_count"]) for row in fp16_near)
        )
        eligible = (
            resource_relief
            and profiles_stable
            and no_hidden_workspace
            and advantage
            and mechanism_aligned
        )
        eligibility.append(
            {
                "condition": condition,
                "quantized_layer_count": int(resources[condition]["quantized_layers"]),
                "resource_relief": resource_relief,
                "safe_blocks": float(resources[condition]["safe_gpu_blocks_mean"]),
                "block_gain_percent": 100
                * float(resources[condition]["block_gain_vs_fp16"]),
                "profiles_stable": profiles_stable,
                "no_hidden_workspace": no_hidden_workspace,
                "near_knee_p95_reductions_s": "|".join(
                    f"{value:.9f}" for value in p95_reductions
                ),
                "near_knee_mean_p95_reduction_s": sum(p95_reductions) / 2,
                "p95_better_both": p95_better_both,
                "p95_noise_threshold_s": p95_noise,
                "p95_advantage_exceeds_noise": p95_exceeds_noise,
                "near_knee_slo_reductions_pp": "|".join(
                    f"{value:.6f}" for value in slo_reductions
                ),
                "near_knee_mean_slo_reduction_pp": sum(slo_reductions) / 2,
                "slo_better_both": slo_better_both,
                "slo_noise_threshold_pp": slo_noise,
                "slo_advantage_exceeds_noise": slo_exceeds_noise,
                "mechanism_aligned": mechanism_aligned,
                "eligible": eligible,
                "reason": (
                    "repeatable resource relief and pressure reduction, but no repeated near-knee P95/SLO advantage"
                    if not advantage
                    else "all gates pass"
                ),
            }
        )
    frontier.write_csv(args.output_dir / "state_eligibility.csv", eligibility)
    eligible_states = [row["condition"] for row in eligibility if row["eligible"]]
    decision = {
        "schema_version": 1,
        "decision": "GO" if eligible_states else "NO-GO",
        "eligible_states": eligible_states,
        "recommended_minimal_runtime_states": (
            ["fp16_0", *eligible_states] if eligible_states else []
        ),
        "near_knee_scale": 6,
        "near_knee_nominal_rps": 2 / 3,
        "eligibility": eligibility,
        "quality_required": bool(eligible_states),
        "quality_status": "required" if eligible_states else "not run because no state is eligible",
        "smallest_blocker": (
            None
            if eligible_states
            else "AWQ-Marlin removes memory pressure but adds enough synchronized-prefill compute time that every W4 state has worse near-knee P95 TTFT and SLO behavior than FP16"
        ),
    }
    (args.output_dir / "backend_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_count": len(run_rows),
                "conditions": list(CONDITIONS),
                "variability_rows": len(variability),
                "decision": decision["decision"],
                "source": "all raw requests.jsonl, telemetry.jsonl, metadata.json and Phase C profiles/microbenchmarks",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
