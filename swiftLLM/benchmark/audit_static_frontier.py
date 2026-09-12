"""Audit v5 raw artifacts and derived deliverables against the frozen protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .summarize import summarize_run

CONDITIONS = ("fp16_0", "w4_8", "w4_16", "w4_32")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    manifest_path = root / "protocol_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = sha256(manifest_path)
    decision = json.loads((root / "calibration/calibration_decision.json").read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "evidence": evidence})
        if not passed:
            raise AssertionError(f"{name}: {evidence}")

    check(
        "scope_and_namespace",
        manifest["artifact_version"] == "static_frontier_v5"
        and manifest["scope"]["excluded"][:3] == ["dynamic adaptation", "runtime layer swapping", "controller implementation"],
        {"artifact_version": manifest["artifact_version"], "excluded": manifest["scope"]["excluded"][:3]},
    )
    for preserved in manifest["repository"]["preserve_namespaces"]:
        check(f"preserved_namespace:{preserved}", Path(preserved).is_dir(), preserved)

    quality_dirs = sorted((root / "quality/runs").glob("quality-*"))
    check("four_quality_conditions", len(quality_dirs) == 4, [path.name for path in quality_dirs])
    quality_conditions: set[str] = set()
    for run_dir in quality_dirs:
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        requests = read_jsonl(run_dir / "requests.jsonl")
        condition = str(metadata["condition"])
        quality_conditions.add(condition)
        check(f"quality_raw_files:{condition}", all((run_dir / name).is_file() for name in ("metadata.json", "requests.jsonl", "telemetry.jsonl", "summary.json")), str(run_dir))
        check(f"quality_protocol:{condition}", metadata["protocol_manifest_sha256"] == manifest_hash, metadata["protocol_manifest_sha256"])
        check(f"quality_common_workload:{condition}", metadata["workload_sha256"] == manifest["frozen_source"]["quality_workload_sha256"], metadata["workload_sha256"])
        check(f"quality_sequential_106:{condition}", metadata["arrival_mode"] == "sequential" and len(requests) == 106, {"mode": metadata["arrival_mode"], "count": len(requests)})
        check(f"quality_complete_512:{condition}", all(row.get("status") == "completed" and row.get("prompt_token_count") == 1024 and row.get("output_token_count") == 512 for row in requests), "all requests complete at 1024/512")
        check(f"quality_warmup:{condition}", metadata["warmup_policy"]["warmup_requests"] == 1 and not metadata["warmup_policy"]["included_in_raw_requests"], metadata["warmup_policy"])
    check("quality_condition_set", quality_conditions == set(CONDITIONS), sorted(quality_conditions))

    expected_matrix = {
        (condition, float(scale), repeat)
        for condition in CONDITIONS
        for scale, repeats in ((8, (0,)), (6, (0, 1)), (4, (0,)))
        for repeat in repeats
    }
    observed_matrix: set[tuple[str, float, int]] = set()
    serving_dirs = sorted((root / "serving/runs").glob("serv-*"))
    for run_dir in serving_dirs:
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        requests = read_jsonl(run_dir / "requests.jsonl")
        telemetry = read_jsonl(run_dir / "telemetry.jsonl")
        condition = str(metadata["condition"])
        scale = float(metadata["workload_time_scale"])
        repeat = int(str(metadata["run_id"]).rsplit("-rep", 1)[1])
        observed_matrix.add((condition, scale, repeat))
        expected_workload_hash = decision["latency_workloads"][format(scale, "g")]["sha256"]
        check(f"serving_raw_files:{metadata['run_id']}", all((run_dir / name).is_file() for name in ("metadata.json", "requests.jsonl", "telemetry.jsonl", "summary.json")), str(run_dir))
        check(f"serving_log:{metadata['run_id']}", (root / f"logs/{metadata['run_id']}.log").is_file(), str(root / f"logs/{metadata['run_id']}.log"))
        check(f"serving_protocol:{metadata['run_id']}", metadata["protocol_manifest_sha256"] == manifest_hash, metadata["protocol_manifest_sha256"])
        check(f"serving_workload:{metadata['run_id']}", metadata["workload_sha256"] == expected_workload_hash, metadata["workload_sha256"])
        check(f"serving_complete_64:{metadata['run_id']}", len(requests) == 64 and all(row.get("status") == "completed" and row.get("prompt_token_count") == 1024 and row.get("output_token_count") == 512 for row in requests), {"count": len(requests), "completed": sum(row.get("status") == "completed" for row in requests)})
        required_telemetry = {"waiting_q_depth", "running_q_count", "swapped_q_count", "logical_kv_utilization", "num_gpu_blocks", "swap_in_count", "swap_out_count", "preemption_count", "gpu_memory_used_bytes", "gpu_memory_free_bytes"}
        check(f"serving_telemetry:{metadata['run_id']}", bool(telemetry) and required_telemetry.issubset(telemetry[0]), {"samples": len(telemetry), "fields": sorted(required_telemetry)})
        saved_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        regenerated = summarize_run(run_dir)
        check(
            f"summary_regeneration:{metadata['run_id']}",
            all(saved_summary.get(key) == value for key, value in regenerated.items()),
            "all raw-derived summary fields exactly match; runner-added condition labels ignored",
        )
    check("serving_matrix", observed_matrix == expected_matrix, {"expected": sorted(expected_matrix), "observed": sorted(observed_matrix)})

    near_orderings: list[list[str]] = []
    serving_summary = json.loads((root / "analysis/serving_summary.json").read_text(encoding="utf-8"))
    for repeat in (0, 1):
        rows = [row for row in serving_summary["runs"] if float(row["time_scale"]) == 6 and int(row["repeat"]) == repeat]
        near_orderings.append([str(row["condition"]) for row in sorted(rows, key=lambda row: float(row["ttft_s_p95"]))])
    check("near_knee_ordering_repeatability", near_orderings[0] == near_orderings[1], near_orderings)

    profiles = sorted((root / "mechanism").glob("profile-*.json"))
    micros = sorted((root / "mechanism").glob("micro-*.json"))
    check("mechanism_profile_matrix", len(profiles) == 8 and {int(json.loads(path.read_text())["quantized_layer_count"]) for path in profiles} == {0, 8, 16, 32}, [path.name for path in profiles])
    check("mechanism_microbench_matrix", len(micros) == 4 and all(int(json.loads(path.read_text())["repeats"]) == 7 for path in micros), [path.name for path in micros])

    required_outputs = [
        "quality_table.csv",
        "quality_paired_differences.csv",
        "quality_output_agreement.csv",
        "serving_runs.csv",
        "serving_variability.csv",
        "near_knee_paired_runs.csv",
        "resource_mechanism_table.csv",
        "state_eligibility.csv",
        "dynamic_decision.json",
        "latency_slo_vs_load.png",
        "queue_kv_vs_load.png",
        "quality_characterization.png",
        "quality_vs_near_knee_latency.png",
        "aggregate.json",
    ]
    missing = [name for name in required_outputs if not (root / "analysis" / name).is_file() or (root / "analysis" / name).stat().st_size == 0]
    check("derived_deliverables", not missing, {"required": required_outputs, "missing": missing})
    quality_table = read_csv(root / "analysis/quality_table.csv")
    paired_quality = read_csv(root / "analysis/quality_paired_differences.csv")
    agreement = read_csv(root / "analysis/quality_output_agreement.csv")
    check("quality_table_coverage", len(quality_table) == 4 and all(int(row["request_count"]) == 106 and row["paired_delta_ci95_low"] and row["paired_delta_ci95_high"] for row in quality_table), {"rows": len(quality_table), "request_counts": [row["request_count"] for row in quality_table]})
    check("paired_quality_coverage", len(paired_quality) == 106 and all(f"{condition}_minus_fp16_f1" in paired_quality[0] for condition in CONDITIONS[1:]), {"rows": len(paired_quality)})
    check("secondary_distortion_coverage", len(agreement) == 3 and all(row["token_agreement_ci95_low"] and row["token_agreement_ci95_high"] for row in agreement), {"rows": len(agreement)})
    serving_table = read_csv(root / "analysis/serving_runs.csv")
    per_run_fields = {"ttft_s_p50", "ttft_s_p95", "ttft_s_p99", "slo_violation_percent", "queueing_delay_s_p50", "queueing_delay_s_p95", "queueing_delay_s_p99", "prefill_to_first_output_s_p95", "decode_tpot_s_p95", "completed_throughput_rps", "waiting_q_depth_p95", "running_q_count_p95", "swapped_q_count_p95", "peak_logical_kv_utilization", "num_gpu_blocks", "swap_out_count", "preemption_count", "peak_hbm_used_gib"}
    check("per_run_metric_coverage", len(serving_table) == 16 and per_run_fields.issubset(serving_table[0]), {"rows": len(serving_table), "fields": sorted(per_run_fields)})
    resource_table = read_csv(root / "analysis/resource_mechanism_table.csv")
    check("resource_table_coverage", len(resource_table) == 4 and all(row["persistent_allocation_gib_mean"] and row["peak_temporary_prefill_workspace_gib_mean"] and row["safe_gpu_blocks_mean"] and row["prefill_1024_median_ms"] and row["decode_one_token_median_ms"] and row["serving_saturation_observation"] for row in resource_table), {"rows": len(resource_table)})
    eligibility_table = read_csv(root / "analysis/state_eligibility.csv")
    check("eligibility_table_coverage", len(eligibility_table) == 4 and all(row["classification"] in {"ELIGIBLE", "INELIGIBLE"} for row in eligibility_table), [(row["condition"], row["classification"]) for row in eligibility_table])
    final_decision = json.loads((root / "analysis/dynamic_decision.json").read_text(encoding="utf-8"))
    check("explicit_dynamic_decision", final_decision["decision"] in {"GO", "NO-GO"}, final_decision["decision"])
    commands = json.loads((root / "execution_commands.json").read_text(encoding="utf-8"))
    check("regeneration_commands", all(key in commands for key in ("prepare_workloads", "condition_runner_template", "aggregate", "audit", "validation")), sorted(commands))
    repo_root = root.parents[1]
    report_path = repo_root / "docs/static-frontier-v5/final-report.md"
    audit_path = repo_root / "docs/static-frontier-v5/completion-audit.md"
    report_text = report_path.read_text(encoding="utf-8") if report_path.is_file() else ""
    required_sections = ("CONFIRMED FINDINGS", "SUPPORTED BUT UNCERTAIN FINDINGS", "BLOCKED QUESTIONS", "REMAINING UNCERTAINTY", "DYNAMIC ADAPTATION DECISION")
    check("final_report_sections", all(section in report_text for section in required_sections), {"path": str(report_path), "sections": required_sections})
    check("human_completion_audit", audit_path.is_file() and "Prompt-to-artifact checklist" in audit_path.read_text(encoding="utf-8"), str(audit_path))

    result = {
        "schema_version": 1,
        "status": "PASS",
        "manifest_sha256": manifest_hash,
        "check_count": len(checks),
        "checks": checks,
        "final_dynamic_decision": final_decision["decision"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "check_count": len(checks), "decision": final_decision["decision"]}, indent=2))


if __name__ == "__main__":
    main()
