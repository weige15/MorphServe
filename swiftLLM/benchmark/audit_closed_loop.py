"""Independent completion audit for the final v9 closed-loop experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .run_closed_loop_plan import REQUIRED_FILES


REQUIRED_ANALYSIS = (
    "serving_runs.csv",
    "run_validation.json",
    "paired_performance.csv",
    "quality_repeat_values.csv",
    "quality_request_pairs.csv",
    "quality_unique_question_aggregate.csv",
    "quality_fidelity_summary.csv",
    "controller_trigger_audit.csv",
    "transition_cost_contribution.csv",
    "phased_reversibility.json",
    "headline_table5.csv",
    "decision.json",
    "headline_latency_queue_kv_state_timeline.png",
    "phased_controller_timeline.png",
)
PRESERVED_NAMESPACES = (
    "benchmark-results/static-quantization-quality-latency",
    "benchmark-results/table5-substrate-v4",
    "benchmark-results/static-frontier-v5",
    "benchmark-results/packed-int4-backend-v6",
    "benchmark-results/fp16-awq-crossover-v7",
    "benchmark-results/runtime-morphing-v8",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(child.read_bytes()).digest())
    return digest.hexdigest()


def direct_transition_ok(event: dict[str, Any]) -> bool:
    trace = event.get("engine_trace") or {}
    resize = trace.get("kv_resize") or {}
    direction = event.get("direction")
    expected_precision = "AWQ_MARLIN_W4_16" if direction == "FP16_TO_AWQ_MARLIN_W4_16" else "FP16" if direction == "AWQ_MARLIN_W4_16_TO_FP16" else None
    expected_blocks = 4170 if expected_precision == "AWQ_MARLIN_W4_16" else 1759
    context = trace.get("engine_context_after") or {}
    before = (resize.get("integrity_before") or {}).get("logical_digest")
    after = (resize.get("integrity_after") or {}).get("logical_digest")
    active = int(trace.get("active_request_count", 0))
    digest_ok = (before is not None and before == after) if active > 0 else ((before is None and after is None) or before == after)
    return bool(
        event.get("result") == "success"
        and not event.get("error")
        and trace.get("status") == "success"
        and trace.get("precision_after") == expected_precision
        and trace.get("physical_blocks_after") == expected_blocks
        and context.get("physical_blocks") == expected_blocks
        and context.get("scheduler_visible_blocks") == expected_blocks
        and resize.get("actual_total_blocks", expected_blocks) == expected_blocks
        and digest_ok
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    repo = Path(__file__).resolve().parents[2]
    manifest = read_json(root / "protocol_manifest.json")
    plan = read_json(root / "run-plan.json")["runs"]
    status = read_json(root / "execution_status.json")
    selected = status.get("selected_runs", {})
    plan = [dict(row, actual_run_id=selected.get(row["run_id"])) for row in plan]
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    required_top = (
        root / "protocol_manifest.json",
        root / "run-plan.json",
        root / "input" / "workload_metadata.json",
        root / "execution_commands.json",
        root / "regenerate.sh",
        repo / "docs" / "closed-loop-runtime-v9" / "protocol.md",
        repo / "docs" / "closed-loop-runtime-v9" / "final-report.md",
        repo / "docs" / "closed-loop-runtime-v9" / "completion-audit.md",
    )
    check("required_top_level_artifacts_exist", all(path.is_file() for path in required_top), [str(path) for path in required_top if not path.is_file()])
    check("all_required_analysis_artifacts_exist", all((root / "analysis" / name).is_file() and (root / "analysis" / name).stat().st_size > 0 for name in REQUIRED_ANALYSIS))
    check("frozen_plan_hash", sha256_file(root / "run-plan.json") == manifest["run_plan"]["sha256"])
    protocol_path = repo / manifest["repository"]["protocol_path"]
    check("frozen_protocol_hash", sha256_file(protocol_path) == manifest["repository"]["protocol_sha256"])
    v9_sources = {
        relative: sha256_file(repo / relative)
        for relative in manifest["v9_source_sha256_at_preregistration"]
    }
    check("frozen_v9_implementation_hashes", v9_sources == manifest["v9_source_sha256_at_preregistration"], v9_sources)
    policy_helpers = {
        relative: sha256_file(repo / relative)
        for relative in manifest["policy_helper_source_sha256"]
    }
    check("frozen_v7_policy_helper_hashes", policy_helpers == manifest["policy_helper_source_sha256"], policy_helpers)
    check("frozen_workload_metadata_hash", sha256_file(root / "input" / "workload_metadata.json") == manifest["workloads"]["metadata_sha256"])
    workload_hashes_ok = all(
        sha256_file(repo / item["path"]) == item["sha256"]
        for item in manifest["workloads"]["files"].values()
    )
    check("all_frozen_workload_hashes", workload_hashes_ok)
    preserved = {
        relative: tree_sha256(repo / relative) for relative in PRESERVED_NAMESPACES
    }
    check("v2_v4_v5_v6_v7_v8_namespaces_unchanged", preserved == manifest["preserved_namespace_tree_sha256"], preserved)
    runtime_sources = {
        relative: sha256_file(repo / relative)
        for relative in manifest["runtime_source_sha256"]
    }
    check("v8_runtime_substrate_sources_unchanged", runtime_sources == manifest["runtime_source_sha256"], runtime_sources)

    expected_cells = {
        (workload, condition, repeat)
        for workload in ("low", "high", "low_high_low")
        for condition in ("runtime_static_fp16", "runtime_static_awq_w4_16", "closed_loop_dynamic")
        for repeat in (0, 1)
    }
    check("plan_is_exact_3x3x2_matrix", len(plan) == 18 and {(row["workload_class"], row["condition"], int(row["repeat"])) for row in plan} == expected_cells)
    check(
        "execution_selects_one_attempt_per_cell",
        set(selected) == {row["run_id"] for row in plan}
        and len(set(selected.values())) == 18
        and all(row.get("actual_run_id") for row in plan),
        selected,
    )
    attempts = status.get("attempts", [])
    check(
        "all_attempts_and_failures_preserved_with_lineage",
        all(
            (root / "runs" / row["actual_run_id"]).is_dir()
            and Path(row["log"]).is_file()
            and row.get("planned_run_id") in selected
            and (row.get("valid_completed_run") or selected[row["planned_run_id"]] != row["actual_run_id"])
            for row in attempts
        ),
        attempts,
    )

    raw_checks: dict[str, Any] = {}
    expected_recorded_sources = {
        **manifest["runtime_source_sha256"],
        **manifest["policy_helper_source_sha256"],
        **manifest["v9_source_sha256_at_preregistration"],
    }
    for plan_row in plan:
        run_dir = root / "runs" / str(plan_row["actual_run_id"])
        present = all((run_dir / name).is_file() for name in REQUIRED_FILES)
        result: dict[str, Any] = {"required_files": present}
        if present:
            metadata = read_json(run_dir / "metadata.json")
            requests = read_jsonl(run_dir / "requests.jsonl")
            telemetry = read_jsonl(run_dir / "telemetry.jsonl")
            batches = read_jsonl(run_dir / "batches.jsonl")
            controller = read_jsonl(run_dir / "controller.jsonl")
            transitions = read_jsonl(run_dir / "transitions.jsonl")
            initialization = read_jsonl(run_dir / "initialization_transitions.jsonl")
            expected = int(plan_row["request_count"])
            result.update(
                {
                    "identity_and_hash": metadata.get("run_id") == plan_row["actual_run_id"]
                    and metadata.get("planned_run_id") == plan_row["run_id"]
                    and metadata.get("condition") == plan_row["condition"]
                    and metadata.get("workload_class") == plan_row["workload_class"]
                    and metadata.get("workload_sha256") == plan_row["workload_sha256"],
                    "same_runtime_envelope": metadata.get("engine_config_actual", {}).get("enable_runtime_morphing") is True
                    and metadata.get("engine_config_actual", {}).get("quantized_layer_count") == 0
                    and metadata.get("engine_config_actual", {}).get("runtime_awq_target_blocks") == 4170,
                    "exact_completion": len(requests) == expected
                    and all(row.get("status") == "completed" and row.get("prompt_token_count") == 1024 and row.get("output_token_count") == 512 for row in requests),
                    "request_state_integrity": all(
                        row.get("input_positions_contiguous")
                        and row.get("engine_request_id_stable")
                        and len(row.get("output_token_ids", [])) == 512
                        and all(
                            isinstance(token, int) and 0 <= token < int(metadata["vocab_size"])
                            for token in row.get("output_token_ids", [])
                        )
                        and len(row.get("step_precision_states", []))
                        == len(row.get("step_input_positions", []))
                        == len(row.get("engine_request_ids", []))
                        == len(row.get("step_received_time_ns", []))
                        == 512
                        and set(row.get("step_precision_states", [])) <= {"FP16", "AWQ_MARLIN_W4_16"}
                        and int(row.get("fp16_output_step_count", -1))
                        + int(row.get("awq_output_step_count", -1)) == 512
                        and all(
                            int(left) <= int(right)
                            for left, right in zip(
                                row.get("step_received_time_ns", []),
                                row.get("step_received_time_ns", [])[1:],
                            )
                        )
                        for row in requests
                    ),
                    "arrival_integrity": all(
                        abs(float(row.get("arrival_jitter_s", float("inf")))) <= 0.25
                        and abs(
                            float(row["actual_arrival_offset_s"])
                            - (int(row["arrival_time_ns"]) - int(metadata["measurement_start_time_ns"])) / 1e9
                        ) <= 1e-9
                        for row in requests
                    )
                    and all(
                        int(left["arrival_time_ns"]) <= int(right["arrival_time_ns"])
                        for left, right in zip(requests, requests[1:])
                    ),
                    "raw_source_provenance_frozen": all(
                        metadata.get("runtime_source_files_sha256", {}).get(relative) == expected_hash
                        for relative, expected_hash in expected_recorded_sources.items()
                    ),
                    "raw_streams_nonempty": bool(telemetry) and bool(batches),
                    "policy_physical_separation": all(
                        row.get("fp16_policy_reference_blocks") == 1768
                        and row.get("base_physical_kv_blocks") == 1759
                        and 0 <= row.get("physical_used_kv_blocks", -1) <= row.get("physical_total_kv_blocks", -2)
                        for row in telemetry
                    ),
                    "controller_presence_only_dynamic": bool(controller) == (plan_row["condition"] == "closed_loop_dynamic"),
                    "static_never_actuates": plan_row["condition"] == "closed_loop_dynamic" or not transitions,
                    "static_awq_preinitialized": (len(initialization) == 1) == (plan_row["condition"] == "runtime_static_awq_w4_16"),
                    "transitions_successful_capacity_and_kv_valid": all(direct_transition_ok(row) for row in transitions),
                    "no_failure_artifact": not (run_dir / "failure.json").exists(),
                }
            )
        result["all_pass"] = present and all(value for key, value in result.items() if key != "required_files") and result["required_files"]
        raw_checks[f"{plan_row['run_id']}->{plan_row['actual_run_id']}"] = result
    check("all_18_raw_runs_directly_valid", all(row["all_pass"] for row in raw_checks.values()), raw_checks)

    validation = read_json(root / "analysis" / "run_validation.json")
    decision = read_json(root / "analysis" / "decision.json")
    phased = read_json(root / "analysis" / "phased_reversibility.json")
    check("analyzer_validates_all_runs", validation.get("all_pass") is True)
    check("low_negative_control_gate_covered", "low_no_false_entry_both" in decision.get("performance_gates", {}) and "low_fp16_like_within_noise_both" in decision.get("performance_gates", {}), decision.get("performance_gates"))
    check("high_repeated_mechanism_gate_covered", all(name in decision.get("performance_gates", {}) for name in (
        "high_enters_awq_both",
        "high_improvement_exceeds_noise_both",
        "high_queue_area_kv_aligned_both",
        "high_preemption_aligned",
        "high_throughput_gate",
    )), decision.get("performance_gates"))
    check("phased_useful_restore_gate_covered", len(phased.get("repeats", [])) == 2 and all("completed_recovery_requests_arriving_after_restore" in row for row in phased.get("repeats", [])), phased)
    check("quality_unique_unit_and_ci_covered", "dynamic_minus_awq_paired_unique_question_ci95_low_pp" in decision.get("quality_gate", {}), decision.get("quality_gate"))
    check("both_final_decisions_present", decision.get("systems_adaptation_decision") in ("GO", "NO-GO") and decision.get("quality_latency_tradeoff_decision") in ("STRONG GO", "SUPPORTED / UNCERTAIN", "NO-GO"), decision)

    quality_unique_lines = sum(1 for line in (root / "analysis" / "quality_unique_question_aggregate.csv").read_text(encoding="utf-8").splitlines() if line.strip()) - 1
    check("quality_uncertainty_uses_64_unique_questions", quality_unique_lines == 64, quality_unique_lines)
    report = (repo / "docs" / "closed-loop-runtime-v9" / "final-report.md").read_text(encoding="utf-8") if (repo / "docs" / "closed-loop-runtime-v9" / "final-report.md").exists() else ""
    required_sections = (
        "## CONFIRMED FINDINGS",
        "## SUPPORTED BUT UNCERTAIN FINDINGS",
        "## BLOCKED QUESTIONS",
        "## REMAINING UNCERTAINTY",
        "## SYSTEMS ADAPTATION DECISION",
        "## QUALITY–LATENCY TRADE-OFF DECISION",
    )
    check("final_report_has_required_ending_sections", all(section in report for section in required_sections))

    result = {
        "schema_version": 1,
        "status": "PASS" if all(row["passed"] for row in checks) else "FAIL",
        "check_count": len(checks),
        "failed_checks": [row["name"] for row in checks if not row["passed"]],
        "checks": checks,
        "systems_adaptation_decision": decision.get("systems_adaptation_decision"),
        "quality_latency_tradeoff_decision": decision.get("quality_latency_tradeoff_decision"),
    }
    (root / "completion_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
