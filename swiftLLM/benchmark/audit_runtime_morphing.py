"""Completion audit for the runtime-morphing-v8 evidence namespace."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
PREVIOUS_NAMESPACES = (
    "benchmark-results/static-quantization-quality-latency/",
    "benchmark-results/table5-substrate-v4/",
    "benchmark-results/static-frontier-v5/",
    "benchmark-results/packed-int4-backend-v6/",
    "benchmark-results/fp16-awq-crossover-v7/",
    "docs/static-quantization-quality-latency/",
    "docs/table5-substrate-v4/",
    "docs/static-frontier-v5/",
    "docs/packed-int4-backend-v6/",
    "docs/fp16-awq-crossover-v7/",
)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("benchmark-results/runtime-morphing-v8")
    )
    args = parser.parse_args()
    root = args.root
    raw = root / "raw"
    analysis = root / "analysis"
    docs = Path("docs/runtime-morphing-v8")
    checks = []

    def check(name: str, passed: bool, evidence) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    required = [
        docs / "protocol.md",
        docs / "transition-architecture.md",
        docs / "state-memory-ownership.md",
        docs / "transition-trace-format.md",
        docs / "final-report.md",
        docs / "completion-audit.md",
        raw / "state-fp16.json",
        raw / "state-static-awq.json",
        raw / "state-morph.json",
        raw / "state-roundtrip.json",
        raw / "capacity-cycles.json",
        raw / "transition-traces.jsonl",
        analysis / "capacity.csv",
        analysis / "transition_costs.csv",
        analysis / "transition_cost_summary.json",
        analysis / "memory_cycles.csv",
        analysis / "state_preservation_summary.json",
        root / "execution_commands.json",
        root / "regenerate.sh",
        Path("swiftLLM/benchmark/test_runtime_morphing.py"),
        Path("swiftLLM/benchmark/run_runtime_morphing_validation.py"),
        Path("swiftLLM/benchmark/analyze_runtime_morphing.py"),
        Path("swiftLLM/benchmark/audit_runtime_morphing.py"),
    ]
    check(
        "required_artifacts_exist",
        all(path.is_file() for path in required),
        [str(path) for path in required if not path.is_file()],
    )

    fp16 = load(raw / "state-fp16.json")
    static_awq = load(raw / "state-static-awq.json")
    morph = load(raw / "state-morph.json")
    roundtrip = load(raw / "state-roundtrip.json")
    cycles = load(raw / "capacity-cycles.json")
    preservation = load(analysis / "state_preservation_summary.json")
    costs = load(analysis / "transition_cost_summary.json")
    commands = load(root / "execution_commands.json")
    diagnostic = load(root / "diagnostics/capacity-4286-unsafe-envelope.json")

    payloads = (fp16, static_awq, morph, roundtrip, cycles)
    check(
        "all_final_raw_runs_same_device",
        all(
            payload["gpu"]["name"] == "NVIDIA GeForce RTX 3090"
            and payload["gpu"]["cuda_visible_devices"] == "5"
            for payload in payloads
        ),
        [(payload["condition"], payload["gpu"]) for payload in payloads],
    )
    check(
        "all_final_raw_runs_share_provenance",
        all(payload["provenance"] == fp16["provenance"] for payload in payloads),
        fp16["provenance"]["runtime_source_diff_sha256"],
    )
    current_hashes = {
        relative: sha256(REPO_ROOT / relative)
        for relative in fp16["provenance"]["runtime_source_files_sha256"]
    }
    check(
        "raw_source_hashes_match_current_tree",
        current_hashes == fp16["provenance"]["runtime_source_files_sha256"],
        current_hashes,
    )
    check(
        "raw_git_base_is_current_head",
        fp16["provenance"]["git_commit"]
        == subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        fp16["provenance"]["git_commit"],
    )

    changed = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "diff", "--name-only", "HEAD"], text=True
    ).splitlines()
    touched_previous = [
        path for path in changed if any(path.startswith(prefix) for prefix in PREVIOUS_NAMESPACES)
    ]
    check("v2_v4_v5_v6_v7_namespaces_unchanged", not touched_previous, touched_previous)

    engine_source = Path("swiftLLM/swiftllm/server/engine.py").read_text(encoding="utf-8")
    scheduler_source = Path("swiftLLM/swiftllm/server/scheduler.py").read_text(encoding="utf-8")
    runtime_source = engine_source + scheduler_source
    forbidden_controller_terms = (
        "sustained_compound_pressure",
        "capacity_margin_queue_integral",
        "release_hysteresis",
        "crossover_threshold",
    )
    check(
        "no_pressure_controller_implemented",
        not any(term in runtime_source for term in forbidden_controller_terms),
        forbidden_controller_terms,
    )
    check(
        "manual_api_and_forward_boundary_present",
        all(
            marker in engine_source
            for marker in (
                "async def morph_to_awq_w4_16",
                "async def restore_to_fp16",
                "await self._service_pending_transition()",
                "self._transition_lock",
            )
        ),
        "Engine manual API, main-loop service, and serialization lock",
    )
    check(
        "restore_admission_hold_is_bounded",
        "self.scheduler.admissions_paused = True" in engine_source
        and "allocated > self.model.base_num_blocks" in engine_source
        and "elif not self.admissions_paused" in scheduler_source,
        "explicit drain guard; FCFS queues retained",
    )

    v6_manifest = load(
        Path("benchmark-results/packed-int4-backend-v6/phase-b/awq-checkpoint-manifest.json")
    )
    awq_path = Path(v6_manifest["output_checkpoint"])
    checkpoint_results = {}
    for item in v6_manifest["files"]:
        path = awq_path / item["name"]
        checkpoint_results[item["name"]] = {
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256(path) if path.is_file() else None,
            "expected_bytes": item["bytes"],
            "expected_sha256": item["sha256"],
        }
    check(
        "validated_awq_checkpoint_hashes_match",
        all(
            row["exists"]
            and row["bytes"] == row["expected_bytes"]
            and row["sha256"] == row["expected_sha256"]
            for row in checkpoint_results.values()
        ),
        checkpoint_results,
    )
    check(
        "state_pair_is_exactly_fp16_awq_w4_16",
        fp16["condition"] == "fp16"
        and static_awq["condition"] == "static_awq"
        and all(
            trace["direction"]
            in ("FP16_TO_AWQ_MARLIN_W4_16", "AWQ_MARLIN_W4_16_TO_FP16")
            for trace in cycles["transition_traces"]
        ),
        [trace["direction"] for trace in cycles["transition_traces"]],
    )

    check("analysis_invariants_all_pass", preservation["all_pass"], preservation)
    check(
        "static_capacities_reproduced",
        fp16["num_gpu_blocks"] == 1768 and static_awq["num_gpu_blocks"] == 4286,
        {"fp16": fp16["num_gpu_blocks"], "awq": static_awq["num_gpu_blocks"]},
    )
    envelope = cycles["max_shape_capacity_validation"]
    check(
        "dynamic_capacity_is_physical_and_max_shape_safe",
        cycles["base_blocks"] == 1759
        and cycles["dynamic_awq_blocks"] == 4170
        and envelope["derived_safe_total_blocks"] == 4170
        and envelope["driver_used_bytes_at_envelope"] <= envelope["usable_budget_bytes"]
        and envelope["outputs_valid"],
        envelope,
    )
    rejected_envelope = diagnostic["max_shape_capacity_validation"]
    check(
        "unsafe_static_target_is_not_advertised",
        diagnostic["dynamic_awq_blocks"] == 4286
        and rejected_envelope["driver_used_bytes_at_envelope"]
        > rejected_envelope["usable_budget_bytes"]
        and cycles["dynamic_awq_blocks"] < diagnostic["dynamic_awq_blocks"],
        {
            "rejected": diagnostic["dynamic_awq_blocks"],
            "accepted": cycles["dynamic_awq_blocks"],
            "over_budget_bytes": rejected_envelope["driver_used_bytes_at_envelope"]
            - rejected_envelope["usable_budget_bytes"],
        },
    )
    check(
        "capacity_growth_is_meaningful",
        cycles["dynamic_awq_blocks"] - cycles["base_blocks"] == 2411,
        {"base": cycles["base_blocks"], "dynamic": cycles["dynamic_awq_blocks"]},
    )

    check(
        "runtime_awq_matches_static_bytes",
        cycles["awq_runtime_equivalence"]["byte_exact"]
        and cycles["awq_runtime_equivalence"]["observed_sha256"]
        == static_awq["selected_layer_digest"]["sha256"],
        cycles["awq_runtime_equivalence"],
    )
    check(
        "restored_fp16_matches_static_bytes",
        cycles["fp16_runtime_equivalence"]["byte_exact"]
        and cycles["fp16_runtime_equivalence"]["observed_sha256"]
        == fp16["selected_layer_digest"]["sha256"],
        cycles["fp16_runtime_equivalence"],
    )

    transitions = (
        morph["transition_boundaries"]
        + roundtrip["transition_boundaries"]
        + cycles["transition_traces"]
    )
    check(
        "nine_successful_raw_transitions",
        len(transitions) == 9 and all(row["status"] == "success" for row in transitions),
        len(transitions),
    )
    check(
        "every_transition_covers_layers_0_15",
        all(
            [layer["layer_id"] for layer in row["weight_transition"]["layers"]]
            == list(range(16))
            for row in transitions
        ),
        "9 transitions x 16 ordered layers",
    )
    check(
        "transition_peak_and_coexistence_instrumented",
        all(
            row["transition_peak_allocated_bytes"] > 0
            and all(
                layer["memory_with_both_representations"]["allocated_bytes"] > 0
                for layer in row["weight_transition"]["layers"]
            )
            for row in transitions
        ),
        max(row["transition_peak_allocated_bytes"] for row in transitions),
    )
    check(
        "transition_fields_complete",
        all(
            all(
                key in row
                for key in (
                    "started_ns",
                    "ended_ns",
                    "elapsed_ns",
                    "boundary_cuda_sync_ns",
                    "memory_before",
                    "memory_after",
                    "active_request_count",
                    "used_kv_blocks",
                    "precision_before",
                    "precision_after",
                    "kv_resize",
                )
            )
            for row in transitions
        ),
        "timestamps/state/HBM/requests/KV/copy fields",
    )
    trace_lines = [
        json.loads(line)
        for line in (raw / "transition-traces.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    check("raw_transition_jsonl_complete", len(trace_lines) == len(transitions), len(trace_lines))

    for condition, payload, directions in (
        ("morph", morph, 1),
        ("roundtrip", roundtrip, 2),
    ):
        prefill_counts = {f"runtime-r{i}": 0 for i in range(2)}
        for event in payload["batch_events"]:
            if event["num_prefill_sequences"]:
                for request_id in event["benchmark_request_ids"]:
                    prefill_counts[request_id] += 1
        request_ids = {
            sequence: {row["request_id"] for row in payload["steps"] if row["sequence"] == sequence}
            for sequence in range(2)
        }
        step_ok = all(
            [row["step_index"] for row in sorted(
                (item for item in payload["steps"] if item["sequence"] == sequence),
                key=lambda item: item["step_index"],
            )]
            == list(range(24))
            for sequence in range(2)
        )
        check(
            f"{condition}_two_request_state_preservation",
            len(payload["transition_boundaries"]) == directions
            and all(count == 1 for count in prefill_counts.values())
            and request_ids == {0: {0}, 1: {1}}
            and step_ok,
            {"prefills": prefill_counts, "request_ids": {k: sorted(v) for k, v in request_ids.items()}},
        )

    live = cycles["full_model_extension_execution"]
    check(
        "live_extension_history_restored_and_decoded",
        live["block_table_ids_before_restore"] == [0, 1759]
        and live["block_table_ids_after_restore"] == [0, 1]
        and live["logical_kv_before_restore"]["logical_digest"]
        == live["logical_kv_after_restore"]["logical_digest"]
        and live["request_id_before_after"] == [0, 0]
        and live["post_restore_token_valid"],
        live,
    )
    check(
        "three_cycle_memory_gate",
        preservation["allocated_final_drift_bytes"] <= preservation["leak_tolerance_bytes"]
        and preservation["reserved_final_drift_bytes"] <= preservation["leak_tolerance_bytes"]
        and preservation["checks"]["repeated_cycles_have_no_monotonic_memory_leak"],
        {
            "allocated_final_drift_bytes": preservation["allocated_final_drift_bytes"],
            "reserved_final_drift_bytes": preservation["reserved_final_drift_bytes"],
            "reserved_cycle_spread_bytes": preservation["reserved_cycle_spread_bytes"],
        },
    )
    check(
        "transition_cost_distribution_complete",
        costs["FP16_TO_AWQ_MARLIN_W4_16"]["count"] == 3
        and costs["AWQ_MARLIN_W4_16_TO_FP16"]["count"] == 3
        and costs["v7_amortization_estimate"]["one_way_transition_cost_s"] > 0,
        costs,
    )

    report = (docs / "final-report.md").read_text(encoding="utf-8")
    required_sections = (
        "## CONFIRMED FINDINGS",
        "## SUPPORTED BUT UNCERTAIN FINDINGS",
        "## BLOCKED QUESTIONS",
        "## REMAINING UNCERTAINTY",
        "## RUNTIME MORPHING DECISION: **GO**",
    )
    check(
        "final_report_ending_contract",
        all(section in report for section in required_sections)
        and report.rstrip().endswith("## RUNTIME MORPHING DECISION: **GO**"),
        required_sections,
    )
    check(
        "commands_cover_raw_analysis_tests_audit",
        len(commands["raw_generation"]) == 5
        and all("CUDA_VISIBLE_DEVICES=5" in command for command in commands["raw_generation"])
        and "MORPHSERVE_RUN_CUDA_TESTS=1" in commands["cuda_tests"]
        and "analyze_runtime_morphing" in commands["analysis"]
        and "audit_runtime_morphing" in commands["audit"],
        commands,
    )

    result = {
        "schema_version": 1,
        "status": "PASS" if all(row["passed"] for row in checks) else "FAIL",
        "check_count": len(checks),
        "passed_count": sum(row["passed"] for row in checks),
        "failed_count": sum(not row["passed"] for row in checks),
        "checks": checks,
        "decision": "GO" if all(row["passed"] for row in checks) else "NO-GO",
    }
    (root / "completion_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": result["status"],
        "checks": result["check_count"],
        "passed": result["passed_count"],
        "failed": result["failed_count"],
        "decision": result["decision"],
    }, indent=2))
    if result["status"] != "PASS":
        for row in checks:
            if not row["passed"]:
                print(f"FAIL: {row['name']}: {row['evidence']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
