"""Independent raw/artifact audit for release-side runtime v10."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from .analyze_crossover import counter_value_at, integrate, telemetry_segments
from .run_release_plan import REQUIRED_FILES


ENTRY = "FP16_TO_AWQ_MARLIN_W4_16"
RESTORE = "AWQ_MARLIN_W4_16_TO_FP16"
CONDITIONS = (
    "runtime_static_fp16",
    "runtime_static_awq_w4_16",
    "closed_loop_dynamic",
)
REQUIRED_ANALYSIS = (
    "serving_runs.csv",
    "phase_metrics.csv",
    "matched_phase_comparison.csv",
    "transition_timeline.csv",
    "post_restore_requests.csv",
    "state_timeline.csv",
    "multi_cycle_state_timeline.json",
    "gpu_environment_summary.csv",
    "quality_descriptive.csv",
    "run_validation.json",
    "decision.json",
)
REQUIRED_REPORT_SECTIONS = (
    "CONFIRMED FINDINGS",
    "SUPPORTED BUT UNCERTAIN FINDINGS",
    "BLOCKED QUESTIONS",
    "REMAINING UNCERTAINTY",
    "RELEASE-SIDE SYSTEMS DECISION",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(child.read_bytes()).digest())
    return digest.hexdigest()


def release_formula_first(
    rows: list[dict[str, Any]],
    duration: float,
    awq_stable_s: float,
    before_s: float,
) -> float | None:
    segments = telemetry_segments(rows, duration)
    for row in rows:
        now = float(row["elapsed_s"])
        if now < awq_stable_s + 15.0 or now >= before_s:
            continue
        recent_start = now - 5.0
        prior_start = now - 15.0
        waiting_mean = integrate(
            segments,
            lambda value: float(value["waiting_q_depth"]),
            start=recent_start,
            end=now,
        ) / 5.0
        recent_utilization = integrate(
            segments,
            lambda value: float(value["scheduler_used_kv_blocks"]) / 1768,
            start=recent_start,
            end=now,
        ) / 5.0
        prior_utilization = integrate(
            segments,
            lambda value: float(value["scheduler_used_kv_blocks"]) / 1768,
            start=prior_start,
            end=recent_start,
        ) / 10.0
        recent_preemptions = int(row["preemption_count"]) - counter_value_at(
            rows, "preemption_count", recent_start
        )
        if (
            row.get("runtime_precision_state") == "AWQ_MARLIN_W4_16"
            and int(row["waiting_q_depth"]) == 0
            and waiting_mean <= 0.5
            and recent_preemptions == 0
            and float(row["scheduler_used_kv_blocks"]) / 1768 <= 1.25
            and prior_utilization > 0
            and recent_utilization / prior_utilization <= 0.80
        ):
            return now
    return None


def request_integrity(
    requests: list[dict[str, Any]], batches: list[dict[str, Any]], expected: int
) -> bool:
    occurrences = Counter(
        request_id
        for batch in batches
        for request_id in batch.get("benchmark_request_ids", [])
    )
    prefill_occurrences = Counter(
        request_id
        for batch in batches
        if batch.get("batch_kind") == "prefill"
        for request_id in batch.get("benchmark_request_ids", [])
    )
    return bool(
        len(requests) == expected
        and all(
            row.get("status") == "completed"
            and int(row.get("prompt_token_count", -1)) == 1024
            and int(row.get("output_token_count", -1)) == 512
            and len(row.get("output_token_ids", [])) == 512
            and len(row.get("step_precision_states", [])) == 512
            and len(row.get("step_input_positions", [])) == 512
            and row.get("step_input_positions") == list(range(1023, 1535))
            and len(set(row.get("engine_request_ids", []))) == 1
            and occurrences[row["benchmark_request_id"]] == 512
            and prefill_occurrences[row["benchmark_request_id"]] == 1
            for row in requests
        )
    )


def lifecycle_integrity(event: dict[str, Any]) -> bool:
    trace = event.get("engine_trace") or {}
    lifecycle = trace.get("restore_lifecycle") or {}
    ordered = [
        event.get("release_intent_condition_first_true_ns"),
        event.get("requested_ns"),
        lifecycle.get("engine_restore_request_received_ns"),
        lifecycle.get("admission_pause_ns"),
        lifecycle.get("drain_start_ns"),
        lifecycle.get("first_physical_shrink_legal_ns"),
        trace.get("started_ns"),
        trace.get("ended_ns"),
        lifecycle.get("capacity_published_ns"),
        lifecycle.get("admission_resume_ns"),
    ]
    context_safe = lifecycle.get("context_at_first_physical_shrink_legal") or {}
    before = ((trace.get("kv_resize") or {}).get("integrity_before") or {}).get(
        "logical_digest"
    )
    after = ((trace.get("kv_resize") or {}).get("integrity_after") or {}).get(
        "logical_digest"
    )
    active = int(trace.get("active_request_count", 0))
    digest_ok = (
        before is not None and before == after
        if active > 0
        else (before is None and after is None) or before == after
    )
    return bool(
        all(value is not None for value in ordered)
        and all(int(left) <= int(right) for left, right in zip(ordered, ordered[1:]))
        and int(context_safe.get("allocated_gpu_kv_blocks", 1760)) <= 1759
        and trace.get("status") == "success"
        and trace.get("precision_after") == "FP16"
        and int(trace.get("physical_blocks_after", -1)) == 1759
        and digest_ok
    )


def git_file_at_commit(repo: Path, commit: str, relative: str) -> bytes | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{commit}:{relative}"],
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return None


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
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    protocol = repo / manifest["repository"]["protocol_path"]
    report = repo / "docs" / "release-side-runtime-v10" / "final-report.md"
    completion = repo / "docs" / "release-side-runtime-v10" / "completion-audit.md"
    check("frozen_protocol_hash", sha256_file(protocol) == manifest["repository"]["protocol_sha256"])
    check("frozen_plan_hash", sha256_file(root / "run-plan.json") == manifest["run_plan"]["sha256"])
    check(
        "frozen_workload_metadata_hash",
        sha256_file(root / "input" / "workload_metadata.json")
        == manifest["workloads"]["metadata_sha256"],
    )
    check(
        "all_workload_hashes",
        all(
            sha256_file(repo / record["path"]) == record["sha256"]
            for record in manifest["workloads"]["files"].values()
        ),
    )
    source_hashes = {
        relative: sha256_file(repo / relative)
        for relative in manifest["source_sha256_at_preregistration"]
    }
    check(
        "preregistered_sources_unchanged",
        source_hashes == manifest["source_sha256_at_preregistration"],
        source_hashes,
    )
    historical = {
        relative: tree_sha256(repo / relative)
        for relative in manifest["preserved_namespace_tree_sha256"]
    }
    check(
        "v2_through_v9_artifacts_unchanged",
        historical == manifest["preserved_namespace_tree_sha256"],
        historical,
    )
    check(
        "analysis_outputs_exist",
        all(
            (root / "analysis" / name).is_file()
            and (root / "analysis" / name).stat().st_size > 0
            for name in REQUIRED_ANALYSIS
        ),
    )
    check("final_docs_exist", report.is_file() and completion.is_file())
    if report.is_file():
        text = report.read_text(encoding="utf-8")
        check(
            "required_final_report_sections",
            all(section in text for section in REQUIRED_REPORT_SECTIONS),
        )

    expected_cells = {
        (workload, condition, repeat)
        for workload in ("heldout_one_cycle", "heldout_two_cycle")
        for condition in CONDITIONS
        for repeat in (0, 1)
    } | {
        ("low_only", condition, repeat)
        for condition in ("runtime_static_fp16", "closed_loop_dynamic")
        for repeat in (0, 1)
    }
    check(
        "plan_exact_16_cells",
        len(plan) == 16
        and {
            (row["workload_class"], row["condition"], int(row["repeat"]))
            for row in plan
        }
        == expected_cells,
    )
    check(
        "one_selected_fresh_process_per_cell",
        set(selected) == {row["run_id"] for row in plan}
        and len(set(selected.values())) == 16,
        selected,
    )

    raw_results: dict[str, Any] = {}
    commits: set[str] = set()
    for plan_row in plan:
        actual = selected.get(plan_row["run_id"])
        directory = root / "runs" / str(actual)
        present = bool(actual) and all((directory / name).is_file() for name in REQUIRED_FILES)
        result: dict[str, Any] = {"required_files": present}
        if present:
            metadata = read_json(directory / "metadata.json")
            requests = read_jsonl(directory / "requests.jsonl")
            telemetry = read_jsonl(directory / "telemetry.jsonl")
            controller = read_jsonl(directory / "controller.jsonl")
            batches = read_jsonl(directory / "batches.jsonl")
            transitions = read_jsonl(directory / "transitions.jsonl")
            gpu = read_jsonl(directory / "gpu_environment.jsonl")
            expected = int(plan_row["request_count"])
            commit = str(metadata.get("current_morphserve_git_commit"))
            commits.add(commit)
            directions = [event.get("direction") for event in transitions]
            expected_directions = (
                []
                if plan_row["condition"] != "closed_loop_dynamic"
                or plan_row["workload_class"] == "low_only"
                else [ENTRY, RESTORE]
                if plan_row["workload_class"] == "heldout_one_cycle"
                else [ENTRY, RESTORE, ENTRY, RESTORE]
            )
            release_replay: list[dict[str, Any]] = []
            if plan_row["condition"] == "closed_loop_dynamic":
                entries = [event for event in transitions if event.get("direction") == ENTRY]
                restores = [event for event in transitions if event.get("direction") == RESTORE]
                for index, entry in enumerate(entries):
                    awq_completed = (
                        int(entry["api_task_ended_ns"])
                        - int(metadata["measurement_start_time_ns"])
                    ) / 1e9
                    awq_stable = next(
                        float(row["elapsed_s"])
                        for row in controller
                        if float(row["elapsed_s"]) >= awq_completed
                        and row.get("runtime_precision_state") == "AWQ_MARLIN_W4_16"
                    )
                    restore = restores[index] if index < len(restores) else None
                    stop = (
                        float(restore["requested_elapsed_s"])
                        if restore is not None
                        else float(metadata["measurement_duration_s"]) + 1
                    )
                    replayed = release_formula_first(
                        controller,
                        float(metadata["measurement_duration_s"]),
                        awq_stable,
                        stop + 1e-6,
                    )
                    observed = (
                        None
                        if restore is None
                        else float(restore["release_intent_condition_first_true_elapsed_s"])
                    )
                    release_replay.append(
                        {
                            "replayed_s": replayed,
                            "observed_s": observed,
                            "match": replayed is not None
                            and observed is not None
                            and abs(replayed - observed) <= 1e-6,
                        }
                    )
            result.update(
                {
                    "identity": metadata.get("run_id") == actual
                    and metadata.get("planned_run_id") == plan_row["run_id"]
                    and metadata.get("experiment") == "release_side_runtime_v10"
                    and metadata.get("workload_sha256") == plan_row["workload_sha256"]
                    and metadata.get("protocol_manifest_sha256")
                    == sha256_file(root / "protocol_manifest.json"),
                    "clean_recorded_commit": not metadata.get("current_morphserve_git_dirty"),
                    "recorded_sources_match_manifest": metadata.get(
                        "runtime_source_files_sha256"
                    )
                    == manifest["source_sha256_at_preregistration"],
                    "request_integrity": request_integrity(requests, batches, expected),
                    "arrival_integrity": all(
                        abs(float(row["arrival_jitter_s"])) <= 0.25 for row in requests
                    ),
                    "telemetry_capacity_integrity": bool(telemetry)
                    and all(
                        int(row["base_physical_kv_blocks"]) == 1759
                        and 0
                        <= int(row["physical_used_kv_blocks"])
                        <= int(row["physical_total_kv_blocks"])
                        and int(row["fp16_policy_reference_blocks"]) == 1768
                        for row in telemetry
                    ),
                    "gpu_environment_available": bool(gpu)
                    and any(
                        row.get("gpu_environment_source") == "pynvml"
                        and row.get("gpu_sm_clock_mhz") is not None
                        and row.get("gpu_temperature_c") is not None
                        and row.get("gpu_power_w") is not None
                        for row in gpu
                    ),
                    "exact_transition_sequence": directions == expected_directions,
                    "all_restore_lifecycles_valid": all(
                        lifecycle_integrity(event)
                        for event in transitions
                        if event.get("direction") == RESTORE
                    ),
                    "independent_release_replay": release_replay,
                    "independent_release_replay_matches": (
                        not expected_directions
                        or len(release_replay) == expected_directions.count(RESTORE)
                        and all(row["match"] for row in release_replay)
                    ),
                    "no_errors": not metadata.get("run_exception")
                    and all(not row.get("error") for row in requests)
                    and all(event.get("result") == "success" for event in transitions),
                }
            )
        result["all_pass"] = all(
            value
            for key, value in result.items()
            if key not in {"independent_release_replay"}
        )
        raw_results[str(plan_row["run_id"])] = result
    check(
        "all_raw_runs_pass_direct_audit",
        len(raw_results) == 16 and all(row["all_pass"] for row in raw_results.values()),
    )
    check("all_runs_same_clean_preregistered_commit", len(commits) == 1, sorted(commits))
    if len(commits) == 1:
        commit = next(iter(commits))
        protocol_at_run = git_file_at_commit(
            repo, commit, manifest["repository"]["protocol_path"]
        )
        manifest_at_run = git_file_at_commit(
            repo, commit, str(root.relative_to(repo) / "protocol_manifest.json")
        )
        check(
            "protocol_and_manifest_existed_at_run_commit",
            protocol_at_run is not None
            and hashlib.sha256(protocol_at_run).hexdigest()
            == manifest["repository"]["protocol_sha256"]
            and manifest_at_run is not None
            and hashlib.sha256(manifest_at_run).hexdigest()
            == sha256_file(root / "protocol_manifest.json"),
            commit,
        )

    decision = read_json(root / "analysis" / "decision.json")
    check(
        "decision_is_explicit_go_or_no_go",
        decision.get("release_side_systems_decision") in {"GO", "NO-GO"},
        decision.get("release_side_systems_decision"),
    )
    audit = {
        "schema_version": 1,
        "status": "PASS" if all(row["passed"] for row in checks) else "FAIL",
        "check_count": len(checks),
        "passed_count": sum(row["passed"] for row in checks),
        "checks": checks,
        "raw_runs": raw_results,
        "scientific_decision": decision.get("release_side_systems_decision"),
        "note": "Artifact-audit PASS does not imply scientific GO.",
    }
    (root / "completion_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    if audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
