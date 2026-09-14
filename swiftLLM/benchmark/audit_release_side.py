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


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percent / 100
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def catch_up_duration(
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
                return contiguous_start + 3.0 - resume_s
        else:
            contiguous_start = None
    return None


def independent_decision_from_raw(
    root: Path,
    plan: list[dict[str, Any]],
    selected: dict[str, str],
    raw_results: dict[str, Any],
) -> dict[str, Any]:
    workload_metadata = read_json(root / "input" / "workload_metadata.json")
    phase_rows: list[dict[str, Any]] = []
    run_data: dict[tuple[str, str, int], dict[str, Any]] = {}
    for plan_row in plan:
        workload = str(plan_row["workload_class"])
        condition = str(plan_row["condition"])
        repeat = int(plan_row["repeat"])
        directory = root / "runs" / selected[plan_row["run_id"]]
        metadata = read_json(directory / "metadata.json")
        requests = read_jsonl(directory / "requests.jsonl")
        telemetry = read_jsonl(directory / "telemetry.jsonl")
        transitions = read_jsonl(directory / "transitions.jsonl")
        duration = float(metadata["measurement_duration_s"])
        segments = telemetry_segments(telemetry, duration)
        boundaries = workload_metadata["phase_boundaries"][workload]
        run_data[(workload, condition, repeat)] = {
            "metadata": metadata,
            "requests": requests,
            "telemetry": telemetry,
            "transitions": transitions,
            "boundaries": boundaries,
        }
        for index, boundary in enumerate(boundaries):
            phase_index = int(boundary["phase_index"])
            phase_requests = [
                row for row in requests if int(row.get("phase_index", -1)) == phase_index
            ]
            ttft = [
                (int(row["first_stream_token_received_time_ns"]) - int(row["arrival_time_ns"]))
                / 1e9
                for row in phase_requests
            ]
            start = float(boundary["start_offset_s"])
            end = (
                float(boundaries[index + 1]["start_offset_s"])
                if index + 1 < len(boundaries)
                else duration
            )
            phase_transitions = [
                event
                for event in transitions
                if start <= float(event["requested_elapsed_s"]) < end
            ]
            phase_rows.append(
                {
                    "workload": workload,
                    "condition": condition,
                    "repeat": repeat,
                    "phase_index": phase_index,
                    "regime": boundary["regime"],
                    "p95": percentile(ttft, 95),
                    "waiting_area": integrate(
                        segments,
                        lambda row: float(row["waiting_q_depth"]),
                        start=start,
                        end=end,
                    ),
                    "preemptions": counter_value_at(telemetry, "preemption_count", end)
                    - counter_value_at(telemetry, "preemption_count", start),
                    "entries": sum(event.get("direction") == ENTRY for event in phase_transitions),
                    "releases": sum(event.get("direction") == RESTORE for event in phase_transitions),
                }
            )

    phase_lookup = {
        (row["workload"], row["condition"], row["repeat"], row["phase_index"]): row
        for row in phase_rows
    }

    def epsilon(workload: str, phase_index: int) -> float:
        fp16 = [
            phase_lookup[(workload, "runtime_static_fp16", repeat, phase_index)]["p95"]
            for repeat in (0, 1)
        ]
        dynamic = [
            phase_lookup[(workload, "closed_loop_dynamic", repeat, phase_index)]["p95"]
            for repeat in (0, 1)
        ]
        return max(
            0.111,
            max(fp16) - min(fp16),
            max(dynamic) - min(dynamic),
            0.05 * sum(fp16) / len(fp16),
        )

    low_entry_ok = True
    low_latency_ok = True
    for repeat in (0, 1):
        dynamic = phase_lookup[("low_only", "closed_loop_dynamic", repeat, 0)]
        fp16 = phase_lookup[("low_only", "runtime_static_fp16", repeat, 0)]
        low_entry_ok &= dynamic["entries"] == 0
        low_latency_ok &= abs(dynamic["p95"] - fp16["p95"]) <= epsilon(
            "low_only", 0
        )

    high_rows = [
        row
        for row in phase_rows
        if row["condition"] == "closed_loop_dynamic" and row["regime"] == "high"
    ]
    high_ok = len(high_rows) == 6
    for dynamic in high_rows:
        fp16 = phase_lookup[
            (
                dynamic["workload"],
                "runtime_static_fp16",
                dynamic["repeat"],
                dynamic["phase_index"],
            )
        ]
        high_ok &= (
            fp16["p95"] - dynamic["p95"]
            > epsilon(dynamic["workload"], dynamic["phase_index"])
            and dynamic["waiting_area"] < fp16["waiting_area"]
            and dynamic["preemptions"] <= fp16["preemptions"]
            and dynamic["entries"] == 1
            and dynamic["releases"] == 0
        )

    throughput_ok = True
    throughput_records: list[dict[str, Any]] = []
    for workload in ("low_only", "heldout_one_cycle", "heldout_two_cycle"):
        fp16 = [
            run_data[(workload, "runtime_static_fp16", repeat)]["metadata"]
            for repeat in (0, 1)
        ]
        dynamic = [
            run_data[(workload, "closed_loop_dynamic", repeat)]["metadata"]
            for repeat in (0, 1)
        ]
        ratios = [
            (
                int(dynamic[index]["completed_request_count"])
                / float(dynamic[index]["measurement_duration_s"])
            )
            / (
                int(fp16[index]["completed_request_count"])
                / float(fp16[index]["measurement_duration_s"])
            )
            for index in range(2)
        ]
        aggregate = sum(
            int(row["completed_request_count"]) / float(row["measurement_duration_s"])
            for row in dynamic
        ) / sum(
            int(row["completed_request_count"]) / float(row["measurement_duration_s"])
            for row in fp16
        )
        passed = min(ratios) >= 0.98 and aggregate >= 0.98
        throughput_ok &= passed
        throughput_records.append(
            {"workload": workload, "repeat_ratios": ratios, "aggregate_ratio": aggregate, "passed": passed}
        )

    expected_directions = {
        "low_only": [],
        "heldout_one_cycle": [ENTRY, RESTORE],
        "heldout_two_cycle": [ENTRY, RESTORE, ENTRY, RESTORE],
    }
    sequence_ok = True
    recovery_ok = True
    recovery_count = 0
    no_chatter = True
    for workload in ("low_only", "heldout_one_cycle", "heldout_two_cycle"):
        for repeat in (0, 1):
            data = run_data[(workload, "closed_loop_dynamic", repeat)]
            transitions = data["transitions"]
            directions = [event.get("direction") for event in transitions]
            sequence_ok &= directions == expected_directions[workload]
            if workload != "low_only":
                sequence_ok &= (data["metadata"].get("final_engine_snapshot") or {}).get(
                    "precision_state"
                ) == "FP16"
            restores = [event for event in transitions if event.get("direction") == RESTORE]
            for restore in restores:
                recovery_count += 1
                lifecycle = (restore.get("engine_trace") or {}).get("restore_lifecycle") or {}
                start_ns = int(data["metadata"]["measurement_start_time_ns"])
                resume_s = (int(lifecycle["admission_resume_ns"]) - start_ns) / 1e9
                request_s = float(restore["requested_elapsed_s"])
                phase = data["boundaries"][0]
                for boundary in data["boundaries"]:
                    if float(boundary["start_offset_s"]) > request_s:
                        break
                    phase = boundary
                post = [
                    row
                    for row in data["requests"]
                    if int(row.get("phase_index", -1)) == int(phase["phase_index"])
                    and float(row["actual_arrival_offset_s"]) > resume_s
                    and float(row["planned_arrival_offset_s"])
                    <= float(phase["last_arrival_offset_s"])
                ]
                catch_up = catch_up_duration(
                    data["telemetry"],
                    float(data["metadata"]["measurement_duration_s"]),
                    resume_s,
                )
                next_entry = next(
                    (
                        event
                        for event in transitions
                        if event.get("direction") == ENTRY
                        and float(event["requested_elapsed_s"]) > request_s
                    ),
                    None,
                )
                next_phase = next(
                    (
                        boundary
                        for boundary in data["boundaries"]
                        if float(boundary["start_offset_s"])
                        > float(phase["start_offset_s"])
                    ),
                    None,
                )
                chatter_ok = next_entry is None or (
                    float(next_entry["requested_elapsed_s"]) - resume_s >= 15.0
                    and next_phase is not None
                    and next_phase["regime"] == "high"
                    and float(next_entry["requested_elapsed_s"])
                    >= float(next_phase["start_offset_s"])
                )
                no_chatter &= chatter_ok
                dynamic_phase = phase_lookup[
                    (workload, "closed_loop_dynamic", repeat, int(phase["phase_index"]))
                ]
                fp16_phase = phase_lookup[
                    (workload, "runtime_static_fp16", repeat, int(phase["phase_index"]))
                ]
                awq_phase = phase_lookup[
                    (workload, "runtime_static_awq_w4_16", repeat, int(phase["phase_index"]))
                ]
                noise = epsilon(workload, int(phase["phase_index"]))
                not_worse_both = not (
                    dynamic_phase["p95"] > fp16_phase["p95"] + noise
                    and dynamic_phase["p95"] > awq_phase["p95"] + noise
                )
                recovery_ok &= (
                    lifecycle_integrity(restore)
                    and phase["regime"] == "low"
                    and len(post) >= 8
                    and sum(
                        row.get("prefill_precision_state") == "FP16"
                        and int(row.get("fp16_output_step_count", 0)) > 0
                        for row in post
                    )
                    >= 8
                    and float(phase["last_arrival_offset_s"]) - resume_s >= 20.0
                    and resume_s
                    - (int(lifecycle["admission_pause_ns"]) - start_ns) / 1e9
                    <= 5.0
                    and catch_up is not None
                    and catch_up <= 10.0
                    and chatter_ok
                    and not_worse_both
                )

    gpu_ok = all(
        any(
            row.get("gpu_environment_source") == "pynvml"
            for row in read_jsonl(
                root / "runs" / selected[plan_row["run_id"]] / "gpu_environment.jsonl"
            )
        )
        for plan_row in plan
    )
    checks = {
        "all_16_runs_raw_valid": len(raw_results) == 16
        and all(row["all_pass"] for row in raw_results.values()),
        "low_only_zero_false_entries_both": low_entry_ok,
        "low_only_fp16_like_within_noise_both": low_latency_ok,
        "every_high_phase_preserves_entry_benefit": high_ok,
        "throughput_noninferior_with_0p98_margin": throughput_ok,
        "every_release_has_complete_useful_bounded_recovery": recovery_count == 6
        and recovery_ok,
        "ordered_one_and_two_cycle_sequences_both_repeats": sequence_ok,
        "no_chatter_or_release_during_high": no_chatter
        and all(row["releases"] == 0 for row in high_rows),
        "gpu_clock_temperature_power_available": gpu_ok,
    }
    scientific = {
        key: value
        for key, value in checks.items()
        if key != "gpu_clock_temperature_power_available"
    }
    return {
        "checks": checks,
        "throughput": throughput_records,
        "release_side_systems_decision": "GO" if all(scientific.values()) else "NO-GO",
    }


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
    source_mismatches = {
        relative: {
            "preregistered_sha256": manifest["source_sha256_at_preregistration"][relative],
            "current_sha256": digest,
        }
        for relative, digest in source_hashes.items()
        if digest != manifest["source_sha256_at_preregistration"][relative]
    }
    correction_path = root / "analysis_correction.json"
    correction = read_json(correction_path) if correction_path.is_file() else {}
    declared_corrections = correction.get("source_corrections", {})
    check(
        "actuation_sources_unchanged_and_analysis_repair_bounded",
        set(source_mismatches) == set(declared_corrections)
        and all(
            record.get("preregistered_sha256")
            == source_mismatches[relative]["preregistered_sha256"]
            and record.get("corrected_sha256")
            == source_mismatches[relative]["current_sha256"]
            for relative, record in declared_corrections.items()
        )
        and set(source_mismatches)
        <= {
            "swiftLLM/benchmark/analyze_release_side.py",
            "swiftLLM/benchmark/audit_release_side.py",
            "swiftLLM/benchmark/test_release_analysis.py",
        }
        and correction.get("controller_or_gate_changed") is False,
        {"mismatches": source_mismatches, "declared": declared_corrections},
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
    check(
        "post_result_analysis_correction_disclosed",
        correction_path.is_file()
        and correction.get("reason")
        == "percentile helper expects 0-100, but the preregistered analyzer passed fractions"
        and correction.get("controller_or_gate_changed") is False,
        correction,
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
            repo, commit, str(root.resolve().relative_to(repo) / "protocol_manifest.json")
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
    independently_recomputed = independent_decision_from_raw(
        root, plan, selected, raw_results
    )
    check(
        "scientific_gates_and_decision_match_independent_raw_recomputation",
        decision.get("checks") == independently_recomputed["checks"]
        and decision.get("release_side_systems_decision")
        == independently_recomputed["release_side_systems_decision"],
        independently_recomputed,
    )
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
