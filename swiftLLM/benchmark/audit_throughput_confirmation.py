"""Independently audit v11 paired evidence, policy preservation, and v10 immutability."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from scipy.stats import t as student_t

from .run_throughput_confirmation_condition import ARTIFACT_VERSION, sha256_file


MARGIN = 0.98
ALPHA = 0.025
ENTRY = "FP16_TO_AWQ_MARLIN_W4_16"
RESTORE = "AWQ_MARLIN_W4_16_TO_FP16"
REQUIRED_RAW = (
    "metadata.json",
    "requests.jsonl",
    "telemetry.jsonl",
    "batches.jsonl",
    "controller.jsonl",
    "gpu_environment.jsonl",
    "event_loop_lag.jsonl",
    "transitions.jsonl",
    "initialization_transitions.jsonl",
    "runtime_preparation.json",
    "summary.json",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file_path.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        data = file_path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def expected_directions(workload: str) -> list[str]:
    return {
        "low_only": [],
        "heldout_one_cycle": [ENTRY, RESTORE],
        "heldout_two_cycle": [ENTRY, RESTORE, ENTRY, RESTORE],
    }[workload]


def raw_check(run_dir: Path, row: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    if not all((run_dir / name).is_file() for name in REQUIRED_RAW):
        return False, {"missing": [name for name in REQUIRED_RAW if not (run_dir / name).is_file()]}
    metadata = read_json(run_dir / "metadata.json")
    requests = read_jsonl(run_dir / "requests.jsonl")
    telemetry = read_jsonl(run_dir / "telemetry.jsonl")
    controller = read_jsonl(run_dir / "controller.jsonl")
    gpu = read_jsonl(run_dir / "gpu_environment.jsonl")
    lag = read_jsonl(run_dir / "event_loop_lag.jsonl")
    transitions = read_jsonl(run_dir / "transitions.jsonl")
    expected = int(row["request_count"])
    directions = [event.get("direction") for event in transitions]
    dynamic = row["condition"] == "closed_loop_dynamic"
    final = metadata.get("final_engine_snapshot") or {}
    integrity = bool(
        len(requests) == expected
        and len({request.get("benchmark_request_id") for request in requests}) == expected
        and all(
            request.get("status") == "completed"
            and int(request.get("output_token_count", -1)) == 512
            and len(request.get("step_precision_states", [])) == 512
            and request.get("engine_request_id_stable") is True
            and request.get("input_positions_contiguous") is True
            and request.get("error") is None
            for request in requests
        )
    )
    controller_timing = (
        all(
            int(sample.get("controller_evaluate_wall_ns", 0)) > 0
            and int(sample.get("controller_window_wall_ns", 0)) >= 0
            and int(sample.get("controller_trace_fields_wall_ns", 0)) > 0
            for sample in controller
        )
        if dynamic
        else not controller
    )
    transitions_valid = directions == (expected_directions(str(row["workload_class"])) if dynamic else [])
    transition_integrity = all(
        event.get("result") == "success"
        and (event.get("engine_trace") or {}).get("status") == "success"
        and (
            not (event.get("engine_trace") or {}).get("kv_resize")
            or (
                ((event["engine_trace"]["kv_resize"].get("integrity_before") or {}).get("logical_digest"))
                == ((event["engine_trace"]["kv_resize"].get("integrity_after") or {}).get("logical_digest"))
            )
        )
        for event in transitions
    )
    phase_correct = True
    if dynamic:
        phase_by_request = {str(request["benchmark_request_id"]): request.get("phase") for request in requests}
        for event in transitions:
            requested_s = float(event["requested_elapsed_s"])
            nearest = min(requests, key=lambda request: abs(float(request["planned_arrival_offset_s"]) - requested_s))
            phase = str(nearest.get("phase"))
            if event["direction"] == ENTRY:
                phase_correct &= "high" in phase
            else:
                phase_correct &= "low" in phase
                lifecycle = (event.get("engine_trace") or {}).get("restore_lifecycle") or {}
                observed = event.get("release_intent_observation") or {}
                if int(observed.get("blocks_above_fp16_base", 0)) > 0:
                    phase_correct &= lifecycle.get("drain_required") is True
                resume_ns = lifecycle.get("admission_resume_ns")
                if resume_ns is not None:
                    later_low = [
                        request for request in requests
                        if request.get("phase") == nearest.get("phase")
                        and int(request["arrival_time_ns"]) > int(resume_ns)
                    ]
                    phase_correct &= bool(later_low) and all(
                        request.get("prefill_precision_state") == "FP16"
                        and int(request.get("fp16_output_step_count", 0)) > 0
                        for request in later_low
                    )
    passed = bool(
        metadata.get("experiment") == ARTIFACT_VERSION
        and metadata.get("planned_run_id") == row["run_id"]
        and metadata.get("condition") == row["condition"]
        and metadata.get("workload_sha256") == row["workload_sha256"]
        and metadata.get("completed_request_count") == expected
        and metadata.get("generated_output_token_count") == expected * 512
        and float(metadata.get("measurement_duration_s", 0)) > 0
        and integrity
        and telemetry
        and lag
        and any(sample.get("gpu_environment_source") == "pynvml" for sample in gpu)
        and controller_timing
        and transitions_valid
        and transition_integrity
        and phase_correct
        and final.get("precision_state") == "FP16"
        and int(final.get("num_gpu_blocks", -1)) == 1759
        and not (run_dir / "failure.json").exists()
    )
    return passed, {
        "request_integrity": integrity,
        "controller_timing": controller_timing,
        "transition_sequence": directions,
        "transition_integrity": transition_integrity,
        "phase_and_restore_behavior": phase_correct,
        "final_state": final.get("precision_state"),
        "final_capacity": final.get("num_gpu_blocks"),
        "nvml_samples": sum(sample.get("gpu_environment_source") == "pynvml" for sample in gpu),
    }


def independent_inference(plan: list[dict[str, Any]], root: Path, selected: dict[str, str]) -> list[dict[str, Any]]:
    pairs: dict[str, dict[str, float | str]] = {}
    workload_by_pair: dict[str, str] = {}
    for row in plan:
        metadata = read_json(root / "runs" / selected[row["run_id"]] / "metadata.json")
        throughput = int(metadata["completed_request_count"]) / float(metadata["measurement_duration_s"])
        pairs.setdefault(str(row["pair_id"]), {})[str(row["condition"])] = throughput
        workload_by_pair[str(row["pair_id"])] = str(row["workload_class"])
    output = []
    for workload in sorted(set(workload_by_pair.values())):
        values = [
            math.log(float(pair["closed_loop_dynamic"]) / float(pair["runtime_static_fp16"]))
            for pair_id, pair in sorted(pairs.items())
            if workload_by_pair[pair_id] == workload
        ]
        avg = mean(values)
        sd = stdev(values)
        se = sd / math.sqrt(len(values))
        lower = math.exp(avg - float(student_t.ppf(1 - ALPHA, len(values) - 1)) * se)
        output.append(
            {
                "workload": workload,
                "pair_count": len(values),
                "geometric_mean_ratio": math.exp(avg),
                "one_sided_lower_confidence_bound": lower,
                "noninferior": lower > MARGIN,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan-key", choices=("phase_a", "phase_c"), required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    root = args.root.resolve()
    manifest = read_json(args.manifest)
    plan = read_json(args.plan)["runs"]
    status = read_json(args.status)
    selected = status.get("selected_runs", {})
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    check("manifest_artifact_version", manifest.get("artifact_version") == ARTIFACT_VERSION)
    protocol = repo / manifest["repository"]["protocol_path"]
    check("protocol_hash", sha256_file(protocol) == manifest["repository"]["protocol_sha256"])
    check("run_plan_hash", sha256_file(args.plan) == manifest["run_plans"][args.plan_key]["sha256"])
    current_sources = {
        path: sha256_file(repo / path) for path in manifest["source_sha256_at_preregistration"]
    }
    check("preregistered_sources_unchanged", current_sources == manifest["source_sha256_at_preregistration"])
    preserved = {
        path: tree_sha256(repo / path) for path in manifest["preserved_namespace_tree_sha256"]
    }
    check("v10_and_prior_namespaces_unchanged", preserved == manifest["preserved_namespace_tree_sha256"])
    v10_decision = read_json(repo / "benchmark-results/release-side-runtime-v10/analysis/decision.json")
    check("v10_decision_remains_no_go", v10_decision.get("release_side_systems_decision") == "NO-GO")
    parity = read_json(root / "archived_policy_parity.json")
    check(
        "archived_policy_parity_passes",
        parity.get("status") == "PASS"
        and parity.get("coverage", {}).get("v7_streams") == 36
        and parity.get("coverage", {}).get("v9_dynamic_streams") == 6
        and parity.get("coverage", {}).get("v10_dynamic_streams") == 6,
        parity.get("coverage"),
    )
    check("execution_complete", status.get("complete") is True and len(selected) == len(plan))

    raw_results = {}
    for row in plan:
        actual = selected.get(row["run_id"])
        passed, evidence = raw_check(root / "runs" / actual, row) if actual else (False, {"missing_selection": True})
        raw_results[row["run_id"]] = {"passed": passed, **evidence}
    check("all_selected_raw_runs_valid", all(row["passed"] for row in raw_results.values()), raw_results)

    independent = independent_inference(plan, root, selected)
    analyzed = read_json(root / "analysis" / args.plan_key / "inference.json")
    compared = all(
        left["workload"] == right["workload"]
        and left["pair_count"] == right["pair_count"]
        and math.isclose(left["geometric_mean_ratio"], right["geometric_mean_ratio"], rel_tol=0, abs_tol=1e-12)
        and math.isclose(left["one_sided_lower_confidence_bound"], right["one_sided_lower_confidence_bound"], rel_tol=0, abs_tol=1e-12)
        and left["noninferior"] == right["noninferior"]
        for left, right in zip(independent, analyzed)
    ) and len(independent) == len(analyzed)
    check("throughput_inference_independently_recomputed", compared, independent)

    decision = read_json(root / "analysis" / args.plan_key / "decision.json")
    check(
        "decision_matches_inference_and_guardrails",
        decision.get("all_workloads_noninferior") == all(row["noninferior"] for row in independent)
        and decision.get("all_guardrails_pass") is True,
        decision,
    )
    result = {
        "schema_version": 1,
        "plan_key": args.plan_key,
        "status": "PASS" if all(row["passed"] for row in checks) else "FAIL",
        "passed_check_count": sum(row["passed"] for row in checks),
        "check_count": len(checks),
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
