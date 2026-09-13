"""Audit v7 crossover artifacts against the frozen protocol and user objective."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from .analyze_crossover import CONDITIONS, EXPECTED_BLOCKS, REQUIRED_BATCH_FIELDS, read_jsonl


REQUIRED_ANALYSIS_COLUMNS = {
    "ttft_s_p50",
    "ttft_s_p95",
    "ttft_s_p99",
    "strict_ttft_gt_2s_percent",
    "queueing_delay_s_p50",
    "queueing_delay_s_p95",
    "queueing_delay_s_p99",
    "first_prefill_to_first_output_s_p50",
    "first_prefill_to_first_output_s_p95",
    "first_prefill_to_first_output_s_p99",
    "tpot_s_p50",
    "tpot_s_p95",
    "tpot_s_p99",
    "completed_throughput_rps",
    "waiting_q_depth_p50",
    "waiting_q_depth_p95",
    "waiting_q_depth_p99",
    "running_q_count_p50",
    "running_q_count_p95",
    "running_q_count_p99",
    "swapped_q_count_p50",
    "swapped_q_count_p95",
    "swapped_q_count_p99",
    "num_decoding_gpu_blocks_p50",
    "num_decoding_gpu_blocks_p95",
    "num_decoding_gpu_blocks_p99",
    "logical_kv_utilization_p50",
    "logical_kv_utilization_p95",
    "logical_kv_utilization_p99",
    "preemption_count",
    "prefill_sequence_count_p50",
    "prefill_sequence_count_p95",
    "prefill_sequence_count_p99",
    "prefill_effective_gemm_m_p50",
    "prefill_effective_gemm_m_p95",
    "prefill_effective_gemm_m_p99",
    "prefill_execution_duration_s_p50",
    "prefill_execution_duration_s_p95",
    "prefill_execution_duration_s_p99",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_quiet(*args: str) -> bool:
    return subprocess.run(
        ["git", *args], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
    ).returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    manifest = json.loads((root / "protocol_manifest.json").read_text(encoding="utf-8"))
    plan = json.loads((root / "run-plan.json").read_text(encoding="utf-8"))
    workload = json.loads((root / "input/workload_metadata.json").read_text(encoding="utf-8"))
    decision = json.loads((root / "analysis/crossover_decision.json").read_text(encoding="utf-8"))

    check("protocol status is pre-registered", manifest["status"] == "pre_registered_before_new_intermediate_serving_results")
    check("protocol commit exists", git_quiet("cat-file", "-e", "72da2d5^{commit}"))
    check("protocol commit precedes first result commit", git_quiet("merge-base", "--is-ancestor", "72da2d5", "83996c6"))
    check("all prior v2/v4/v5/v6 namespaces unchanged", git_quiet(
        "diff", "--quiet", "d29cb79", "--",
        "benchmark-results/static-quantization-quality-latency",
        "benchmark-results/table5-substrate-v4",
        "benchmark-results/static-frontier-v5",
        "benchmark-results/packed-int4-backend-v6",
        "docs/packed-int4-backend-v6",
    ))
    check("scheduler unchanged", git_quiet("diff", "--quiet", "d29cb79", "--", "swiftLLM/swiftllm/server/scheduler.py"))
    check("KV/model/backend execution code unchanged", git_quiet(
        "diff", "--quiet", "d29cb79", "--",
        "swiftLLM/swiftllm/worker/block_manager.py",
        "swiftLLM/swiftllm/worker/model.py",
        "swiftLLM/swiftllm/worker/weight.py",
        "swiftLLM/swiftllm/worker/kernels",
        "swiftLLM/swiftllm/worker/layers",
    ))

    scales = manifest["workloads"]["fixed_scales_low_to_high_load"]
    check("nine predeclared loads", scales == [6.0, 5.75, 5.5, 5.25, 5.0, 4.75, 4.5, 4.25, 4.0])
    check("36 unique predeclared runs", len(plan["runs"]) == 36 and len({row["run_id"] for row in plan["runs"]}) == 36)
    check("two states only", {row["condition"] for row in plan["runs"]} == set(CONDITIONS))
    check("two repeats per state/load", all(
        sum(row["scale"] == scale and row["condition"] == condition for row in plan["runs"]) == 2
        for scale in scales for condition in CONDITIONS
    ))
    check("run-plan hash frozen in manifest", sha256(root / "run-plan.json") == manifest["serving"]["run_plan_sha256"])
    check("workload metadata hash frozen in manifest", sha256(root / "input/workload_metadata.json") == manifest["workloads"]["metadata_sha256"])
    check("common request-content hash", len({row["request_content_sha256"] for row in workload["outputs"]}) == 1)
    endpoint_checks = [row for row in workload["outputs"] if float(row["scale"]) in (6.0, 4.0)]
    check("scale endpoints byte-identical to v5", len(endpoint_checks) == 2 and all(row["byte_identical_to_v5_endpoint"] for row in endpoint_checks))

    protocol_hash = sha256(root / "protocol_manifest.json")
    run_failures = []
    total_requests = total_batches = total_prefills = 0
    for plan_row in plan["runs"]:
        run_dir = root / "runs" / plan_row["run_id"]
        required = ("metadata.json", "requests.jsonl", "telemetry.jsonl", "batches.jsonl", "summary.json")
        if not all((run_dir / name).is_file() for name in required):
            run_failures.append(f"{plan_row['run_id']}: missing raw file")
            continue
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        requests = read_jsonl(run_dir / "requests.jsonl")
        telemetry = read_jsonl(run_dir / "telemetry.jsonl")
        batches = read_jsonl(run_dir / "batches.jsonl")
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        total_requests += len(requests)
        total_batches += len(batches)
        total_prefills += sum(int(row["num_prefill_sequences"]) > 0 for row in batches)
        failures = []
        failures += ["metadata identity"] if not (
            metadata["run_id"] == plan_row["run_id"]
            and metadata["condition"] == plan_row["condition"]
            and float(metadata["workload_time_scale"]) == float(plan_row["scale"])
        ) else []
        failures += ["workload/protocol hash"] if not (
            metadata["workload_sha256"] == plan_row["workload_sha256"]
            and metadata["protocol_manifest_sha256"] == protocol_hash
        ) else []
        config = metadata["engine_config_actual"]
        failures += ["engine config"] if not (
            config["block_size"] == 16
            and config["gpu_mem_utilization"] == 0.99
            and config["num_cpu_blocks"] == 4096
            and config["max_batch_size"] == 32
            and config["max_tokens_in_batch"] == 49152
        ) else []
        failures += ["GPU/device"] if not (
            metadata["gpu"]["gpu_name"] == "NVIDIA GeForce RTX 3090"
            and metadata["cuda_device_requested"] == "5"
        ) else []
        failures += ["safe blocks"] if metadata["num_gpu_blocks"] != EXPECTED_BLOCKS[metadata["condition"]] else []
        failures += ["request completion/length"] if not (
            len(requests) == 64
            and summary["completed_request_count"] == 64
            and summary["failed_or_incomplete_request_count"] == 0
            and all(row["status"] == "completed" and row["prompt_token_count"] == 1024 and row["output_token_count"] == 512 for row in requests)
        ) else []
        failures += ["telemetry empty"] if not telemetry else []
        failures += ["batch telemetry fields"] if not batches or not all(REQUIRED_BATCH_FIELDS <= row.keys() for row in batches) else []
        failures += ["batch identity"] if batches and not all(
            int(row["effective_gemm_m"]) == int(row["total_prefill_tokens"]) + int(row["num_decoding_sequences"])
            and int(row["safe_kv_blocks"]) == EXPECTED_BLOCKS[metadata["condition"]]
            for row in batches
        ) else []
        failures += ["batch indices"] if [row.get("batch_index") for row in batches] != list(range(len(batches))) else []
        if failures:
            run_failures.append(f"{plan_row['run_id']}: {', '.join(failures)}")
    check("all 36 raw runs protocol-valid", not run_failures, run_failures)
    check("2304 complete requests", total_requests == 36 * 64, total_requests)
    check("batch observation coverage", total_batches == 83041 and total_prefills == 421, {"forward": total_batches, "prefill": total_prefills})

    micro = [
        json.loads((root / f"microbenchmark/prefill-{condition}.json").read_text(encoding="utf-8"))
        for condition in CONDITIONS
    ]
    expected_m = manifest["prefill_microbenchmark"]["effective_gemm_m"]
    check("pre-registered multi-M microbenchmarks", all(
        [row["effective_gemm_m"] for row in result["results"]] == expected_m
        and all(row["sample_count"] == 7 for row in result["results"])
        for result in micro
    ))
    supplemental = [
        json.loads((root / f"microbenchmark/supplemental-prefill-m2048-{condition}.json").read_text(encoding="utf-8"))
        for condition in CONDITIONS
    ]
    check("bounded matched M2048 supplement", all(result["results"][0]["effective_gemm_m"] == 2048 and result["results"][0]["sample_count"] == 7 for result in supplemental))

    import csv
    with (root / "analysis/serving_runs.csv").open(newline="", encoding="utf-8") as handle:
        serving_rows = list(csv.DictReader(handle))
    check("one fully reported row per serving run", len(serving_rows) == 36 and REQUIRED_ANALYSIS_COLUMNS <= serving_rows[0].keys())
    check("all analysis rows protocol-valid", all(row["protocol_valid"] == "True" for row in serving_rows))
    point_rows = list(csv.DictReader((root / "analysis/crossover_points.csv").open(newline="", encoding="utf-8")))
    classes = {float(row["scale"]): row["classification"] for row in point_rows}
    check("predeclared point classification reproduced", classes == {
        6.0: "FP16-default/non-inferior",
        5.75: "FP16-default/non-inferior",
        5.5: "FP16-default/non-inferior",
        5.25: "ambiguous",
        5.0: "AWQ-preferred",
        4.75: "ambiguous",
        4.5: "AWQ-preferred",
        4.25: "AWQ-preferred",
        4.0: "AWQ-preferred",
    }, classes)
    check("decision is GO for exact state pair", decision["decision"] == "GO" and decision["validated_state_pair"] == ["FP16", "AWQ-W4-16"])
    check("all frozen decision gates pass", all(decision["gates"].values()), decision["gates"])
    check("two qualifying causal signals", decision["qualifying_online_signals"] == ["capacity_margin_queue_integral", "sustained_compound_pressure"])
    check("no controller implemented", decision["controller_implemented"] is False)

    required_docs = ("protocol.md", "prefill-diagnosis.md", "final-report.md", "completion-audit.md")
    check("required human reports exist", all((Path("docs/fp16-awq-crossover-v7") / name).is_file() for name in required_docs))
    report = (Path("docs/fp16-awq-crossover-v7/final-report.md").read_text(encoding="utf-8") if Path("docs/fp16-awq-crossover-v7/final-report.md").is_file() else "")
    check("required final sections", all(section in report for section in (
        "CONFIRMED FINDINGS", "SUPPORTED BUT UNCERTAIN FINDINGS", "BLOCKED QUESTIONS", "REMAINING UNCERTAINTY", "CROSSOVER DECISION: **GO**"
    )))

    result = {
        "schema_version": 1,
        "status": "PASS" if all(row["passed"] for row in checks) else "FAIL",
        "check_count": len(checks),
        "failure_count": sum(not row["passed"] for row in checks),
        "checks": checks,
        "decision": decision["decision"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
