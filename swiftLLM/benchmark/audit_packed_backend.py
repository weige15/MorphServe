"""Completion audit for the packed INT4 backend v6 evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from . import analyze_static_frontier as frontier


CONDITIONS = {
    "fp16_0": 0,
    "awq_w4_8": 8,
    "awq_w4_16": 16,
    "awq_w4_32": 32,
}
SCALES_REPEATS = {8: (0,), 6: (0, 1), 4: (0,)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checks: list[dict] = []

    def check(name: str, condition: bool, detail=None) -> None:
        checks.append(
            {"name": name, "status": "PASS" if condition else "FAIL", "detail": detail}
        )

    root = args.root
    protocol = read_json(root / "protocol_manifest.json")
    check("protocol pre-registered", protocol["manifest_status"] == "recorded_before_v6_gpu_kernel_measurements")
    candidate = protocol["candidate"]
    check(
        "quantization config frozen",
        (candidate["weight_bits"], candidate["group_size"], candidate["zero_point"])
        == (4, 128, True),
    )
    check("dynamic work excluded", all(term in protocol["scope"]["excluded"] for term in ("runtime controller", "dynamic adaptation", "runtime layer swapping", "KVResizer", "scheduler changes")))

    phase_a = read_json(root / "phase-a" / "awq-marlin-kernel-feasibility.json")
    for key in (
        "all_shapes_supported",
        "all_outputs_finite",
        "relative_l2_below_0p20",
        "cosine_above_0p98",
        "packed_fraction_below_0p35",
        "temporary_below_64_mib",
        "no_static_dequantize_or_matmul_hot_path",
        "phase_a_representation_and_numerical_gate_pass",
    ):
        check(f"Phase A {key}", phase_a["gates"][key] is True)
    check("Phase A exact row counts", phase_a["configuration"]["rows"] == [1, 16, 128, 1024])
    check("Phase A samples", phase_a["configuration"]["warmup"] == 10 and phase_a["configuration"]["repeats"] == 20)
    check("Phase A shape classes", len(phase_a["matrices"]) == 5)
    inventory = read_json(root / "phase-a" / "backend-inventory.json")
    compatible = inventory["compatible_kernel_environment"]
    check("vLLM 0.11.2 compiled ops load", compatible["custom_ops_import"] == "PASS" and compatible["packages"]["vllm"] == "0.11.2")
    check("RTX 3090 SM86", compatible["cuda"]["name"] == "NVIDIA GeForce RTX 3090" and compatible["cuda"]["capability"] == [8, 6])
    check("all installed shape checks pass", all(value[0] for value in compatible["ops"]["shape_support"].values()))
    check("Marlin workspace is 328 bytes", compatible["ops"]["workspace_bytes"] == 328)
    check("validated environment ABI blocker recorded", "undefined symbol" in inventory["main_swiftllm_environment"]["custom_ops_import"])

    checkpoint_manifest = read_json(root / "phase-b" / "awq-checkpoint-manifest.json")
    emitted = checkpoint_manifest["emitted_quantization_config"]
    check("offline AWQ emitted config", (emitted["quant_method"], emitted["bits"], emitted["group_size"], emitted["zero_point"], emitted["version"]) == ("awq", 4, 128, True, "gemm"))
    calibration_path = Path(checkpoint_manifest["calibration"]["selected_jsonl"])
    check("calibration hash", sha256(calibration_path) == checkpoint_manifest["calibration"]["selected_jsonl_sha256"])
    check("calibration count", len(read_jsonl(calibration_path)) == 128 and checkpoint_manifest["calibration"]["full_512_token_blocks"] == 59)
    checkpoint_path = Path(checkpoint_manifest["output_checkpoint"])
    all_checkpoint_hashes = all(
        (checkpoint_path / row["name"]).stat().st_size == row["bytes"]
        and sha256(checkpoint_path / row["name"]) == row["sha256"]
        for row in checkpoint_manifest["files"]
    )
    check("external checkpoint files and hashes", all_checkpoint_hashes, len(checkpoint_manifest["files"]))

    checkpoint_sanity = read_json(root / "phase-b" / "awq-checkpoint-sanity.json")
    check("672 packed component mappings", checkpoint_sanity["all_component_checks_pass"] and sum(len(row["components"]) for row in checkpoint_sanity["component_checks"]) == 672)
    check("packed mapping numerics", checkpoint_sanity["all_numerical_checks_finite"] and checkpoint_sanity["max_marlin_vs_autoawq_dequant_relative_l2"] < 0.001)
    check("packed object has no FP16 matrix", checkpoint_sanity["full_fp16_matrix_retained_by_packed_object"] is False)
    parity = read_json(root / "phase-b" / "fp16-hf-parity-torch29.json")
    check("torch-2.9 FP16 parity", parity["first_token_match"] and parity["greedy_exact_match"] and parity["greedy_common_prefix_tokens"] == 8)
    for layers in (0, 8, 16, 32):
        smoke = read_json(root / "phase-b" / "smoke" / f"awq-w4-{layers}.json")
        check(f"W4-{layers} short generation", len(smoke["output_token_ids"]) == 8 and all(isinstance(value, int) for value in smoke["output_token_ids"]))
        if layers:
            selected = smoke["layer_layout"][:layers]
            unselected = smoke["layer_layout"][layers:]
            check(f"W4-{layers} selected packed", all(row["selected"] and row["q_proj_type"] == "AWQMarlinMatrix" and row["norm_source"] == "awq_checkpoint" for row in selected))
            check(f"W4-{layers} unselected FP16", all(not row["selected"] and row["q_proj_type"] == "Tensor" and row["norm_source"] == "base_checkpoint" for row in unselected))
            check(f"W4-{layers} fused MLP mapping", all(row["mlp_projection_type"] == "AWQMarlinMatrix" and len(row["mlp_source_keys"]) == 6 for row in selected))
    separate = read_json(root / "phase-b" / "separate-mlp-full-profile-w4-8.json")
    fused = read_json(root / "phase-b" / "fused-mlp-full-profile-w4-8.json")
    separate_temp = separate["profile_peak_allocated_bytes"] - separate["resident_weight_bytes_after_load"]
    fused_temp = fused["profile_peak_allocated_bytes"] - fused["resident_weight_bytes_after_load"]
    check("MLP fusion removes activation pathology", fused_temp < separate_temp and fused["num_gpu_blocks"] > separate["num_gpu_blocks"], {"separate_blocks": separate["num_gpu_blocks"], "fused_blocks": fused["num_gpu_blocks"]})

    resource_rows = list(csv.DictReader((root / "phase-c" / "resource-performance-gate.csv").open()))
    resources = {row["condition"]: row for row in resource_rows}
    check("Phase C four states", set(resources) == set(CONDITIONS))
    block_values = []
    for condition, layers in CONDITIONS.items():
        profiles = [read_json(root / "phase-c" / f"profile-{condition}-rep{repeat}.json") for repeat in (0, 1)]
        blocks = [row["num_gpu_blocks"] for row in profiles]
        block_values.append(blocks[0])
        check(f"{condition} profile repeatability", blocks[0] == blocks[1] and all(row["engine_config"]["max_batch_size"] == 32 and row["engine_config"]["max_tokens_in_batch"] == 49152 for row in profiles))
        micro = read_json(root / "phase-c" / f"micro-{condition}.json")
        check(f"{condition} seven-sample microbenchmark", micro["repeats"] == 7 and len(micro["prefill_ms"]) == 7 and len(micro["decode_one_token_ms"]) == 7)
        check(f"{condition} finite short profile generation", all(len(row["output_token_ids"]) == 8 for row in profiles))
    check("safe KV monotonic", block_values == sorted(block_values) and block_values == [1768, 3034, 4286, 6766])
    check("all W4 states pass Phase C", all(row["phase_d_viable"] == "True" for row in resource_rows[1:]))

    plan = read_json(root / "phase-d-run-plan.json")
    check("serving plan frozen", plan["status"] == "frozen_before_any_v6_serving_run" and len(plan["runs"]) == 16)
    protocol_hash = sha256(root / "protocol_manifest.json")
    run_dirs = []
    for condition, layers in CONDITIONS.items():
        for scale, repeats in SCALES_REPEATS.items():
            for repeat in repeats:
                run_id = f"serv-{condition}-scale-{scale}-rep{repeat}"
                run_dir = root / "phase-d" / "runs" / run_id
                run_dirs.append(run_dir)
                metadata = read_json(run_dir / "metadata.json")
                requests = read_jsonl(run_dir / "requests.jsonl")
                telemetry = read_jsonl(run_dir / "telemetry.jsonl")
                summary = read_json(run_dir / "summary.json")
                check(f"{run_id} identity", metadata["condition"] == condition and metadata["quantized_layer_count"] == layers)
                check(f"{run_id} hashes", metadata["protocol_manifest_sha256"] == protocol_hash and metadata["workload_sha256"] == sha256(Path(metadata["workload_path"])))
                check(f"{run_id} exact request protocol", len(requests) == 64 and all(row["status"] == "completed" and row["prompt_token_count"] == 1024 and row["output_token_count"] == 512 for row in requests))
                check(f"{run_id} warmup excluded", metadata["warmup_policy"]["warmup_requests"] == 1 and metadata["warmup_policy"]["included_in_raw_requests"] is False)
                required_telemetry = {"waiting_q_depth", "running_q_count", "swapped_q_count", "logical_kv_utilization", "num_gpu_blocks", "swap_in_count", "swap_out_count", "preemption_count", "gpu_memory_used_bytes", "gpu_memory_free_bytes"}
                check(f"{run_id} telemetry", bool(telemetry) and all(required_telemetry <= set(row) for row in telemetry))
                check(f"{run_id} raw summary", summary["request_count"] == 64 and summary["completed_request_count"] == 64 and summary["failed_or_incomplete_request_count"] == 0)
                if condition.startswith("awq_"):
                    runtime = metadata["quantization_runtime"]
                    check(f"{run_id} packed metadata", runtime["weight_bits"] == 4 and runtime["group_size"] == 128 and runtime["zero_point"] and runtime["full_weight_dequantization_hot_path"] is False and runtime["packed_representation_bytes"] > 0)

    frontier.CONDITIONS = tuple(CONDITIONS)
    frontier.COLORS = {"fp16_0": "#1f77b4", "awq_w4_8": "#ff7f0e", "awq_w4_16": "#2ca02c", "awq_w4_32": "#d62728"}
    with tempfile.TemporaryDirectory() as directory:
        regenerated = Path(directory)
        frontier.serving_analysis(run_dirs, regenerated, near_scale=6.0)
        for filename in ("serving_runs.csv", "serving_variability.csv", "near_knee_paired_runs.csv", "serving_summary.json"):
            check(f"exact regeneration {filename}", (regenerated / filename).read_bytes() == (root / "analysis" / filename).read_bytes())

    serving_rows = list(csv.DictReader((root / "analysis" / "serving_runs.csv").open()))
    required_metrics = {"ttft_s_p50", "ttft_s_p95", "ttft_s_p99", "slo_violation_percent", "queueing_delay_s_p95", "prefill_to_first_output_s_p95", "decode_tpot_s_p95", "completed_throughput_rps", "waiting_q_depth_p95", "running_q_count_p95", "swapped_q_count_p95", "peak_logical_kv_utilization", "num_gpu_blocks", "swap_in_count", "swap_out_count", "peak_hbm_used_gib"}
    check("all required raw-derived metrics", len(serving_rows) == 16 and all(required_metrics <= set(row) for row in serving_rows))
    decision = read_json(root / "analysis" / "backend_decision.json")
    check("explicit NO-GO", decision["decision"] == "NO-GO" and not decision["eligible_states"] and all(not row["eligible"] for row in decision["eligibility"]))
    check("quality correctly gated", decision["quality_required"] is False and not (root / "phase-d" / "quality").exists())

    exploratory = list((root / "phase-d" / "exploratory-runs").glob("*/summary.json"))
    check("high-load exploratory repeats", len(exploratory) == 2 and all(read_json(path)["completed_request_count"] == 64 for path in exploratory))
    tuning = read_json(root / "phase-c" / "tuning" / "decision.json")
    qkv = read_json(root / "phase-c" / "tuning" / "qkv-fusion-kernel.json")
    qkv_profile = read_json(root / "phase-c" / "tuning" / "profile-awq_w4_8-fused-qkv.json")
    check("reduction tuning stopped", tuning["decision"] == "KEEP_DEFAULT_STOP_TUNING" and not tuning["qualifying_settings"])
    check("QKV tuning rejected on capacity", qkv["decision"] == "IMPLEMENT_FOR_INTEGRATED_GATE" and qkv_profile["num_gpu_blocks"] < 3034)

    preserved = subprocess.run(
        ["git", "diff", "--quiet", "3efe426", "--", "benchmark-results/static-quantization-quality-latency", "benchmark-results/table5-substrate-v4", "benchmark-results/static-frontier-v5"],
        check=False,
    ).returncode == 0
    scheduler_unchanged = subprocess.run(
        ["git", "diff", "--quiet", "3efe426", "--", "swiftLLM/swiftllm/server/scheduler.py"],
        check=False,
    ).returncode == 0
    check("v2/v4/v5 artifacts preserved", preserved)
    check("scheduler unchanged", scheduler_unchanged)

    report = (Path("docs/packed-int4-backend-v6/final-report.md")).read_text(encoding="utf-8")
    for heading in ("## CONFIRMED FINDINGS", "## SUPPORTED BUT UNCERTAIN FINDINGS", "## BLOCKED QUESTIONS", "## REMAINING UNCERTAINTY", "## BACKEND DECISION: **NO-GO**"):
        check(f"report section {heading}", heading in report)
    check("report forbids controller", "Do not implement a controller" in report)

    failures = [row for row in checks if row["status"] == "FAIL"]
    result = {
        "schema_version": 1,
        "status": "PASS" if not failures else "FAIL",
        "check_count": len(checks),
        "pass_count": len(checks) - len(failures),
        "failure_count": len(failures),
        "decision": decision["decision"],
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "check_count", "failure_count", "decision")}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
