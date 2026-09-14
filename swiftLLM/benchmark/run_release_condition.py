"""Run one preregistered release-side v10 condition."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from typing import Any

import swiftllm
from transformers import AutoTokenizer

from .closed_loop_controller import FP16_POLICY_REFERENCE_BLOCKS
from .release_intent_controller import (
    PRIMARY_RELEASE_POLICY,
    ReleaseIntentController,
)
from .metrics import derive_request_metrics
from .run import NS_PER_SECOND, _git_info, _gpu_memory, _gpu_metadata, _safe_model_path, _utc_now
from .summarize import summarize_run


CONDITIONS = (
    "runtime_static_fp16",
    "runtime_static_awq_w4_16",
    "closed_loop_dynamic",
)
RUNTIME_FP16_BLOCKS = 1759
RUNTIME_AWQ_BLOCKS = 4170
SOURCE_PATHS = (
    "swiftLLM/swiftllm/engine_config.py",
    "swiftLLM/swiftllm/server/engine.py",
    "swiftLLM/swiftllm/server/scheduler.py",
    "swiftLLM/swiftllm/server/structs.py",
    "swiftLLM/swiftllm/worker/block_manager.py",
    "swiftLLM/swiftllm/worker/kernels/kvcache_mgmt.py",
    "swiftLLM/swiftllm/worker/kernels/paged_attn.py",
    "swiftLLM/swiftllm/worker/layers/transformer_layer.py",
    "swiftLLM/swiftllm/worker/model.py",
    "swiftLLM/swiftllm/worker/weight.py",
    "swiftLLM/benchmark/metrics.py",
    "swiftLLM/benchmark/run.py",
    "swiftLLM/benchmark/analyze_crossover.py",
    "swiftLLM/benchmark/analyze_static_frontier.py",
    "swiftLLM/benchmark/analyze_closed_loop.py",
    "swiftLLM/benchmark/closed_loop_controller.py",
    "swiftLLM/benchmark/release_intent_controller.py",
    "swiftLLM/benchmark/run_release_condition.py",
    "swiftLLM/benchmark/prepare_release_workloads.py",
    "swiftLLM/benchmark/run_release_plan.py",
    "swiftLLM/benchmark/analyze_release_latency.py",
    "swiftLLM/benchmark/analyze_release_side.py",
    "swiftLLM/benchmark/audit_release_side.py",
    "swiftLLM/benchmark/test_release_intent_controller.py",
    "swiftLLM/benchmark/test_release_runtime_telemetry.py",
    "swiftLLM/benchmark/test_release_workloads.py",
    "swiftLLM/benchmark/test_release_analysis.py",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def make_gpu_environment_reader():
    """Return a non-failing NVML reader for clock/thermal/power evidence."""
    try:
        import pynvml

        pynvml.nvmlInit()
        physical_index = int((os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0])
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_index)

        def read_environment() -> dict[str, Any]:
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                throttle = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
                return {
                    "gpu_environment_source": "pynvml",
                    "gpu_physical_index": physical_index,
                    "gpu_sm_clock_mhz": int(
                        pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                    ),
                    "gpu_memory_clock_mhz": int(
                        pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                    ),
                    "gpu_temperature_c": int(
                        pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                    ),
                    "gpu_power_w": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
                    "gpu_power_limit_w": pynvml.nvmlDeviceGetPowerManagementLimit(handle)
                    / 1000.0,
                    "gpu_utilization_percent": int(utilization.gpu),
                    "gpu_memory_utilization_percent": int(utilization.memory),
                    "gpu_performance_state": int(pynvml.nvmlDeviceGetPerformanceState(handle)),
                    "gpu_fan_percent": int(pynvml.nvmlDeviceGetFanSpeed(handle)),
                    "gpu_clock_throttle_reasons_mask": int(throttle),
                }
            except Exception as exc:
                return {
                    "gpu_environment_source": "unavailable",
                    "gpu_environment_error": f"{type(exc).__name__}: {exc}",
                }

        return read_environment
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return lambda: {
            "gpu_environment_source": "unavailable",
            "gpu_environment_error": error,
        }


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--quantized-model-path", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--planned-run-id", required=True)
    parser.add_argument("--attempt-index", type=int, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--telemetry-interval-s", type=float, default=0.25)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--warmup-output-token-count", type=int, default=8)
    parser.add_argument("--timeout-s", type=float, default=7200.0)


def make_config(args: argparse.Namespace) -> swiftllm.EngineConfig:
    return swiftllm.EngineConfig(
        model_path=str(args.model_path),
        use_dummy=False,
        block_size=16,
        gpu_mem_utilization=0.99,
        num_cpu_blocks=4096,
        max_seqs_in_block_table=128,
        max_blocks_per_seq=3072,
        max_batch_size=32,
        max_tokens_in_batch=49152,
        quantized_layer_count=0,
        quantization_backend="awq_marlin",
        quantized_model_path=str(args.quantized_model_path),
        enable_runtime_morphing=True,
        runtime_awq_target_blocks=RUNTIME_AWQ_BLOCKS,
        runtime_verify_kv=True,
    )


def source_provenance() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in SOURCE_PATHS
    }


def validate_preregistration(args: argparse.Namespace) -> None:
    repo = Path(__file__).resolve().parents[2]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("artifact_version") != "release_side_runtime_v10":
        raise RuntimeError("wrong release-side protocol manifest")
    if source_provenance() != manifest.get("source_sha256_at_preregistration"):
        raise RuntimeError("runtime/source hashes differ from the preregistered manifest")
    protocol = repo / manifest["repository"]["protocol_path"]
    if sha256_file(protocol) != manifest["repository"]["protocol_sha256"]:
        raise RuntimeError("protocol bytes differ from the preregistered manifest")
    workload_hash = sha256_file(args.workload)
    if workload_hash not in {
        record["sha256"] for record in manifest["workloads"]["files"].values()
    }:
        raise RuntimeError("workload is not one of the preregistered v10 inputs")
    _, dirty, _, status = _git_info()
    if dirty:
        raise RuntimeError(f"release-side runs require a clean Git tree:\n{status}")


def validate_workload(workload: list[dict[str, Any]]) -> None:
    if len(workload) not in (64, 192, 320):
        raise ValueError(
            f"frozen release workload must contain 64, 192, or 320 requests, got {len(workload)}"
        )
    if len({str(row.get("request_id")) for row in workload}) != len(workload):
        raise ValueError("workload request IDs must be unique")
    previous = -1.0
    for expected_sequence, row in enumerate(workload):
        if int(row.get("sequence", -1)) != expected_sequence:
            raise ValueError("workload sequence is not contiguous")
        if int(row.get("prompt_token_count", -1)) != 1024:
            raise ValueError(f"request {row.get('request_id')} is not exactly 1024 prompt tokens")
        if int(row.get("requested_output_token_count", -1)) != 512:
            raise ValueError(f"request {row.get('request_id')} does not request exactly 512 outputs")
        offset = float(row.get("planned_arrival_offset_s", -1))
        if offset < previous:
            raise ValueError("workload offsets must be nondecreasing")
        previous = offset


def compressed_precision_transitions(states: list[str], positions: list[int | None]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, state in enumerate(states):
        if not result or result[-1]["precision_state"] != state:
            result.append(
                {
                    "output_step_index": index,
                    "input_position": positions[index],
                    "precision_state": state,
                }
            )
    return result


async def run(args: argparse.Namespace) -> Path:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "5":
        raise RuntimeError(
            "the frozen release-side v10 protocol requires CUDA_VISIBLE_DEVICES=5"
        )
    if args.telemetry_interval_s != 0.25:
        raise ValueError("the frozen controller/telemetry cadence is exactly 0.25 seconds")
    if args.seed != 2025 or args.warmup_requests != 1 or args.warmup_output_token_count != 8:
        raise ValueError("the frozen seed and one-request/eight-token warmup cannot be changed")

    validate_preregistration(args)
    workload = read_jsonl(args.workload)
    validate_workload(workload)
    run_dir = args.output_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    commit, dirty, diff_sha256, git_status = _git_info()
    config = make_config(args)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "release_side_runtime_v10",
        "run_id": args.run_id,
        "planned_run_id": args.planned_run_id,
        "attempt_index": args.attempt_index,
        "condition": args.condition,
        "workload_class": workload[0].get("v10_workload_class"),
        "created_at_utc": _utc_now(),
        "current_morphserve_git_commit": commit,
        "current_morphserve_git_dirty": dirty,
        "current_morphserve_git_diff_sha256": diff_sha256,
        "current_morphserve_git_status_porcelain": git_status,
        "runtime_source_files_sha256": source_provenance(),
        "model_path_safe": _safe_model_path(str(args.model_path)),
        "model_path_sha256": hashlib.sha256(str(args.model_path.resolve()).encode()).hexdigest(),
        "quantized_model_path_safe": _safe_model_path(str(args.quantized_model_path)),
        "quantized_model_config_sha256": sha256_file(args.quantized_model_path / "config.json"),
        "quantized_model_index_sha256": sha256_file(args.quantized_model_path / "model.safetensors.index.json"),
        "gpu": _gpu_metadata(),
        "cuda_device_requested": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "workload_path": str(args.workload),
        "workload_sha256": sha256_file(args.workload),
        "protocol_manifest": str(args.manifest),
        "protocol_manifest_sha256": sha256_file(args.manifest),
        "request_count_requested": len(workload),
        "prompt_token_count_requested": 1024,
        "output_token_count_requested": 512,
        "random_seed": args.seed,
        "telemetry_interval_s": args.telemetry_interval_s,
        "warmup_policy": {
            "warmup_requests": 1,
            "output_token_count": 8,
            "included_in_measurement": False,
        },
        "arrival_mode": "trace_open_loop",
        "arrival_lateness_tolerance_s": 0.25,
        "planned_schedule_last_offset_s": float(workload[-1]["planned_arrival_offset_s"]),
        "engine_config_requested": asdict(config),
        "runtime_envelope": {
            "initial_fp16_blocks_expected": RUNTIME_FP16_BLOCKS,
            "safe_dynamic_awq_blocks": RUNTIME_AWQ_BLOCKS,
            "policy_reference_fp16_blocks": FP16_POLICY_REFERENCE_BLOCKS,
            "policy_reference_is_not_physical_safety_capacity": True,
        },
        "controller": {
            "enabled": args.condition == "closed_loop_dynamic",
            "entry_policy": "sustained_compound_pressure",
            "release_policy": PRIMARY_RELEASE_POLICY,
            "input_exclusions": [
                "workload scale or nominal RPS",
                "future arrivals",
                "request completion outcomes",
                "final TTFT",
                "DuReader answers/F1",
                "offline crossover labels",
            ],
        },
        "generation": {
            "decoding": "greedy_argmax",
            "do_sample": False,
            "eos_stopping": "exactly 512 output steps",
        },
        "raw_files": [
            "metadata.json",
            "requests.jsonl",
            "telemetry.jsonl",
            "batches.jsonl",
            "controller.jsonl",
            "gpu_environment.jsonl",
            "transitions.jsonl",
            "initialization_transitions.jsonl",
            "runtime_preparation.json",
        ],
        "timestamp_definition": {
            "clock": "time.perf_counter_ns monotonic process clock",
            "arrival_time_ns": "actual open-loop task launch before Engine.add_request_and_stream",
            "transition_requested_ns": "controller latch time",
            "model_transition_started_ns": "v8 synchronized hot-path trace start after any pending drain",
            "model_transition_ended_ns": "v8 hot-path completion",
            "release_intent_first_true_ns": "causal controller sample timestamp",
            "admission_pause_and_drain_timestamps": "observation-only Engine restore lifecycle",
        },
        "metric_formula": {
            "ttft_s": "(first_stream_token_received_time_ns-arrival_time_ns)/1e9",
            "queueing_delay_s": "(first_prefill_time_ns-scheduler_eligible_time_ns)/1e9",
            "first_prefill_to_first_output_s": "(first_output_token_time_ns-first_prefill_time_ns)/1e9",
            "strict_slo_violation": "finite ttft_s > 2.0",
        },
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    request_rows: dict[str, dict[str, Any]] = {}
    telemetry_rows: list[dict[str, Any]] = []
    controller_rows: list[dict[str, Any]] = []
    gpu_environment_rows: list[dict[str, Any]] = []
    counters = {"launched": 0, "completed": 0, "generated_tokens": 0}
    initialization_transitions: list[dict[str, Any]] = []
    controller: ReleaseIntentController | None = None
    transition_tasks: list[asyncio.Task[None]] = []
    request_tasks: list[asyncio.Task[None]] = []
    engine_task: asyncio.Task[Any] | None = None
    telemetry_task: asyncio.Task[Any] | None = None
    stop_telemetry = asyncio.Event()
    requests_done = asyncio.Event()
    engine: swiftllm.Engine | None = None
    measurement_start_ns: int | None = None

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        engine = swiftllm.Engine(config)
        await engine.initialize()
        initial_snapshot = engine.get_benchmark_snapshot()
        if initial_snapshot["num_gpu_blocks"] != RUNTIME_FP16_BLOCKS:
            raise RuntimeError(
                f"runtime FP16 capacity {initial_snapshot['num_gpu_blocks']} != frozen {RUNTIME_FP16_BLOCKS}"
            )
        metadata["engine_config_actual"] = asdict(config)
        metadata["vocab_size"] = int(engine.model.model_config.vocab_size)
        metadata["initial_engine_snapshot"] = initial_snapshot
        metadata["runtime_preparation_elapsed_s"] = (
            engine.model.runtime_preparation_trace.get("elapsed_ns", 0) / NS_PER_SECOND
        )
        (run_dir / "runtime_preparation.json").write_text(
            json.dumps(engine.model.runtime_preparation_trace, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if args.condition == "runtime_static_awq_w4_16":
            setup_requested_ns = time.perf_counter_ns()
            setup_trace = await engine.morph_to_awq_w4_16()
            setup_trace = dict(setup_trace)
            setup_trace["initialization_only"] = True
            setup_trace["controller_requested_ns"] = setup_requested_ns
            initialization_transitions.append(setup_trace)
            if engine.get_benchmark_snapshot()["num_gpu_blocks"] != RUNTIME_AWQ_BLOCKS:
                raise RuntimeError("pre-measurement AWQ morph did not publish 4170 blocks")

        engine_task = asyncio.create_task(engine.start_all_event_loops())
        await asyncio.sleep(0)
        for warmup_index in range(args.warmup_requests):
            warmup = swiftllm.RawRequest(
                str(workload[warmup_index % len(workload)]["prompt"]),
                args.warmup_output_token_count,
            )
            async for _ in engine.add_request_and_stream(warmup):
                pass
            print(f"warmup completed: {warmup_index + 1}/{args.warmup_requests}", flush=True)

        engine.start_benchmark_batch_observation()
        loop = asyncio.get_running_loop()
        measurement_start_loop = loop.time()
        measurement_start_ns = time.perf_counter_ns()
        metadata["measurement_start_time_ns"] = measurement_start_ns
        metadata["measurement_start_utc"] = datetime.now(timezone.utc).isoformat()
        baseline = engine.get_benchmark_snapshot()
        baseline_swap_in = baseline["swap_in_count"]
        baseline_swap_out = baseline["swap_out_count"]
        if args.condition == "closed_loop_dynamic":
            controller = ReleaseIntentController(measurement_start_ns)
        read_gpu_environment = make_gpu_environment_reader()

        async def execute_transition(direction: str, transition_id: int) -> None:
            assert engine is not None and controller is not None
            controller.mark_api_started(transition_id, time.perf_counter_ns())
            try:
                trace = (
                    await engine.morph_to_awq_w4_16()
                    if direction == "FP16_TO_AWQ_MARLIN_W4_16"
                    else await engine.restore_to_fp16()
                )
            except BaseException as exc:
                controller.fail_transition(transition_id, exc, time.perf_counter_ns())
            else:
                controller.complete_transition(transition_id, trace, time.perf_counter_ns())

        async def sample_telemetry() -> None:
            assert engine is not None and measurement_start_ns is not None
            tick = 0
            while True:
                deadline = measurement_start_loop + tick * args.telemetry_interval_s
                delay = deadline - loop.time()
                if delay > 0:
                    try:
                        await asyncio.wait_for(stop_telemetry.wait(), timeout=delay)
                    except asyncio.TimeoutError:
                        pass
                if stop_telemetry.is_set():
                    return
                now_ns = time.perf_counter_ns()
                snapshot = engine.get_benchmark_snapshot()
                manager = engine.model.gpu_block_manager
                physical_total = int(manager.num_blocks)
                physical_free = int(manager.num_free_blocks)
                physical_used = physical_total - physical_free
                elapsed_s = (now_ns - measurement_start_ns) / NS_PER_SECOND
                preemptions = int(snapshot["swap_out_count"] - baseline_swap_out)
                row: dict[str, Any] = {
                    "schema_version": 1,
                    "condition": args.condition,
                    "timestamp_ns": now_ns,
                    "elapsed_s": elapsed_s,
                    "sample_index": tick,
                    "scheduled_elapsed_s": tick * args.telemetry_interval_s,
                    "sampling_jitter_s": elapsed_s - tick * args.telemetry_interval_s,
                    "runtime_precision_state": snapshot["precision_state"],
                    "pending_transition_target": snapshot["pending_transition_target"],
                    "waiting_q_depth": snapshot["waiting_q_depth"],
                    "running_q_count": snapshot["running_q_count"],
                    "swapped_q_count": snapshot["swapped_q_count"],
                    "admissions_paused": snapshot["admissions_paused"],
                    "physical_used_kv_blocks": physical_used,
                    "physical_free_kv_blocks": physical_free,
                    "physical_total_kv_blocks": physical_total,
                    "scheduler_used_kv_blocks": snapshot["num_decoding_gpu_blocks"],
                    "scheduler_visible_kv_blocks": snapshot["num_gpu_blocks"],
                    "base_physical_kv_blocks": snapshot["base_gpu_blocks"],
                    "extension_physical_kv_blocks": snapshot["extension_gpu_blocks"],
                    "fp16_policy_reference_blocks": FP16_POLICY_REFERENCE_BLOCKS,
                    "fp16_equivalent_kv_utilization": (
                        snapshot["num_decoding_gpu_blocks"] / FP16_POLICY_REFERENCE_BLOCKS
                    ),
                    "native_current_state_kv_utilization": (
                        physical_used / physical_total if physical_total else None
                    ),
                    # Backward-compatible names consumed by existing summary helpers.
                    "num_decoding_gpu_blocks": snapshot["num_decoding_gpu_blocks"],
                    "num_gpu_blocks": snapshot["num_gpu_blocks"],
                    "logical_kv_utilization": (
                        snapshot["num_decoding_gpu_blocks"] / snapshot["num_gpu_blocks"]
                        if snapshot["num_gpu_blocks"] else None
                    ),
                    "launched_request_count": counters["launched"],
                    "completed_request_count": counters["completed"],
                    "generated_output_token_count": counters["generated_tokens"],
                    "swap_in_count": int(snapshot["swap_in_count"] - baseline_swap_in),
                    "swap_out_count": preemptions,
                    "preemption_count": preemptions,
                }
                row.update(_gpu_memory())
                telemetry_rows.append(row)
                if tick % 4 == 0:
                    environment_row = {
                        "schema_version": 1,
                        "timestamp_ns": now_ns,
                        "elapsed_s": elapsed_s,
                        "sample_index": tick // 4,
                    }
                    environment_row.update(read_gpu_environment())
                    gpu_environment_rows.append(environment_row)

                if controller is not None:
                    decision = controller.evaluate(row)
                    action = decision["requested_action"]
                    if action is not None and not requests_done.is_set():
                        requested_ns = time.perf_counter_ns()
                        transition_id = controller.request_transition(action, requested_ns)
                        if action == "AWQ_MARLIN_W4_16_TO_FP16":
                            controller.transitions[transition_id - 1].update(
                                {
                                    "release_intent_condition_first_true_ns": now_ns,
                                    "release_intent_condition_first_true_elapsed_s": elapsed_s,
                                    "release_intent_observation": {
                                        "waiting_q_depth": row["waiting_q_depth"],
                                        "running_q_count": row["running_q_count"],
                                        "swapped_q_count": row["swapped_q_count"],
                                        "scheduler_used_kv_blocks": row[
                                            "scheduler_used_kv_blocks"
                                        ],
                                        "physical_used_kv_blocks": row[
                                            "physical_used_kv_blocks"
                                        ],
                                        "blocks_above_fp16_base": row[
                                            "physical_used_kv_blocks"
                                        ]
                                        - RUNTIME_FP16_BLOCKS,
                                        "fp16_reference_utilization": row[
                                            "fp16_equivalent_kv_utilization"
                                        ],
                                        "preemption_count": row["preemption_count"],
                                        "release_recent_waiting_mean": decision[
                                            "release_recent_waiting_mean"
                                        ],
                                        "release_recent_preemptions": decision[
                                            "release_recent_preemptions"
                                        ],
                                        "release_recent_utilization_mean": decision[
                                            "release_recent_fp16_reference_utilization_mean"
                                        ],
                                        "release_prior_utilization_mean": decision[
                                            "release_prior_fp16_reference_utilization_mean"
                                        ],
                                        "release_pressure_ratio": decision[
                                            "release_recent_to_prior_utilization_ratio"
                                        ],
                                    },
                                }
                            )
                        task = asyncio.create_task(execute_transition(action, transition_id))
                        transition_tasks.append(task)
                    controller_row = dict(row)
                    controller_row.update(decision)
                    controller_fields = controller.trace_fields()
                    if (
                        controller_fields["controller_state"] == "ENTRY_REQUESTED"
                        and snapshot["precision_state"] == "MORPHING_TO_AWQ"
                    ):
                        controller_fields["controller_state"] = "MORPHING_TO_AWQ"
                    controller_row.update(controller_fields)
                    controller_rows.append(controller_row)
                tick += 1

        telemetry_task = asyncio.create_task(sample_telemetry())

        async def consume(item: dict[str, Any], planned_time_ns: int) -> None:
            assert engine is not None and measurement_start_ns is not None
            request_id = str(item["request_id"])
            actual_arrival_ns = time.perf_counter_ns()
            row: dict[str, Any] = {
                "schema_version": 3,
                "benchmark_request_id": request_id,
                "sequence": int(item["sequence"]),
                "phase": item.get("v10_phase"),
                "phase_index": item.get("v10_phase_index"),
                "status": "launched",
                "planned_arrival_offset_s": float(item["planned_arrival_offset_s"]),
                "planned_arrival_time_ns": planned_time_ns,
                "actual_arrival_offset_s": (actual_arrival_ns - measurement_start_ns) / NS_PER_SECOND,
                "arrival_jitter_s": (actual_arrival_ns - planned_time_ns) / NS_PER_SECOND,
                "arrival_time_ns": actual_arrival_ns,
                "source_trace_timestamp_s": item.get("source_trace_timestamp_s"),
                "dataset_question_id": item.get("dataset_question_id"),
                "dataset_reference_answers": item.get("dataset_reference_answers"),
                "prompt_token_count_requested": 1024,
                "requested_output_token_count": 512,
                "prompt_token_count": None,
                "output_token_count": 0,
                "output_token_ids": [],
                "generated_answer": None,
                "engine_request_ids": [],
                "step_precision_states": [],
                "step_input_positions": [],
                "step_received_time_ns": [],
                "precision_state_transitions": [],
                "prefill_precision_state": None,
                "fp16_output_step_count": 0,
                "awq_output_step_count": 0,
                "awq_output_step_fraction": None,
                "spans_precision_states": False,
                "spans_hot_transition": False,
                "arrival_during_hot_transition": False,
                "arrival_transition_ids": [],
                "first_prefill_time_ns": None,
                "scheduler_eligible_time_ns": None,
                "first_output_token_time_ns": None,
                "first_stream_token_received_time_ns": None,
                "completion_time_ns": None,
                "stream_completion_time_ns": None,
                "error": None,
            }
            request_rows[request_id] = row
            counters["launched"] += 1
            raw = swiftllm.RawRequest(
                str(item["prompt"]),
                512,
                benchmark_request_id=request_id,
                benchmark_arrival_time_ns=actual_arrival_ns,
            )
            try:
                async for step in engine.add_request_and_stream(raw):
                    received_ns = time.perf_counter_ns()
                    counters["generated_tokens"] += 1
                    if row["first_stream_token_received_time_ns"] is None:
                        row["first_stream_token_received_time_ns"] = received_ns
                    if step.request.prompt_len:
                        row["prompt_token_count"] = step.request.prompt_len
                    row["output_token_ids"] = list(step.request.output_token_ids)
                    row["output_token_count"] = len(row["output_token_ids"])
                    row["engine_request_ids"].append(step.request.request_id)
                    row["step_precision_states"].append(str(step.precision_state))
                    row["step_input_positions"].append(step.input_position)
                    row["step_received_time_ns"].append(received_ns)
                    if step.request.is_finished():
                        row["stream_completion_time_ns"] = received_ns
                        row["status"] = "completed"
                        row["first_prefill_time_ns"] = step.request.benchmark_first_prefill_time_ns
                        row["scheduler_eligible_time_ns"] = step.request.benchmark_scheduler_eligible_time_ns
                        row["first_output_token_time_ns"] = step.request.benchmark_first_output_token_time_ns
                        row["completion_time_ns"] = step.request.benchmark_completion_time_ns
                        row["generated_answer"] = tokenizer.decode(
                            row["output_token_ids"], skip_special_tokens=True
                        )
                        counters["completed"] += 1
                        break
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = f"{type(exc).__name__}: {exc}"
            states = list(row["step_precision_states"])
            positions = list(row["step_input_positions"])
            row["precision_state_transitions"] = compressed_precision_transitions(states, positions)
            row["prefill_precision_state"] = states[0] if states else None
            row["fp16_output_step_count"] = states.count("FP16")
            row["awq_output_step_count"] = states.count("AWQ_MARLIN_W4_16")
            row["awq_output_step_fraction"] = (
                row["awq_output_step_count"] / len(states) if states else None
            )
            row["spans_precision_states"] = len(set(states)) > 1
            row.update(derive_request_metrics(row))
            if row.get("first_output_token_time_ns") is not None and row.get("first_prefill_time_ns") is not None:
                row["first_prefill_to_first_output_s"] = (
                    int(row["first_output_token_time_ns"]) - int(row["first_prefill_time_ns"])
                ) / NS_PER_SECOND
            else:
                row["first_prefill_to_first_output_s"] = None

        async def launch() -> None:
            assert measurement_start_ns is not None
            for item in workload:
                offset = float(item["planned_arrival_offset_s"])
                planned_ns = measurement_start_ns + int(offset * NS_PER_SECOND)
                delay = measurement_start_loop + offset - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                request_tasks.append(asyncio.create_task(consume(item, planned_ns)))
            await asyncio.gather(*request_tasks)

        await asyncio.wait_for(launch(), timeout=args.timeout_s)
        requests_done.set()
        if transition_tasks:
            await asyncio.gather(*transition_tasks)
        if controller is not None and controller.error:
            raise RuntimeError(f"controller transition failed: {controller.error}")
    except Exception as exc:
        metadata["run_exception"] = f"{type(exc).__name__}: {exc}"
        metadata["run_traceback"] = traceback.format_exc()
        (run_dir / "failure.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    finally:
        requests_done.set()
        stop_telemetry.set()
        if telemetry_task is not None:
            await asyncio.gather(telemetry_task, return_exceptions=True)
        for task in request_tasks + transition_tasks:
            if not task.done():
                task.cancel()
        if request_tasks or transition_tasks:
            await asyncio.gather(*(request_tasks + transition_tasks), return_exceptions=True)
        if engine_task is not None:
            engine_task.cancel()
            await asyncio.gather(engine_task, return_exceptions=True)

        transitions = controller.transitions if controller is not None else []
        for row in request_rows.values():
            arrival = row.get("arrival_time_ns")
            completion = row.get("completion_time_ns")
            for event in transitions:
                start = event.get("model_transition_started_ns")
                end = event.get("model_transition_ended_ns")
                if start is None or end is None or arrival is None:
                    continue
                if int(start) <= int(arrival) <= int(end):
                    row["arrival_during_hot_transition"] = True
                    row["arrival_transition_ids"].append(event["controller_transition_id"])
                if completion is not None and max(int(arrival), int(start)) <= min(int(completion), int(end)):
                    row["spans_hot_transition"] = True
            row["engine_request_id_stable"] = len(set(row.get("engine_request_ids", []))) <= 1
            expected_positions = list(range(1023, 1023 + int(row.get("output_token_count", 0))))
            row["input_positions_contiguous"] = row.get("step_input_positions") == expected_positions

        # Backfill eventual transition timestamps/results into every sample that
        # referred to that latched transition while it was in progress.
        for row in controller_rows:
            transition_id = row.get("active_transition_id")
            if transition_id is None:
                continue
            event = transitions[int(transition_id) - 1]
            row["transition_start_timestamp_ns"] = event.get("model_transition_started_ns")
            row["transition_end_timestamp_ns"] = event.get("model_transition_ended_ns")
            row["transition_pending_or_drain_duration_s"] = event.get("pending_or_drain_duration_s")
            row["transition_result"] = event.get("result")
            row["transition_error"] = event.get("error")

        end_ns = time.perf_counter_ns()
        metadata["measurement_end_time_ns"] = end_ns
        metadata["measurement_end_utc"] = _utc_now()
        if measurement_start_ns is not None:
            metadata["measurement_duration_s"] = (end_ns - measurement_start_ns) / NS_PER_SECOND
        metadata["launched_request_count"] = counters["launched"]
        metadata["completed_request_count"] = counters["completed"]
        metadata["generated_output_token_count"] = counters["generated_tokens"]
        metadata["observed_forward_batch_count"] = (
            len(engine.get_benchmark_batch_events()) if engine is not None else 0
        )
        metadata["measured_transition_count"] = len(transitions)
        metadata["final_controller_state"] = controller.state if controller is not None else None
        metadata["final_engine_snapshot"] = (
            engine.get_benchmark_snapshot() if engine is not None and engine.initialized else None
        )
        metadata["final_gpu_memory"] = _gpu_memory()
        metadata["completed_at_utc"] = _utc_now()
        (run_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_jsonl(
            run_dir / "requests.jsonl",
            sorted(request_rows.values(), key=lambda row: int(row["sequence"])),
        )
        write_jsonl(run_dir / "telemetry.jsonl", telemetry_rows)
        write_jsonl(
            run_dir / "batches.jsonl",
            engine.get_benchmark_batch_events() if engine is not None else [],
        )
        write_jsonl(run_dir / "controller.jsonl", controller_rows)
        write_jsonl(run_dir / "gpu_environment.jsonl", gpu_environment_rows)
        write_jsonl(run_dir / "transitions.jsonl", transitions)
        write_jsonl(run_dir / "initialization_transitions.jsonl", initialization_transitions)
        try:
            summary = summarize_run(run_dir)
            summary["condition"] = args.condition
            summary["workload_class"] = metadata.get("workload_class")
            (run_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except Exception:
            (run_dir / "summary_failure.txt").write_text(traceback.format_exc(), encoding="utf-8")

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    args = parser.parse_args()
    path = asyncio.run(run(args))
    print(f"raw release-side v10 run: {path}")


if __name__ == "__main__":
    main()
