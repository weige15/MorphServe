"""Run one static quantization condition against a frozen workload manifest."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback
from typing import Any
import time

import swiftllm
from transformers import AutoTokenizer

from .metrics import derive_request_metrics
from .run import NS_PER_SECOND, _git_info, _gpu_memory, _gpu_metadata, _safe_model_path, _utc_now
from .summarize import summarize_run


QUANTIZATION_LABEL = "uniform_w4_nf4_bitsandbytes_proxy"
LAYER_ORDER_LABEL = "lis_order_unavailable_front_to_back"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--condition",
        choices=(
            "fp16_0", "w4_8", "w4_16", "w4_32",
            "awq_w4_8", "awq_w4_16", "awq_w4_32",
        ),
        required=True,
    )
    parser.add_argument("--quantized-layer-count", type=int, choices=(0, 8, 16, 32), required=True)
    parser.add_argument(
        "--quantization-backend",
        choices=("nf4_bitsandbytes", "awq_marlin"),
        default="nf4_bitsandbytes",
    )
    parser.add_argument("--quantized-model-path", type=Path)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--launch-mode",
        choices=("trace_open_loop", "sequential"),
        default="trace_open_loop",
        help="Use saved arrival offsets or wait for each request to complete (quality-only runs).",
    )
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--warmup-output-token-count", type=int, default=8)
    parser.add_argument("--telemetry-interval-s", type=float, default=0.25)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--max-tokens-in-batch", type=int, default=49152)
    parser.add_argument("--num-cpu-blocks", type=int, default=4096)
    parser.add_argument("--gpu-mem-utilization", type=float, default=0.99)
    parser.add_argument("--timeout-s", type=float, default=3600.0)


def make_config(args: argparse.Namespace) -> swiftllm.EngineConfig:
    return swiftllm.EngineConfig(
        model_path=str(args.model_path),
        use_dummy=False,
        block_size=16,
        gpu_mem_utilization=args.gpu_mem_utilization,
        num_cpu_blocks=args.num_cpu_blocks,
        max_seqs_in_block_table=128,
        max_blocks_per_seq=3072,
        max_batch_size=args.max_batch_size,
        max_tokens_in_batch=args.max_tokens_in_batch,
        quantized_layer_count=args.quantized_layer_count,
        quantization_backend=args.quantization_backend,
        quantized_model_path=(str(args.quantized_model_path) if args.quantized_model_path else None),
    )


def initial_metadata(args: argparse.Namespace, workload: list[dict[str, Any]]) -> dict[str, Any]:
    commit, dirty, diff_sha256, git_status = _git_info()
    return {
        "schema_version": 2,
        "run_id": args.run_id,
        "condition": args.condition,
        "quantized_layer_count": args.quantized_layer_count,
        "quantization_label": (
            "fp16_control"
            if args.quantized_layer_count == 0
            else ("awq_marlin_w4_g128_zp" if args.quantization_backend == "awq_marlin" else QUANTIZATION_LABEL)
        ),
        "quantization_backend": args.quantization_backend,
        "layer_order_label": LAYER_ORDER_LABEL,
        "layer_order": list(range(args.quantized_layer_count)),
        "created_at_utc": _utc_now(),
        "model_path_safe": _safe_model_path(str(args.model_path)),
        "model_path_sha256": hashlib.sha256(str(args.model_path.resolve()).encode()).hexdigest(),
        "quantized_model_path_safe": (
            _safe_model_path(str(args.quantized_model_path))
            if args.quantized_model_path else None
        ),
        "current_morphserve_git_commit": commit,
        "current_morphserve_git_dirty": dirty,
        "current_morphserve_git_diff_sha256": diff_sha256,
        "current_morphserve_git_status_porcelain": git_status,
        "gpu": _gpu_metadata(),
        "cuda_device_requested": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        "workload_path": str(args.workload),
        "workload_sha256": sha256_file(args.workload),
        "protocol_manifest": str(args.manifest),
        "protocol_manifest_sha256": sha256_file(args.manifest),
        "request_count_requested": len(workload),
        "prompt_token_count_requested": 1024,
        "output_token_count_requested": 512,
        "random_seed": args.seed,
        "warmup_policy": {
            "warmup_requests": args.warmup_requests,
            "output_token_count": args.warmup_output_token_count,
            "included_in_raw_requests": False,
        },
        "arrival_mode": args.launch_mode,
        "workload_kind": workload[0].get("frontier_workload_kind"),
        "workload_time_scale": workload[0].get("frontier_time_scale"),
        "nominal_offered_rps": workload[0].get("frontier_nominal_offered_rps"),
        "telemetry_interval_s": args.telemetry_interval_s,
        "generation": {
            "decoding": "greedy_argmax",
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "eos_stopping": "not supported by unchanged SwiftLLM; exactly 512 output steps",
        },
        "timestamp_definition": {
            "clock": "time.perf_counter_ns monotonic process clock",
            "arrival_time_ns": "immediately before Engine.add_request_and_stream",
            "scheduler_eligible_time_ns": "after tokenization before scheduler arrival",
            "first_prefill_time_ns": "immediately before first model forward containing request",
            "first_output_token_time_ns": "after model forward returned first output",
            "first_stream_token_received_time_ns": "benchmark consumer receipt time",
            "stream_completion_time_ns": "benchmark consumer receipt of final output",
        },
        "metric_formula": {
            "ttft_s": "(first_stream_token_received_time_ns - arrival_time_ns) / 1e9",
            "queueing_delay_s": "(first_prefill_time_ns - scheduler_eligible_time_ns) / 1e9",
            "slo_violation": "finite ttft_s > 2.0",
        },
        "engine_config_requested": asdict(make_config(args)),
        "raw_files": ["metadata.json", "requests.jsonl", "telemetry.jsonl"],
    }


async def run(args: argparse.Namespace) -> Path:
    workload = read_jsonl(args.workload)
    if not workload:
        raise ValueError("empty workload")
    expected_layers = {
        "fp16_0": 0,
        "w4_8": 8,
        "w4_16": 16,
        "w4_32": 32,
        "awq_w4_8": 8,
        "awq_w4_16": 16,
        "awq_w4_32": 32,
    }
    if args.quantized_layer_count != expected_layers[args.condition]:
        raise ValueError("quantized layer count does not match the named condition")
    if args.condition.startswith("awq_") != (args.quantization_backend == "awq_marlin"):
        raise ValueError("AWQ condition and quantization backend must agree")
    if args.quantization_backend == "awq_marlin" and args.quantized_model_path is None:
        raise ValueError("AWQ-Marlin requires --quantized-model-path")
    if args.telemetry_interval_s <= 0 or args.timeout_s <= 0:
        raise ValueError("telemetry interval and timeout must be positive")
    if args.warmup_requests < 0 or args.warmup_output_token_count <= 0:
        raise ValueError("warmup request count must be non-negative and output count positive")
    for expected_sequence, row in enumerate(workload):
        if row.get("sequence") != expected_sequence:
            raise ValueError("workload sequence is not contiguous")
        if int(row.get("prompt_token_count", -1)) != 1024:
            raise ValueError(f"workload prompt is not 1024 tokens: {row.get('request_id')}")
        if int(row.get("requested_output_token_count", -1)) != 512:
            raise ValueError(f"workload output is not 512 tokens: {row.get('request_id')}")

    run_dir = args.output_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = initial_metadata(args, workload)
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    request_rows: dict[str, dict[str, Any]] = {}
    telemetry_rows: list[dict[str, Any]] = []
    counters = {"launched": 0, "completed": 0, "generated_tokens": 0}
    engine_task: asyncio.Task[Any] | None = None
    telemetry_task: asyncio.Task[Any] | None = None
    request_tasks: list[asyncio.Task[Any]] = []
    stop_telemetry = asyncio.Event()
    engine: swiftllm.Engine | None = None

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        config = make_config(args)
        engine = swiftllm.Engine(config)
        await engine.initialize()
        initial_snapshot = engine.get_benchmark_snapshot()
        metadata["engine_config_actual"] = asdict(config)
        metadata["initial_engine_snapshot"] = initial_snapshot
        metadata["num_gpu_blocks"] = initial_snapshot["num_gpu_blocks"]
        metadata["gpu_kv_token_slots"] = 16 * initial_snapshot["num_gpu_blocks"]
        if args.quantized_layer_count == 0:
            metadata["quantization_runtime"] = {
                "implementation": "unchanged SwiftLLM FP16",
                "weight_bits": 16,
            }
        elif args.quantization_backend == "awq_marlin":
            awq_config = args.quantized_model_path / "config.json"
            awq_index = args.quantized_model_path / "model.safetensors.index.json"
            metadata["quantization_runtime"] = {
                "implementation": "offline AutoAWQ GEMM layout -> one-time vLLM 0.11.2 Marlin repack -> gptq_marlin_gemm for decode and prefill",
                "quant_type": "asymmetric_uint4",
                "weight_bits": 4,
                "group_size": 128,
                "zero_point": True,
                "activation_dtype": "float16",
                "full_weight_dequantization_hot_path": False,
                "awq_config_sha256": sha256_file(awq_config),
                "awq_weight_index_sha256": sha256_file(awq_index),
                "packed_representation_bytes": engine.model.weight.quantized_representation_bytes(),
            }
        else:
            metadata["quantization_runtime"] = {
                "implementation": "bitsandbytes.functional.quantize_4bit + bitsandbytes.gemv_4bit decode + bitsandbytes.matmul_4bit prefill fallback",
                "quant_type": "nf4",
                "weight_bits": 4,
                "blocksize": 64,
                "compress_statistics": False,
                "awq_attempt": "AutoAWQ 0.2.9 source import succeeded; AWQ fused kernels/selective loading not available in this SwiftLLM engine, so all W4 conditions use the uniform NF4 proxy",
            }
        loop = asyncio.get_running_loop()
        engine_task = asyncio.create_task(engine.start_all_event_loops())
        await asyncio.sleep(0)
        for warmup_index in range(args.warmup_requests):
            warmup = swiftllm.RawRequest(
                str(workload[warmup_index % len(workload)]["prompt"]),
                args.warmup_output_token_count,
            )
            async for _ in engine.add_request_and_stream(warmup):
                pass
            print(f"warmup completed: {warmup_index + 1}/{args.warmup_requests}")
        measurement_start_loop = loop.time()
        measurement_start_ns = time.perf_counter_ns()
        metadata["measurement_start_time_ns"] = measurement_start_ns
        metadata["measurement_start_utc"] = datetime.now(timezone.utc).isoformat()
        baseline_snapshot = engine.get_benchmark_snapshot()
        baseline_swap_in = baseline_snapshot["swap_in_count"]
        baseline_swap_out = baseline_snapshot["swap_out_count"]

        async def sample_telemetry() -> None:
            while True:
                now_ns = time.perf_counter_ns()
                snapshot = engine.get_benchmark_snapshot()
                elapsed_s = (now_ns - measurement_start_ns) / NS_PER_SECOND
                row = {
                    "condition": args.condition,
                    "quantized_layer_count": args.quantized_layer_count,
                    "timestamp_ns": now_ns,
                    "elapsed_s": elapsed_s,
                    "waiting_q_depth": snapshot["waiting_q_depth"],
                    "running_q_count": snapshot["running_q_count"],
                    "swapped_q_count": snapshot["swapped_q_count"],
                    "num_decoding_gpu_blocks": snapshot["num_decoding_gpu_blocks"],
                    "num_gpu_blocks": snapshot["num_gpu_blocks"],
                    "logical_kv_utilization": snapshot["num_decoding_gpu_blocks"] / snapshot["num_gpu_blocks"] if snapshot["num_gpu_blocks"] else None,
                    "completed_request_count": counters["completed"],
                    "generated_output_token_count": counters["generated_tokens"],
                    "swap_in_count": snapshot["swap_in_count"] - baseline_swap_in,
                    "swap_out_count": snapshot["swap_out_count"] - baseline_swap_out,
                    "preemption_count": snapshot["swap_out_count"] - baseline_swap_out,
                }
                row.update(_gpu_memory())
                telemetry_rows.append(row)
                if stop_telemetry.is_set():
                    return
                try:
                    await asyncio.wait_for(stop_telemetry.wait(), timeout=args.telemetry_interval_s)
                except asyncio.TimeoutError:
                    pass

        telemetry_task = asyncio.create_task(sample_telemetry())

        async def consume(item: dict[str, Any], planned_time_ns: int) -> None:
            request_id = str(item["request_id"])
            actual_arrival_ns = time.perf_counter_ns()
            row: dict[str, Any] = {
                "schema_version": 2,
                "benchmark_request_id": request_id,
                "sequence": item["sequence"],
                "status": "launched",
                "planned_arrival_offset_s": item["planned_arrival_offset_s"],
                "source_trace_timestamp_s": item["source_trace_timestamp_s"],
                "dataset_question_id": item["dataset_question_id"],
                "dataset_reference_answers": item["dataset_reference_answers"],
                "prompt_token_count_requested": 1024,
                "requested_output_token_count": 512,
                "actual_arrival_offset_s": (actual_arrival_ns - measurement_start_ns) / NS_PER_SECOND,
                "arrival_jitter_s": (actual_arrival_ns - planned_time_ns) / NS_PER_SECOND,
                "arrival_time_ns": actual_arrival_ns,
                "prompt_token_count": None,
                "output_token_count": 0,
                "output_token_ids": [],
                "generated_answer": None,
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
            raw_request = swiftllm.RawRequest(
                str(item["prompt"]),
                512,
                benchmark_request_id=request_id,
                benchmark_arrival_time_ns=actual_arrival_ns,
            )
            try:
                async for step_output in engine.add_request_and_stream(raw_request):
                    received_ns = time.perf_counter_ns()
                    counters["generated_tokens"] += 1
                    if row["first_stream_token_received_time_ns"] is None:
                        row["first_stream_token_received_time_ns"] = received_ns
                    if step_output.request.prompt_len:
                        row["prompt_token_count"] = step_output.request.prompt_len
                    row["output_token_ids"] = list(step_output.request.output_token_ids)
                    row["output_token_count"] = len(row["output_token_ids"])
                    if step_output.request.is_finished():
                        row["stream_completion_time_ns"] = received_ns
                        row["status"] = "completed"
                        row["first_prefill_time_ns"] = step_output.request.benchmark_first_prefill_time_ns
                        row["scheduler_eligible_time_ns"] = step_output.request.benchmark_scheduler_eligible_time_ns
                        row["first_output_token_time_ns"] = step_output.request.benchmark_first_output_token_time_ns
                        row["completion_time_ns"] = step_output.request.benchmark_completion_time_ns
                        row["generated_answer"] = tokenizer.decode(row["output_token_ids"], skip_special_tokens=True)
                        counters["completed"] += 1
                        break
            except Exception as exc:  # preserve failed request evidence
                row["status"] = "failed"
                row["error"] = f"{type(exc).__name__}: {exc}"
            row.update(derive_request_metrics(row))

        async def launch() -> None:
            if args.launch_mode == "sequential":
                for item in workload:
                    await consume(item, time.perf_counter_ns())
                return
            for item in workload:
                planned_time_ns = measurement_start_ns + int(float(item["planned_arrival_offset_s"]) * NS_PER_SECOND)
                delay = measurement_start_loop + float(item["planned_arrival_offset_s"]) - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                request_tasks.append(asyncio.create_task(consume(item, planned_time_ns)))
            if request_tasks:
                await asyncio.gather(*request_tasks)

        await asyncio.wait_for(launch(), timeout=args.timeout_s)
    except Exception as exc:
        metadata["run_exception"] = f"{type(exc).__name__}: {exc}"
        metadata["run_traceback"] = traceback.format_exc()
        (run_dir / "failure.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    finally:
        stop_telemetry.set()
        if telemetry_task is not None:
            try:
                await telemetry_task
            except Exception:
                pass
        if request_tasks:
            for task in request_tasks:
                if not task.done():
                    task.cancel()
        if request_tasks:
            await asyncio.gather(*request_tasks, return_exceptions=True)
        if engine_task is not None:
            engine_task.cancel()
            await asyncio.gather(engine_task, return_exceptions=True)
        end_ns = time.perf_counter_ns()
        metadata["measurement_end_time_ns"] = end_ns
        metadata["measurement_end_utc"] = _utc_now()
        if "measurement_start_time_ns" in metadata:
            metadata["measurement_duration_s"] = (end_ns - metadata["measurement_start_time_ns"]) / NS_PER_SECOND
        metadata["launched_request_count"] = counters["launched"]
        metadata["completed_request_count"] = counters["completed"]
        metadata["generated_output_token_count"] = counters["generated_tokens"]
        metadata["num_gpu_blocks_observed_in_telemetry"] = sorted({row.get("num_gpu_blocks") for row in telemetry_rows})
        metadata["completed_at_utc"] = _utc_now()
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with (run_dir / "requests.jsonl").open("w", encoding="utf-8") as handle:
            for row in sorted(request_rows.values(), key=lambda value: value["sequence"]):
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        with (run_dir / "telemetry.jsonl").open("w", encoding="utf-8") as handle:
            for row in telemetry_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        if (run_dir / "requests.jsonl").exists() and (run_dir / "metadata.json").exists():
            try:
                summary = summarize_run(run_dir)
                summary["condition"] = args.condition
                summary["quantized_layer_count"] = args.quantized_layer_count
                (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            except Exception as summary_exc:
                (run_dir / "summary_failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
                print(f"summary regeneration failed: {summary_exc}")

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    args = parser.parse_args()
    path = asyncio.run(run(args))
    print(f"raw static quantization run: {path}")


if __name__ == "__main__":
    main()
