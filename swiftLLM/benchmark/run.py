"""Run an open-loop streaming benchmark against the existing SwiftLLM Engine.

Example (from the repository's ``swiftLLM`` directory)::

  PYTHONPATH="$PWD:$PWD/csrc" CUDA_VISIBLE_DEVICES=3 \
    /nfs/home/s314511048/.venv/bin/python -m benchmark.run \
    --model-path /path/to/Llama-3.1-8B \
    --target-rps 0.2 --arrival-mode fixed --request-count 4 \
    --prompt-token-count 8 --output-token-count 4 \
    --output-dir ../benchmark-results/low-load

The request launcher never waits for a prior request to complete. Every raw
request and telemetry sample is written in machine-readable form when the run
finishes; ``benchmark.summarize`` regenerates the summary from those files.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any
import uuid

import swiftllm
from transformers import AutoTokenizer

from .metrics import derive_request_metrics
from .summarize import summarize_run
from .workload import ScheduledRequest, build_schedule

UPSTREAM_COMMIT = "682cf9a28f97f7490409981a2f181528f377eb5d"
NS_PER_SECOND = 1_000_000_000


class Counters:
    def __init__(self) -> None:
        self.launched = 0
        self.completed = 0
        self.generated_tokens = 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_model_path(model_path: str) -> str:
    path = os.path.abspath(model_path)
    home = os.path.expanduser("~")
    if path == home or path.startswith(home + os.sep):
        return "~" + path[len(home):]
    return path


def _git_info() -> tuple[str | None, bool | None, str | None, str | None]:
    repo = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        diff = subprocess.check_output(
            ["git", "-C", str(repo), "diff", "--binary"],
            stderr=subprocess.DEVNULL,
        )
        return commit, bool(status.strip()), hashlib.sha256(diff).hexdigest(), status
    except (OSError, subprocess.CalledProcessError):
        return None, None, None, None


def _gpu_metadata() -> dict[str, Any]:
    result: dict[str, Any] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": None,
        "gpu_index_visible_to_process": None,
    }
    try:
        import torch

        if torch.cuda.is_available():
            result["gpu_name"] = torch.cuda.get_device_name(0)
            result["gpu_index_visible_to_process"] = 0
    except Exception as exc:  # telemetry must not make a valid run fail
        result["gpu_metadata_error"] = str(exc)
    return result


def _gpu_memory() -> dict[str, Any]:
    """Use driver-level free/total memory, which is meaningful with KV preallocation."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {
                "gpu_memory_used_bytes": None,
                "gpu_memory_free_bytes": None,
                "gpu_memory_total_bytes": None,
                "gpu_memory_source": "unavailable",
            }
        free, total = torch.cuda.mem_get_info()
        return {
            "gpu_memory_used_bytes": int(total - free),
            "gpu_memory_free_bytes": int(free),
            "gpu_memory_total_bytes": int(total),
            "gpu_memory_source": "torch.cuda.mem_get_info",
        }
    except Exception as exc:  # physical memory is optional by contract
        return {
            "gpu_memory_used_bytes": None,
            "gpu_memory_free_bytes": None,
            "gpu_memory_total_bytes": None,
            "gpu_memory_source": f"unavailable: {exc}",
        }


def _make_exact_prompt(tokenizer: Any, token_count: int) -> str:
    """Create a deterministic text prompt that re-tokenizes to token_count."""
    if token_count <= 0:
        raise ValueError("prompt token count must be positive")
    unit_ids = tokenizer.encode(" hello", add_special_tokens=False)
    if len(unit_ids) != 1:
        raise ValueError("could not find a one-token prompt padding unit")
    prompt = tokenizer.decode(
        unit_ids * max(token_count - 1, 0),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    actual = len(tokenizer(prompt, return_attention_mask=False)["input_ids"])
    if actual != token_count:
        raise ValueError(f"generated prompt has {actual} tokens, expected {token_count}")
    return prompt


def _add_args(parser: argparse.ArgumentParser) -> None:
    swiftllm.EngineConfig.add_cli_args(parser)
    # The online.py values are the validated Phase-1 baseline, rather than the
    # broader upstream CLI defaults.
    parser.set_defaults(
        block_size=16,
        gpu_mem_utilization=0.99,
        num_cpu_blocks=1024,
        max_seqs_in_block_table=128,
        max_blocks_per_seq=3072,
        max_batch_size=4,
        max_tokens_in_batch=1024,
    )
    parser.add_argument("--target-rps", type=float, required=True)
    parser.add_argument("--arrival-mode", choices=("fixed", "poisson"), default="fixed")
    schedule_group = parser.add_mutually_exclusive_group(required=True)
    schedule_group.add_argument("--request-count", type=int)
    schedule_group.add_argument("--duration-s", type=float)
    parser.add_argument("--prompt", help="Fixed prompt text; otherwise generate exact --prompt-token-count tokens")
    parser.add_argument("--prompt-token-count", type=int, default=32)
    parser.add_argument("--output-token-count", type=int, default=8)
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--telemetry-interval-s", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--expected-num-gpu-blocks", type=int, default=3880)


def _build_engine_config(args: argparse.Namespace) -> swiftllm.EngineConfig:
    return swiftllm.EngineConfig(
        model_path=args.model_path,
        use_dummy=args.use_dummy,
        block_size=args.block_size,
        gpu_mem_utilization=args.gpu_mem_utilization,
        num_cpu_blocks=args.num_cpu_blocks,
        max_seqs_in_block_table=args.max_seqs_in_block_table,
        max_blocks_per_seq=args.max_blocks_per_seq,
        max_batch_size=args.max_batch_size,
        max_tokens_in_batch=args.max_tokens_in_batch,
    )


async def _run(args: argparse.Namespace) -> Path:
    if args.output_token_count <= 0:
        raise ValueError("output-token-count must be positive")
    if args.warmup_requests < 0:
        raise ValueError("warmup-requests cannot be negative")
    if args.telemetry_interval_s <= 0:
        raise ValueError("telemetry-interval-s must be positive")

    schedule = build_schedule(
        args.target_rps,
        args.arrival_mode,
        request_count=args.request_count,
        duration_s=args.duration_s,
        seed=args.seed,
    )
    engine_config = _build_engine_config(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    prompt = args.prompt or _make_exact_prompt(tokenizer, args.prompt_token_count)
    expected_prompt_len = len(tokenizer(prompt, return_attention_mask=False)["input_ids"])

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    commit, dirty, diff_sha256, git_status = _git_info()
    engine_config_metadata = asdict(engine_config)
    engine_config_metadata["model_path"] = _safe_model_path(args.model_path)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": _utc_now(),
        "upstream_swiftllm_commit": UPSTREAM_COMMIT,
        "current_morphserve_git_commit": commit,
        "current_morphserve_git_dirty": dirty,
        "current_morphserve_git_diff_sha256": diff_sha256,
        "current_morphserve_git_status_porcelain": git_status,
        "model_identifier": Path(args.model_path).name,
        "model_path_safe": _safe_model_path(args.model_path),
        "model_path_sha256": hashlib.sha256(os.path.abspath(args.model_path).encode()).hexdigest(),
        "gpu": _gpu_metadata(),
        "cuda_device_requested": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "engine_config": engine_config_metadata,
        "block_size": args.block_size,
        "expected_num_gpu_blocks": args.expected_num_gpu_blocks,
        "target_rps": args.target_rps,
        "arrival_mode": args.arrival_mode,
        "prompt_token_count_requested": args.prompt_token_count if args.prompt is None else None,
        "prompt_token_count_observed_by_client_tokenizer": expected_prompt_len,
        "output_token_count_requested": args.output_token_count,
        "random_seed": args.seed,
        "warmup_policy": {
            "warmup_requests": args.warmup_requests,
            "included_in_raw_requests": False,
        },
        "request_count_requested": args.request_count,
        "duration_s_requested": args.duration_s,
        "schedule_window_duration_s": args.duration_s if args.duration_s is not None else len(schedule) / args.target_rps,
        "planned_schedule_last_offset_s": schedule[-1].planned_offset_s,
        "telemetry_interval_s": args.telemetry_interval_s,
        "raw_files": ["metadata.json", "requests.jsonl", "telemetry.jsonl"],
        "timestamp_definition": {
            "clock": "time.perf_counter_ns monotonic process clock for *_time_ns fields",
            "arrival_time_ns": "actual open-loop launch time immediately before Engine.add_request_and_stream",
            "planned_arrival_offset_s": "offered schedule offset from measurement start",
            "actual_arrival_offset_s": "(arrival_time_ns - measurement_start_time_ns) / 1e9",
            "arrival_jitter_s": "actual_arrival_offset_s - planned_arrival_offset_s",
            "scheduler_eligible_time_ns": "time when tokenization completed and Engine called Scheduler.on_requests_arrival",
            "first_prefill_time_ns": "time immediately before the model forward containing the request's first prefill",
            "first_output_token_time_ns": "time after model.forward returned the batch containing the first output token",
            "first_stream_token_received_time_ns": "time benchmark consumer received the first StepOutput",
            "completion_time_ns": "time after model.forward returned the final token and before StepOutput enqueue",
            "stream_completion_time_ns": "time benchmark consumer received the final StepOutput",
        },
        "metric_formula": {
            "ttft_s": "(first_stream_token_received_time_ns - arrival_time_ns) / 1e9",
            "engine_ttft_s": "(first_output_token_time_ns - arrival_time_ns) / 1e9",
            "queueing_delay_s": "(first_prefill_time_ns - scheduler_eligible_time_ns) / 1e9",
            "tpot_s": "(stream_completion_time_ns - first_stream_token_received_time_ns) / ((output_token_count - 1) * 1e9), for output_token_count > 1; zero for one token",
            "engine_tpot_s": "(completion_time_ns - first_output_token_time_ns) / ((output_token_count - 1) * 1e9), for output_token_count > 1; zero for one token",
            "logical_kv_utilization": "num_decoding_gpu_blocks / num_gpu_blocks",
            "completed_request_throughput_rps": "completed_request_count / measurement_duration_s",
            "generated_token_throughput_tps": "generated_output_token_count / measurement_duration_s",
        },
    }

    engine = swiftllm.Engine(engine_config)
    await engine.initialize()
    initial_snapshot = engine.get_benchmark_snapshot()
    metadata["num_gpu_blocks"] = initial_snapshot["num_gpu_blocks"]
    metadata["num_cpu_blocks"] = args.num_cpu_blocks
    metadata["gpu_kv_token_slots"] = args.block_size * initial_snapshot["num_gpu_blocks"]
    if args.expected_num_gpu_blocks is not None and initial_snapshot["num_gpu_blocks"] != args.expected_num_gpu_blocks:
        raise RuntimeError(
            f"num_gpu_blocks={initial_snapshot['num_gpu_blocks']} does not match expected "
            f"{args.expected_num_gpu_blocks}; refusing to run a non-baseline configuration"
        )

    engine_task = asyncio.create_task(engine.start_all_event_loops())
    # Let the engine task enter its event loops before the first warmup request.
    await asyncio.sleep(0)
    for warmup_index in range(args.warmup_requests):
        warmup_request = swiftllm.RawRequest(prompt, args.output_token_count)
        async for _ in engine.add_request_and_stream(warmup_request):
            pass
        print(f"warmup completed: {warmup_index + 1}/{args.warmup_requests}")

    baseline_snapshot = engine.get_benchmark_snapshot()
    counters = Counters()
    request_rows: dict[str, dict[str, Any]] = {}
    request_tasks: list[asyncio.Task[None]] = []
    telemetry_rows: list[dict[str, Any]] = []
    stop_telemetry = asyncio.Event()
    measurement_start_loop = asyncio.get_running_loop().time()
    measurement_start_ns = time.perf_counter_ns()
    measurement_start_unix_ns = time.time_ns()
    metadata["measurement_start_time_ns"] = measurement_start_ns
    metadata["measurement_start_utc"] = datetime.fromtimestamp(measurement_start_unix_ns / NS_PER_SECOND, timezone.utc).isoformat()
    baseline_swap_in = baseline_snapshot["swap_in_count"]
    baseline_swap_out = baseline_snapshot["swap_out_count"]

    async def sample_telemetry() -> None:
        while True:
            now_ns = time.perf_counter_ns()
            snapshot = engine.get_benchmark_snapshot()
            elapsed_s = (now_ns - measurement_start_ns) / NS_PER_SECOND
            memory = _gpu_memory()
            row: dict[str, Any] = {
                "timestamp_ns": now_ns,
                "elapsed_s": elapsed_s,
                "waiting_q_depth": snapshot["waiting_q_depth"],
                "running_q_count": snapshot["running_q_count"],
                "swapped_q_count": snapshot["swapped_q_count"],
                "num_decoding_gpu_blocks": snapshot["num_decoding_gpu_blocks"],
                "num_gpu_blocks": snapshot["num_gpu_blocks"],
                "logical_kv_utilization": (
                    snapshot["num_decoding_gpu_blocks"] / snapshot["num_gpu_blocks"]
                    if snapshot["num_gpu_blocks"] else None
                ),
                "completed_request_count": counters.completed,
                "generated_output_token_count": counters.generated_tokens,
                "achieved_request_throughput_rps": counters.completed / elapsed_s if elapsed_s > 0 else None,
                "generated_token_throughput_tps": counters.generated_tokens / elapsed_s if elapsed_s > 0 else None,
                "swap_in_count": snapshot["swap_in_count"] - baseline_swap_in,
                "swap_out_count": snapshot["swap_out_count"] - baseline_swap_out,
                "preemption_count": snapshot["swap_out_count"] - baseline_swap_out,
            }
            row.update(memory)
            telemetry_rows.append(row)
            if stop_telemetry.is_set():
                return
            try:
                await asyncio.wait_for(stop_telemetry.wait(), timeout=args.telemetry_interval_s)
            except asyncio.TimeoutError:
                pass

    telemetry_task = asyncio.create_task(sample_telemetry())

    async def consume_request(item: ScheduledRequest, planned_time_ns: int) -> None:
        request_id = f"r{item.sequence:06d}"
        actual_arrival_ns = time.perf_counter_ns()
        actual_arrival_unix_ns = time.time_ns()
        row: dict[str, Any] = {
            "schema_version": 1,
            "benchmark_request_id": request_id,
            "sequence": item.sequence,
            "status": "launched",
            "planned_arrival_offset_s": item.planned_offset_s,
            "planned_arrival_time_ns": planned_time_ns,
            "actual_arrival_offset_s": (actual_arrival_ns - measurement_start_ns) / NS_PER_SECOND,
            "arrival_jitter_s": (actual_arrival_ns - planned_time_ns) / NS_PER_SECOND,
            "arrival_time_ns": actual_arrival_ns,
            "arrival_time_unix_ns": actual_arrival_unix_ns,
            "prompt_token_count_requested": args.prompt_token_count if args.prompt is None else None,
            "requested_output_token_count": args.output_token_count,
            "prompt_token_count": None,
            "output_token_count": 0,
            "first_prefill_time_ns": None,
            "scheduler_eligible_time_ns": None,
            "first_output_token_time_ns": None,
            "first_stream_token_received_time_ns": None,
            "completion_time_ns": None,
            "stream_completion_time_ns": None,
            "error": None,
        }
        request_rows[request_id] = row
        counters.launched += 1
        raw_request = swiftllm.RawRequest(
            prompt,
            args.output_token_count,
            benchmark_request_id=request_id,
            benchmark_arrival_time_ns=actual_arrival_ns,
        )
        try:
            async for step_output in engine.add_request_and_stream(raw_request):
                received_ns = time.perf_counter_ns()
                counters.generated_tokens += 1
                if row["first_stream_token_received_time_ns"] is None:
                    row["first_stream_token_received_time_ns"] = received_ns
                if step_output.request.prompt_len:
                    row["prompt_token_count"] = step_output.request.prompt_len
                row["output_token_count"] = len(step_output.request.output_token_ids)
                if step_output.request.is_finished():
                    row["stream_completion_time_ns"] = received_ns
                    row["status"] = "completed"
                    counters.completed += 1
                    row["first_prefill_time_ns"] = step_output.request.benchmark_first_prefill_time_ns
                    row["scheduler_eligible_time_ns"] = step_output.request.benchmark_scheduler_eligible_time_ns
                    row["first_output_token_time_ns"] = step_output.request.benchmark_first_output_token_time_ns
                    row["completion_time_ns"] = step_output.request.benchmark_completion_time_ns
                    break
        except Exception as exc:  # retain an auditable failed row
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"

        row.update(derive_request_metrics(row))

    async def launch_schedule() -> None:
        loop = asyncio.get_running_loop()
        # Use the same loop/perf-counter origin captured at measurement start;
        # launch jitter is then directly auditable from the raw row.
        schedule_start_loop = measurement_start_loop
        for item in schedule:
            planned_loop_time = schedule_start_loop + item.planned_offset_s
            delay = planned_loop_time - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            planned_ns = measurement_start_ns + int(item.planned_offset_s * NS_PER_SECOND)
            # This task is deliberately not awaited here: launch cadence is not
            # coupled to any request's token generation or completion.
            request_tasks.append(asyncio.create_task(consume_request(item, planned_ns)))

    try:
        await launch_schedule()
        await asyncio.gather(*request_tasks)
    finally:
        # Stop sampling first so the final sample is inside the recorded
        # measurement interval, then close the interval at the same clock.
        stop_telemetry.set()
        await telemetry_task
        measurement_end_ns = time.perf_counter_ns()
        metadata["measurement_end_time_ns"] = measurement_end_ns
        metadata["measurement_end_utc"] = _utc_now()
        metadata["measurement_duration_s"] = (measurement_end_ns - measurement_start_ns) / NS_PER_SECOND
        metadata["launched_request_count"] = counters.launched
        metadata["completed_request_count"] = counters.completed
        metadata["generated_output_token_count"] = counters.generated_tokens
        engine_task.cancel()
        try:
            await engine_task
        except asyncio.CancelledError:
            pass

    metadata["num_gpu_blocks_observed_in_telemetry"] = sorted({row["num_gpu_blocks"] for row in telemetry_rows})
    metadata["completed_at_utc"] = _utc_now()
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (run_dir / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for row in sorted(request_rows.values(), key=lambda value: value["sequence"]):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (run_dir / "telemetry.jsonl").open("w", encoding="utf-8") as handle:
        for row in telemetry_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = summarize_run(run_dir)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_args(parser)
    args = parser.parse_args()
    run_dir = asyncio.run(_run(args))
    print(f"raw benchmark run: {run_dir}")


if __name__ == "__main__":
    main()
