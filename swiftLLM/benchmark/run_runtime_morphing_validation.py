"""Run deterministic GPU validation of one-process FP16/AWQ runtime morphing."""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import torch
from transformers import AutoTokenizer

from swiftllm.engine_config import EngineConfig
from swiftllm.server.engine import Engine
from swiftllm.server.structs import RawRequest, Request
from swiftllm.worker.model import LlamaModel
from swiftllm.worker.weight import layer_variant_layout, layer_variant_tensor_items


REPO_ROOT = Path(__file__).resolve().parents[2]
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
    "swiftLLM/benchmark/run_runtime_morphing_validation.py",
    "swiftLLM/benchmark/test_runtime_morphing.py",
)

PROMPTS = (
    "The capital of France is Paris. The capital of Germany is",
    "Mercury is the closest planet to the Sun. The largest planet is",
)


def engine_config(
    model_path: Path,
    awq_path: Path,
    *,
    quantized_layers: int,
    runtime: bool,
    verify_kv: bool = True,
) -> EngineConfig:
    return EngineConfig(
        model_path=str(model_path),
        use_dummy=False,
        block_size=16,
        gpu_mem_utilization=0.99,
        num_cpu_blocks=1024,
        max_seqs_in_block_table=128,
        max_blocks_per_seq=3072,
        max_batch_size=32,
        max_tokens_in_batch=49152,
        quantized_layer_count=quantized_layers,
        quantization_backend="awq_marlin",
        quantized_model_path=str(awq_path),
        enable_runtime_morphing=runtime,
        runtime_awq_target_blocks=4170,
        runtime_verify_kv=verify_kv,
    )


def gpu_info() -> dict[str, Any]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "name": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
    }


def provenance() -> dict:
    commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    diff = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "diff", "--binary", "HEAD", "--", *SOURCE_PATHS]
    )
    hashes = {
        relative: hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
        for relative in SOURCE_PATHS
    }
    return {
        "git_commit": commit,
        "runtime_source_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "runtime_source_files_sha256": hashes,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def active_layer_digest(layers) -> dict:
    digest = hashlib.sha256()
    compared_bytes = 0
    for layer in layers:
        for name, tensor in layer_variant_tensor_items(layer):
            if name.endswith((".workspace", ".g_idx", ".g_idx_sort_indices")):
                continue
            host = tensor.detach().contiguous().cpu()
            digest.update(name.encode())
            digest.update(memoryview(host.numpy()))
            compared_bytes += host.numel() * host.element_size()
    return {
        "sha256": digest.hexdigest(),
        "compared_bytes": compared_bytes,
        "layouts": [layer_variant_layout(layer) for layer in layers],
    }


def max_shape_capacity_probe(model: LlamaModel, config: EngineConfig) -> dict:
    """Re-run the frozen static peak envelope with dynamic KV still resident."""
    input_lens = [config.max_tokens_in_batch // config.max_batch_size] * config.max_batch_size
    input_lens[-1] += config.max_tokens_in_batch % config.max_batch_size
    inputs = [[0] * length for length in input_lens]
    block_bytes = config.block_size * model.model_config.get_kvslot_size()
    integrity_before = model.sample_active_kv(max_blocks=2)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    memory_before = LlamaModel._memory_snapshot()
    started_ns = time.perf_counter_ns()
    outputs = model.forward(
        inputs,
        list(range(config.max_batch_size)),
        [],
        ignore_kvcache=True,
    )
    torch.cuda.synchronize()
    memory_at_envelope = LlamaModel._memory_snapshot()
    elapsed_ns = time.perf_counter_ns() - started_ns
    integrity_after = model.sample_active_kv(max_blocks=2)
    if integrity_before["logical_digest"] != integrity_after["logical_digest"]:
        raise RuntimeError("max-shape no-KV probe changed active KV state")
    driver_used = memory_at_envelope["total_bytes"] - memory_at_envelope["free_bytes"]
    usable = int(memory_at_envelope["total_bytes"] * config.gpu_mem_utilization)
    additional_safe_blocks = max(0, usable - driver_used) // block_bytes
    result = {
        "batch_size": config.max_batch_size,
        "token_count": config.max_tokens_in_batch,
        "ignore_kvcache": True,
        "output_count": len(outputs),
        "outputs_valid": len(outputs) == config.max_batch_size and all(
            0 <= token < model.model_config.vocab_size for token in outputs
        ),
        "elapsed_ns": elapsed_ns,
        "memory_before": memory_before,
        "memory_at_envelope": memory_at_envelope,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "driver_used_bytes_at_envelope": driver_used,
        "usable_budget_bytes": usable,
        "headroom_bytes_under_budget": max(0, usable - driver_used),
        "additional_safe_blocks_from_measured_headroom": additional_safe_blocks,
        "derived_safe_total_blocks": model.num_blocks + additional_safe_blocks,
        "physical_blocks_during_probe": model.num_blocks,
        "active_kv_integrity_before": integrity_before,
        "active_kv_integrity_after": integrity_after,
    }
    torch.cuda.empty_cache()
    return result


def static_condition(
    condition: str,
    model_path: Path,
    awq_path: Path,
    output_len: int,
) -> dict:
    quantized_layers = 0 if condition == "fp16" else 16
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompt_ids = [
        tokenizer(prompt, return_attention_mask=False)["input_ids"] for prompt in PROMPTS
    ]
    config = engine_config(
        model_path,
        awq_path,
        quantized_layers=quantized_layers,
        runtime=False,
        verify_kv=False,
    )
    started_ns = time.perf_counter_ns()
    model = LlamaModel(config)
    model.load_weights()
    blocks = model.profile_num_blocks()
    model.init_kvcache_and_swap(blocks)
    selected_layer_digest = active_layer_digest(model.weight.layers[:16])
    output_ids = [[] for _ in PROMPTS]
    steps = []
    for step_index in range(output_len):
        is_prefill = step_index == 0
        inputs = prompt_ids if is_prefill else [[tokens[-1]] for tokens in output_ids]
        lengths = [] if is_prefill else [
            len(ids) + len(tokens) for ids, tokens in zip(prompt_ids, output_ids)
        ]
        tokens = model.forward(inputs, list(range(len(PROMPTS))), lengths)
        timestamp_ns = time.perf_counter_ns()
        for sequence, token in enumerate(tokens):
            output_ids[sequence].append(int(token))
            steps.append({
                "sequence": sequence,
                "request_id": sequence,
                "step_index": step_index,
                "is_prefill": is_prefill,
                "input_position": len(prompt_ids[sequence]) - 1 + step_index,
                "input_token_id": None if is_prefill else inputs[sequence][0],
                "output_token_id": int(token),
                "precision_state": "FP16" if quantized_layers == 0 else "AWQ_MARLIN_W4_16",
                "timestamp_ns": timestamp_ns,
            })
    model.free_seqs_resources(list(range(len(PROMPTS))))
    torch.cuda.synchronize()
    result = {
        "schema_version": 1,
        "condition": condition,
        "provenance": provenance(),
        "gpu": gpu_info(),
        "model_path": str(model_path),
        "quantized_model_path": str(awq_path),
        "num_gpu_blocks": blocks,
        "physical_kv_bytes": blocks * config.block_size * model.model_config.get_kvslot_size(),
        "selected_layer_digest": selected_layer_digest,
        "prompt_token_ids": prompt_ids,
        "output_token_ids": output_ids,
        "steps": steps,
        "transition_boundaries": [],
        "elapsed_ns": time.perf_counter_ns() - started_ns,
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


async def mixed_condition(
    condition: str,
    model_path: Path,
    awq_path: Path,
    output_len: int,
    morph_after: int,
    restore_after: int,
) -> dict:
    config = engine_config(
        model_path,
        awq_path,
        quantized_layers=0,
        runtime=True,
        verify_kv=True,
    )
    engine = Engine(config)
    started_ns = time.perf_counter_ns()
    await engine.initialize()
    initial_snapshot = engine.get_benchmark_snapshot()
    engine.start_benchmark_batch_observation()
    engine_task = asyncio.create_task(engine.start_all_event_loops())
    await asyncio.sleep(0)

    steps: list[list[dict]] = [[] for _ in PROMPTS]
    requests = [None for _ in PROMPTS]

    async def consume(sequence: int) -> None:
        raw = RawRequest(
            PROMPTS[sequence],
            output_len,
            benchmark_request_id=f"runtime-r{sequence}",
        )
        async for output in engine.add_request_and_stream(raw):
            request = output.request
            requests[sequence] = request
            step_index = len(request.output_token_ids) - 1
            steps[sequence].append({
                "sequence": sequence,
                "request_id": request.request_id,
                "benchmark_request_id": request.benchmark_request_id,
                "step_index": step_index,
                "is_prefill": step_index == 0,
                "input_position": output.input_position,
                "input_token_id": (
                    None if step_index == 0 else request.output_token_ids[step_index - 1]
                ),
                "output_token_id": int(output.token_id),
                "precision_state": output.precision_state,
                "timestamp_ns": time.perf_counter_ns(),
            })

    consumers = [asyncio.create_task(consume(i)) for i in range(len(PROMPTS))]

    async def wait_for_prefix(length: int) -> None:
        while min((len(rows) for rows in steps), default=0) < length:
            await asyncio.sleep(0.001)

    transitions = []
    await wait_for_prefix(morph_after)
    boundary_before = [len(rows) for rows in steps]
    morph_trace = await engine.morph_to_awq_w4_16()
    morph_trace["boundary_output_lengths_before_request"] = boundary_before
    morph_trace["boundary_output_lengths_after_completion"] = [len(rows) for rows in steps]
    transitions.append(morph_trace)

    if condition == "roundtrip":
        await wait_for_prefix(restore_after)
        boundary_before = [len(rows) for rows in steps]
        restore_trace = await engine.restore_to_fp16()
        restore_trace["boundary_output_lengths_before_request"] = boundary_before
        restore_trace["boundary_output_lengths_after_completion"] = [len(rows) for rows in steps]
        transitions.append(restore_trace)

    await asyncio.gather(*consumers)
    deadline = asyncio.get_running_loop().time() + 5.0
    while engine.get_benchmark_snapshot()["running_q_count"] and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.001)
    final_snapshot = engine.get_benchmark_snapshot()
    engine_task.cancel()
    try:
        await engine_task
    except asyncio.CancelledError:
        pass

    flattened_steps = [row for sequence_rows in steps for row in sequence_rows]
    request_summaries = []
    for sequence, request in enumerate(requests):
        request_summaries.append({
            "sequence": sequence,
            "request_id": request.request_id,
            "benchmark_request_id": request.benchmark_request_id,
            "prompt_len": request.prompt_len,
            "output_len": len(request.output_token_ids),
            "prefill_timestamp_ns": request.benchmark_first_prefill_time_ns,
            "first_output_timestamp_ns": request.benchmark_first_output_token_time_ns,
            "completion_timestamp_ns": request.benchmark_completion_time_ns,
        })
    return {
        "schema_version": 1,
        "condition": condition,
        "provenance": provenance(),
        "gpu": gpu_info(),
        "model_path": str(model_path),
        "quantized_model_path": str(awq_path),
        "initial_snapshot": initial_snapshot,
        "final_snapshot": final_snapshot,
        "runtime_preparation_trace": engine.model.runtime_preparation_trace,
        "prompt_token_ids": [request.prompt_token_ids for request in requests],
        "output_token_ids": [request.output_token_ids for request in requests],
        "steps": flattened_steps,
        "request_summaries": request_summaries,
        "batch_events": engine.get_benchmark_batch_events(),
        "transition_boundaries": transitions,
        "elapsed_ns": time.perf_counter_ns() - started_ns,
    }


async def capacity_cycles(
    model_path: Path,
    awq_path: Path,
    cycles: int,
) -> dict:
    config = engine_config(
        model_path,
        awq_path,
        quantized_layers=0,
        runtime=True,
        verify_kv=True,
    )
    engine = Engine(config)
    started_ns = time.perf_counter_ns()
    await engine.initialize()
    base_blocks = engine.model.base_num_blocks
    rows = []
    transition_traces = []
    awq_equivalence = None
    fp16_equivalence = None
    full_model_extension_execution = None
    max_shape_capacity_validation = None
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    for cycle in range(cycles):
        before = engine.get_benchmark_snapshot()
        morph = await engine.morph_to_awq_w4_16()
        transition_traces.append(morph)
        if cycle == 0:
            awq_equivalence = engine.model.verify_active_runtime_variant(
                "AWQ_MARLIN_W4_16"
            )

        manager = engine.model.gpu_block_manager
        if cycle == 0:
            prompt_ids = tokenizer(
                PROMPTS[0], return_attention_mask=False
            )["input_ids"]
            generated = engine.model.forward([prompt_ids], [0], [])
            real_seq = torch.tensor([0], dtype=torch.int32, device="cuda")
            crossing_fill_seq = torch.tensor([119], dtype=torch.int32, device="cuda")
            with torch.inference_mode():
                manager.allocate_blocks_for_seqs(
                    crossing_fill_seq,
                    torch.tensor(
                        [(base_blocks - 1) * config.block_size],
                        dtype=torch.int32,
                        device="cuda",
                    ),
                )
            for _ in range(4):
                generated.extend(engine.model.forward(
                    [[generated[-1]]],
                    [0],
                    [len(prompt_ids) + len(generated)],
                ))
            torch.cuda.synchronize()
            crossing_ids = manager.block_table[0, :2].cpu().tolist()
            if crossing_ids[0] >= base_blocks or crossing_ids[1] < base_blocks:
                raise RuntimeError("full-model history did not cross the KV segment boundary")
            full_model_extension_execution = {
                "prompt_token_ids": prompt_ids,
                "output_token_ids_before_restore": generated,
                "block_table_ids_before_restore": crossing_ids,
                "base_blocks": base_blocks,
                "finite_valid_tokens_before_restore": all(
                    0 <= token < engine.model.model_config.vocab_size
                    for token in generated
                ),
                "logical_kv_before_restore": engine.model.sample_active_kv(max_blocks=2),
            }
            max_shape_capacity_validation = max_shape_capacity_probe(engine.model, config)
            with torch.inference_mode():
                manager.free_blocks_for_seqs(crossing_fill_seq)
            torch.cuda.synchronize()

        if cycle == 0:
            live_request = Request(RawRequest(
                PROMPTS[0], 6, benchmark_request_id="extension-live-r0"
            ))
            live_request.prompt_token_ids = prompt_ids
            live_request.prompt_len = len(prompt_ids)
            live_request.request_id = 0
            live_request.output_token_ids = list(generated)
            engine.scheduler.running_q = [live_request]
            engine.scheduler.num_decoding_gpu_blocks = 2
            extension_ids_before = manager.block_table[0, :2].cpu().tolist()
            integrity_before = engine.model.sample_active_kv(max_blocks=2)

            restore = await engine.restore_to_fp16()
            transition_traces.append(restore)
            extension_ids_after = manager.block_table[0, :2].cpu().tolist()
            integrity_after = engine.model.sample_active_kv(max_blocks=2)
            if integrity_before["logical_digest"] != integrity_after["logical_digest"]:
                raise RuntimeError("live extension remap changed KV bytes")
            if not all(block_id < base_blocks for block_id in extension_ids_after):
                raise RuntimeError("live restored block table still references extension")
            fp16_equivalence = engine.model.verify_active_runtime_variant("FP16")
            post_restore_input_position = len(prompt_ids) + len(generated) - 1
            post_restore_token = engine.model.forward(
                [[generated[-1]]],
                [0],
                [len(prompt_ids) + len(generated)],
            )[0]
            live_request.output_token_ids.append(post_restore_token)
            full_model_extension_execution.update({
                "block_table_ids_after_restore": extension_ids_after,
                "logical_kv_after_restore": integrity_after,
                "post_restore_input_position": post_restore_input_position,
                "post_restore_output_token_id": post_restore_token,
                "request_id_before_after": [0, live_request.request_id],
                "output_token_ids_after_restore": live_request.output_token_ids,
                "post_restore_token_valid": (
                    0 <= post_restore_token < engine.model.model_config.vocab_size
                ),
            })
            engine.model.free_seqs_resources([0])
            engine.scheduler.running_q = []
            engine.scheduler.num_decoding_gpu_blocks = 0
        else:
            fill_seq = torch.tensor([120], dtype=torch.int32, device="cuda")
            extension_seq = torch.tensor([121], dtype=torch.int32, device="cuda")
            with torch.inference_mode():
                manager.allocate_blocks_for_seqs(
                    fill_seq,
                    torch.tensor([base_blocks * config.block_size], dtype=torch.int32, device="cuda"),
                )
                manager.allocate_blocks_for_seqs(
                    extension_seq,
                    torch.tensor([2 * config.block_size], dtype=torch.int32, device="cuda"),
                )
            torch.cuda.synchronize()
            extension_ids_before = manager.block_table[121, :2].cpu().tolist()
            if not all(block_id >= base_blocks for block_id in extension_ids_before):
                raise RuntimeError("test allocation did not reach physical extension blocks")
            with torch.inference_mode():
                for index, virtual_id in enumerate(extension_ids_before):
                    local_id = virtual_id - base_blocks
                    engine.model.k_cache_extension[local_id].fill_(cycle + index + 1)
                    engine.model.v_cache_extension[local_id].fill_(cycle + index + 101)
                manager.free_blocks_for_seqs(fill_seq)
            torch.cuda.synchronize()
            integrity_before = engine.model.sample_active_kv(max_blocks=2)

            restore = await engine.restore_to_fp16()
            transition_traces.append(restore)
            extension_ids_after = manager.block_table[121, :2].cpu().tolist()
            integrity_after = engine.model.sample_active_kv(max_blocks=2)
            if integrity_before["logical_digest"] != integrity_after["logical_digest"]:
                raise RuntimeError("extension-to-base remap changed KV bytes")
            if not all(block_id < base_blocks for block_id in extension_ids_after):
                raise RuntimeError("restored block table still references extension")
            with torch.inference_mode():
                manager.free_blocks_for_seqs(extension_seq)

        torch.cuda.synchronize()
        manager.assert_consistent()
        after = engine.get_benchmark_snapshot()
        rows.append({
            "cycle": cycle,
            "snapshot_before": before,
            "snapshot_after_morph": morph["engine_context_after"],
            "snapshot_after_restore": after,
            "extension_ids_before_restore": extension_ids_before,
            "base_ids_after_restore": extension_ids_after,
            "kv_integrity_before": integrity_before,
            "kv_integrity_after": integrity_after,
            "memory_after_restore": LlamaModel._memory_snapshot(),
        })

    return {
        "schema_version": 1,
        "condition": "capacity_cycles",
        "provenance": provenance(),
        "gpu": gpu_info(),
        "model_path": str(model_path),
        "quantized_model_path": str(awq_path),
        "cycles": cycles,
        "base_blocks": base_blocks,
        "dynamic_awq_blocks": config.runtime_awq_target_blocks,
        "base_physical_kv_bytes": base_blocks * config.block_size * engine.model.model_config.get_kvslot_size(),
        "dynamic_awq_physical_kv_bytes": config.runtime_awq_target_blocks * config.block_size * engine.model.model_config.get_kvslot_size(),
        "runtime_preparation_trace": engine.model.runtime_preparation_trace,
        "awq_runtime_equivalence": awq_equivalence,
        "fp16_runtime_equivalence": fp16_equivalence,
        "full_model_extension_execution": full_model_extension_execution,
        "max_shape_capacity_validation": max_shape_capacity_validation,
        "cycle_rows": rows,
        "transition_traces": transition_traces,
        "elapsed_ns": time.perf_counter_ns() - started_ns,
    }


async def run(args: argparse.Namespace) -> dict:
    if args.condition in ("fp16", "static_awq"):
        return static_condition(
            args.condition,
            args.model_path,
            args.quantized_model_path,
            args.output_len,
        )
    if args.condition in ("morph", "roundtrip"):
        return await mixed_condition(
            args.condition,
            args.model_path,
            args.quantized_model_path,
            args.output_len,
            args.morph_after,
            args.restore_after,
        )
    return await capacity_cycles(
        args.model_path,
        args.quantized_model_path,
        args.cycles,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        choices=("fp16", "static_awq", "morph", "roundtrip", "capacity_cycles"),
        required=True,
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--quantized-model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-len", type=int, default=24)
    parser.add_argument("--morph-after", type=int, default=4)
    parser.add_argument("--restore-after", type=int, default=12)
    parser.add_argument("--cycles", type=int, default=3)
    args = parser.parse_args()
    result = asyncio.run(run(args))
    write_json(args.output, result)
    print(json.dumps({
        "condition": result["condition"],
        "output": str(args.output),
        "elapsed_s": result["elapsed_ns"] / 1e9,
    }, indent=2))


if __name__ == "__main__":
    main()
