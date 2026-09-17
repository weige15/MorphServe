#!/usr/bin/env python3
"""Common-engine modified-condition workload benchmark.

This runner deliberately keeps the experiment small in mechanism surface rather
than pretending to reproduce the paper's hidden task/policy inputs.  It uses the
same LlamaModel request path for FP16, static real AutoAWQ W4, and reconstructed
MorphServe.  Arrival tasks are independent; a single GPU work lock makes the
queue visible and deterministic without running model calls concurrently.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from morphserve.autoawq_adapter import (
    AsyncLayerCopier,
    backup_fp16_layer,
    install_autoawq_layer,
    load_packed_layer,
    materialize_packed_tensors,
    register_fp16_layer_region,
)
from morphserve.controller import ServingMonitor
from morphserve.integration import AdaptiveCoordinator, queue_state
from morphserve.real_executor import RealMorphingExecutor
from morphserve.replay import percentile


PROFILE_ORDER = [25, 24, 26, 27, 28, 29, 30, 31]
PRECISION_NAMES = {"fp16": "FP16", "static_w4": "W4", "morphserve-default": "MorphServe-default"}


def _int(value):
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


def _now(origin):
    return time.perf_counter() - origin


def _jsonable(value):
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _load_specs(manifest_path: Path, payload: dict, preflight: bool):
    if preflight:
        return [
            {"request_id": "preflight-0", "scheduled_s": 0.0, "source_timestamp_s": None,
             "trace_source": "integrated-preflight", "prompt_token_count": 32,
             "output_len": 8, "source_row_index": None},
            {"request_id": "preflight-1", "scheduled_s": 0.01, "source_timestamp_s": None,
             "trace_source": "integrated-preflight", "prompt_token_count": 32,
             "output_len": 8, "source_row_index": None},
        ]
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty trace manifest: {manifest_path}")
    if any(float(row["scheduled_s"]) < 0 for row in rows):
        raise ValueError("negative scheduled arrival")
    if any(float(a["scheduled_s"]) > float(b["scheduled_s"]) for a, b in zip(rows, rows[1:])):
        raise ValueError("trace manifest is not monotonic")
    result = []
    for row in rows:
        result.append({
            "request_id": row["request_id"],
            "scheduled_s": float(row["scheduled_s"]),
            "source_timestamp_s": row.get("source_timestamp_s"),
            "trace_source": manifest_path.name,
            "source_row_index": row.get("source_row_index"),
            "prompt_token_count": len(payload["token_ids"]),
            "output_len": 512,
        })
    return result


def _make_config(args, cfg, mode):
    from swiftllm.engine_config import EngineConfig
    return EngineConfig(
        org_model_path=args.fp16_model,
        block_size=int(cfg["block_size"]),
        num_cpu_blocks=0,
        max_seqs_in_block_table=int(cfg["max_seqs_in_block_table"]),
        max_blocks_per_seq=int(cfg["max_blocks_per_seq"]),
        max_batch_size=int(cfg["max_batch_size"]),
        max_tokens_in_batch=int(cfg["max_tokens_in_batch"]),
        quant_serve=False,
        use_dummy=False,
    )


def _delete_fp16_layer_storage(model, layer):
    weight = model.weight.layers[layer]
    if weight is not None:
        for name in ("attn_norm", "q_proj", "k_proj", "v_proj", "o_proj", "ffn_norm", "up_gate_proj", "down_proj"):
            if hasattr(weight, name):
                delattr(weight, name)
    model.weight.layers[layer] = None


def _load_model(args, cfg, kind, mode):
    """Load one of the three configurations before the timed origin."""
    import swiftllm_c
    from swiftllm.worker.model import LlamaModel

    model = LlamaModel(_make_config(args, cfg, mode), args.fp16_model)
    model.load_weights()
    # The benchmark has a fixed, predeclared base capacity.  MorphServe adds
    # only physical tails after a real W4 action; it never changes this value
    # in response to measured benchmark results.
    base_blocks = int(cfg["static_w4_base_kv_blocks"] if kind == "static_w4" else cfg["fp16_base_kv_blocks"])
    if kind == "static_w4":
        # Materialize actual packed W4 tensors in independent GPU storage. This
        # is a legitimate static W4 deployment and does not retain duplicate
        # FP16 layer tensors after each replacement.
        packed_gpu = {}
        module_classes = set()
        for layer in range(model.model_config.num_layers):
            register_fp16_layer_region(model, layer, swiftllm_c)
            packed = load_packed_layer(args.w4_model, layer, swiftllm_c)
            tensors = materialize_packed_tensors(packed)
            packed_gpu[layer] = (packed, tensors)
            wrapper, quant_modules = install_autoawq_layer(model, layer, tensors, packed["tensor_map"])
            module_classes.update([type(wrapper).__name__, *(type(module).__name__ for module in quant_modules)])
            _delete_fp16_layer_storage(model, layer)
            torch.cuda.empty_cache()
        model.async_layer_transfers = True
        model.layer_quant_list = list(range(model.model_config.num_layers))
        model.is_layer_quant_list = [True] * model.model_config.num_layers
        model.init_kvcache_and_swap(base_blocks)
        for layer in model.transformer_layers:
            for parent in ("self_attn", "mlp"):
                branch = getattr(layer, parent, None)
                if branch is not None:
                    module_classes.update(type(module).__name__ for module in vars(branch).values())
        module_classes = sorted(module_classes)
        return model, None, {"static_w4_layers": list(range(model.model_config.num_layers)), "module_classes": module_classes, "packed_gpu": packed_gpu}

    model.init_kvcache_and_swap(base_blocks)
    if kind == "fp16":
        model.async_layer_transfers = False
        return model, None, {"static_w4_layers": [], "packed_gpu": {}}

    count = int(cfg["controller_modes"][mode]["max_quantized_layers"])
    layers = PROFILE_ORDER[:count]
    backups = {layer: backup_fp16_layer(model, layer, swiftllm_c) for layer in layers}
    packed = {layer: load_packed_layer(args.w4_model, layer, swiftllm_c) for layer in layers}
    executor = RealMorphingExecutor(model, swiftllm_c, backups, packed)
    return model, executor, {"profile_order": layers, "backups": backups, "packed": packed}


def _warmup(model, executor, kind, payload, cfg, origin, events):
    """Warm kernels and mapping paths, entirely before timed request origin."""
    warm_ids = payload["token_ids"][:32]
    with torch.inference_mode():
        if kind == "morphserve-default":
            # Prepare a real morph path and undo it before timing.  It is not
            # counted as a request or benchmark controller event.
            warm_layers = list(executor.backups)[:2]
            executor.morph_to_w4(warm_layers)
            executor.expand_kv(warm_layers)
            executor.copier.wait_current(warm_layers[0])
            executor.copier.wait_current(warm_layers[1])
            torch.cuda.synchronize()
            executor.recover_fp16(list(reversed(warm_layers)))
            for layer in warm_layers:
                executor.copier.wait_current(layer)
            torch.cuda.synchronize()
            executor.active_layers.clear()
            executor.kv_groups.clear()
            # The warmup transaction has already mutated allocator metadata;
            # use a clean reinitialization before the timed origin.
            model.init_kvcache_and_swap(int(cfg["fp16_base_kv_blocks"]))
            executor._sync_model_state()
        token = _int(model.forward([warm_ids], [0], [], return_logits=True)[0].argmax())
        model.forward([[token]], [0], [len(warm_ids) + 1])
        model.free_seqs_resources([0])
        torch.cuda.synchronize()
    events.append({"type": "warmup_complete", "timestamp_s": _now(origin), "excluded": True})


def _build_scheduler(states):
    return SimpleNamespace(waiting_q=[], running_q=[], swapped_q=[])


def _snapshot(model, states, current_id, origin, mode, event_type="sample", throughput=None):
    manager = model.gpu_block_manager
    total = int(manager.num_blocks)
    free = _int(manager.num_free_blocks)
    _, total_mem = torch.cuda.mem_get_info()
    free_mem, _ = torch.cuda.mem_get_info()
    waiting = [state for state in states.values() if state["status"] == "waiting"]
    running = [state for state in states.values() if state["status"] == "running"]
    oldest = min((_now(origin) - state["actual_submit_s"] for state in waiting), default=0.0)
    active = list(getattr(model, "layer_quant_list", []))
    return {
        "timestamp_s": _now(origin), "event_type": event_type,
        "current_request_id": current_id,
        "queue_depth": len(waiting), "running_requests": len(running),
        "oldest_queue_delay_s": max(0.0, oldest),
        "request_throughput_s": float(throughput[0]) if throughput else 0.0,
        "token_throughput_s": float(throughput[1]) if throughput else 0.0,
        "gpu_memory_used_bytes": int(total_mem - free_mem),
        "gpu_memory_free_bytes": int(free_mem), "gpu_memory_total_bytes": int(total_mem),
        "kv_capacity_blocks": total, "kv_used_blocks": total - free, "kv_free_blocks": free,
        "kv_usage_fraction": (total - free) / total if total else 0.0,
        "active_w4_layer_count": len(active), "active_w4_layers": active,
        "executor_poisoned": bool(getattr(getattr(model, "morph_executor", None), "poisoned", False)),
        "pending_layer_events": sorted(int(layer) for layer in getattr(model, "layer_transfer_events", {})),
        "scheduler_preemptions": None,
        "scheduler_preemptions_measured": False,
    }


def _update_scheduler(scheduler, states, current_id):
    scheduler.waiting_q[:] = [SimpleNamespace(request_id=state["seq_id"]) for state in states.values() if state["status"] == "waiting"]
    scheduler.running_q[:] = [SimpleNamespace(request_id=state["seq_id"]) for state in states.values() if state["status"] == "running" and state["request_id"] != current_id]


def _controller_sample(model, executor, coordinator, scheduler, states, current_id, origin, mode, system, events, force=None):
    manager = model.gpu_block_manager
    total = int(manager.num_blocks)
    free = _int(manager.num_free_blocks)
    if force == "pressure":
        kv, delay = 0.90, 0.110
    elif force == "recovery":
        kv, delay = 0.50, 0.010
    else:
        kv = (total - free) / total if total else 0.0
        waiting = [state for state in states.values() if state["status"] == "waiting"]
        delay = max((_now(origin) - state["actual_submit_s"] for state in waiting), default=0.0)
    # Throughput/latency signals are measured values for the monitor, not
    # tuning inputs.  Controller settings remain frozen in the config file.
    metrics = {
        "gpu_memory_usage": min(1.0, float(torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory)),
        "kv_usage": kv, "queue_depth": len([s for s in states.values() if s["status"] == "waiting"]),
        "queue_delay_s": delay, "throughput_tokens_s": 0.0, "ttft_s": 0.0, "tpot_s": 0.0,
    }
    before = len(executor.log)
    action_start = _now(origin)
    event = coordinator.step(scheduler, action_start, metrics)
    action_end = _now(origin)
    if event["command"]["layer_delta"] > 0:
        event_kind = "pressure_trigger"
    elif event["command"]["layer_delta"] < 0:
        event_kind = "recovery_trigger"
    else:
        event_kind = "controller_sample"
    event_record = {"type": event_kind, "timestamp_s": action_end, "force": force,
                    "monitor_metrics": metrics, "controller_event": _jsonable(event),
                    "executor_log_delta": _jsonable(executor.log[before:]),
                    "real_w4_module_classes": {
                        str(layer): sorted({type(module).__name__ for module in executor.quant_objects[layer][1]})
                        for layer in executor.quant_objects
                    },
                    "action_duration_s": action_end - action_start}
    events.append(event_record)
    if event["command"]["layer_delta"] < 0:
        for layer in event["selected_layers"]:
            event_record.setdefault("restore_events", []).append({
                "layer": layer, "restore_enqueue_timestamp_s": action_start,
                "h2d_completion_timestamp_s": None,
                "first_affected_layer_wait_timestamp_s": None,
                "remaining_transfer_time_at_wait_s": None,
                "remaining_transfer_time_measurement": "not separately observable; next model forward event is recorded",
            })
    return event


def _mark_restore_waits(events, model, origin, forward_start, forward_end, token_tpot):
    for event in events:
        for restore in event.get("restore_events", []):
            if restore["h2d_completion_timestamp_s"] is not None:
                continue
            layer = restore["layer"]
            # The model loop consumes transfer events at the affected layer.
            # Presence of a wait event confirms a real layer-boundary wait.
            if layer in getattr(model, "layer_transfer_wait_events", {}):
                restore["h2d_completion_timestamp_s"] = forward_end
                restore["first_affected_layer_wait_timestamp_s"] = forward_start
                restore["token_tpot_nearest_wait_s"] = token_tpot
                restore["user_visible_wait_window_s"] = max(0.0, forward_end - forward_start)


def _required_blocks(model, prompt_len, generated_count):
    block = int(model.engine_config.block_size)
    target = math.ceil((prompt_len + generated_count) / block)
    manager = model.gpu_block_manager
    allocated = _int(manager.num_seq_allocated_blocks[0]) if False else None
    return target


async def _run_requests(model, executor, coordinator, specs, payload, cfg, kind, mode, preflight, origin, system, events):
    """Run independent arrivals with continuous-batch prefill/decode steps."""
    loop = asyncio.get_running_loop()
    pool = __import__("concurrent.futures").futures.ThreadPoolExecutor(max_workers=1)
    states = {}
    scheduler = _build_scheduler(states)
    for index, spec in enumerate(specs):
        states[spec["request_id"]] = {"request_id": spec["request_id"], "seq_id": index,
                                       "status": "scheduled", "actual_submit_s": None,
                                       "first_service_s": None, "allocated": False,
                                       "generated": [], "token_times": [], "precisions": [],
                                       "error": None, "timeout": False, "spec": spec}
    records = []
    active = []
    arrival_tasks = []
    timed_tokens = 0
    completed = 0
    benchmark_start = _now(origin)

    def throughput():
        elapsed = max(1e-9, _now(origin) - benchmark_start)
        return completed / elapsed, timed_tokens / elapsed

    def add_system(current_id, event_type="token"):
        system.append(_snapshot(model, states, current_id, origin, mode, event_type, throughput()))

    async def mark_arrival(spec):
        await asyncio.sleep(max(0.0, float(spec["scheduled_s"]) - _now(origin)))
        state = states[spec["request_id"]]
        state["status"] = "waiting"
        state["actual_submit_s"] = _now(origin)
        state["queue_entry_s"] = state["actual_submit_s"]
        _update_scheduler(scheduler, states, None)

    for spec in specs:
        arrival_tasks.append(asyncio.create_task(mark_arrival(spec)))

    async def gpu_forward(inputs, seq_ids, lengths):
        def call():
            torch.cuda.set_device(0)
            return model.forward(inputs, seq_ids, lengths, return_logits=True)
        return await loop.run_in_executor(pool, call)

    def request_blocks(state, next_generated_count):
        target = math.ceil((len(payload["token_ids"]) + next_generated_count) / int(model.engine_config.block_size))
        current = _int(model.gpu_block_manager.num_seq_allocated_blocks[state["seq_id"]])
        return max(0, target - current)

    def make_record(state):
        spec = state["spec"]
        intervals = [b - a for a, b in zip(state["token_times"], state["token_times"][1:])]
        first = state["token_times"][0] if state["token_times"] else None
        return {
            "request_id": spec["request_id"], "trace_source": spec["trace_source"],
            "source_row_index": spec.get("source_row_index"), "source_timestamp_s": spec.get("source_timestamp_s"),
            "scheduled_arrival_s": float(spec["scheduled_s"]), "actual_submission_s": state["actual_submit_s"],
            "queue_entry_s": state["queue_entry_s"], "first_service_s": state["first_service_s"],
            "first_token_s": first, "completion_s": state.get("completion_s"),
            "prompt_token_count": len(payload["token_ids"][: int(spec["prompt_token_count"])]),
            "generated_token_count": len(state["generated"]), "token_ids": state["generated"],
            "token_timestamps_s": state["token_times"],
            "ttft_s": None if first is None else first - state["actual_submit_s"],
            "queue_delay_s": None if state["first_service_s"] is None else state["first_service_s"] - state["actual_submit_s"],
            "tpot_intervals_s": intervals, "mean_tpot_s": sum(intervals) / len(intervals) if intervals else None,
            "active_precision_state": state["precisions"], "status": state["status"],
            "error": state["error"], "timeout": state["timeout"],
            "generation_policy": {"forced_tokens": int(spec["output_len"]), "stop_on_eos": False},
        }

    def finish_state(state, status, error=None, timeout=False):
        nonlocal completed
        state["status"] = status; state["error"] = error; state["timeout"] = timeout
        state["completion_s"] = _now(origin)
        if state["allocated"]:
            model.free_seqs_resources([state["seq_id"]]); state["allocated"] = False
        if status == "done":
            completed += 1
        records.append(make_record(state))

    try:
        while len(records) < len(specs):
            now = _now(origin)
            for state in list(active):
                if now - state["actual_submit_s"] >= float(cfg["request_timeout_s"]):
                    finish_state(state, "timeout", "TimeoutError", True)
                    active.remove(state)
            # Waiting requests are admitted in FCFS order whenever their full
            # prompt blocks fit. Arrival tasks remain independent during GPU
            # work because model calls execute in the one-worker pool.
            waiting = [s for s in states.values() if s["status"] == "waiting"]
            if waiting:
                manager = model.gpu_block_manager
                free = _int(manager.num_free_blocks)
                batch = []
                needed = 0
                for state in waiting:
                    req_blocks = math.ceil(int(state["spec"]["prompt_token_count"]) / int(model.engine_config.block_size))
                    if len(batch) >= min(int(model.engine_config.max_batch_size), int(cfg["max_active_requests"]) - len(active)) or needed + req_blocks > free:
                        break
                    batch.append(state); needed += req_blocks
                if batch:
                    for state in batch:
                        state["status"] = "running"; state["first_service_s"] = _now(origin)
                    _update_scheduler(scheduler, states, None)
                    if executor:
                        force = "pressure" if preflight else None
                        _controller_sample(model, executor, coordinator, scheduler, states, None, origin, mode, system, events, force=force)
                    prompt = payload["token_ids"][: int(batch[0]["spec"]["prompt_token_count"])]
                    start = _now(origin)
                    output = await gpu_forward([prompt for _ in batch], [s["seq_id"] for s in batch], [])
                    end = _now(origin)
                    tokens = output.argmax(dim=-1).detach().cpu().tolist()
                    precision = list(getattr(executor, "active_layers", [])) if executor else (list(range(model.model_config.num_layers)) if kind == "static_w4" else [])
                    for state, token in zip(batch, tokens):
                        state["allocated"] = True; state["generated"].append(int(token)); state["token_times"].append(end)
                        state["precisions"].append({"timestamp_s": end, "active_w4_layers": precision,
                                                      "precision": "W4" if kind == "static_w4" else ("FP16" if not precision else "mixed")})
                    timed_tokens += len(batch); _mark_restore_waits(events, model, origin, start, end, None)
                    active.extend(batch); add_system(",".join(s["request_id"] for s in batch), "prefill")
                    await asyncio.sleep(0); continue

            if active:
                # Select a decode batch whose next block allocations fit. All
                # active requests share the same 512-token target in this
                # protocol, but this guard preserves the allocator invariant.
                free = _int(model.gpu_block_manager.num_free_blocks)
                batch = []; needed = 0
                for state in active:
                    if len(batch) >= int(model.engine_config.max_batch_size): break
                    req_needed = request_blocks(state, len(state["generated"]))
                    if needed + req_needed > free: break
                    batch.append(state); needed += req_needed
                if not batch:
                    await asyncio.sleep(0.005); continue
                _update_scheduler(scheduler, states, None)
                if executor:
                    force = "pressure" if preflight and len(active) < 10**9 else None
                    _controller_sample(model, executor, coordinator, scheduler, states, None, origin, mode, system, events, force=force)
                start = _now(origin)
                inputs = [[state["generated"][-1]] for state in batch]
                lengths = [len(payload["token_ids"]) + len(state["generated"]) for state in batch]
                output = await gpu_forward(inputs, [s["seq_id"] for s in batch], lengths)
                end = _now(origin)
                tokens = output.argmax(dim=-1).detach().cpu().tolist()
                precision = list(getattr(executor, "active_layers", [])) if executor else (list(range(model.model_config.num_layers)) if kind == "static_w4" else [])
                for state, token in zip(batch, tokens):
                    state["generated"].append(int(token)); state["token_times"].append(end)
                    state["precisions"].append({"timestamp_s": end, "active_w4_layers": precision,
                                                  "precision": "W4" if kind == "static_w4" else ("FP16" if not precision else "mixed")})
                    intervals = [b - a for a, b in zip(state["token_times"], state["token_times"][1:])]
                    _mark_restore_waits(events, model, origin, start, end, intervals[-1] if intervals else None)
                timed_tokens += len(batch); add_system(",".join(s["request_id"] for s in batch), "decode")
                for state in list(batch):
                    if len(state["generated"]) >= int(state["spec"]["output_len"]):
                        finish_state(state, "done")
                        active.remove(state)
                await asyncio.sleep(0); continue

            # No active batch: wait for the next independently scheduled
            # arrival rather than turning its timestamp into a completion gate.
            pending = [float(s["spec"]["scheduled_s"]) - _now(origin) for s in states.values() if s["status"] == "scheduled"]
            if pending:
                await asyncio.sleep(max(0.001, min(0.05, max(0.0, min(pending)))))
            else:
                await asyncio.sleep(0.005)
            # Predeclared timeouts apply to requests still waiting for service.
            now = _now(origin)
            for state in list(states.values()):
                if state["status"] == "waiting" and now - state["actual_submit_s"] >= float(cfg["request_timeout_s"]):
                    finish_state(state, "timeout", "TimeoutError", True)
    finally:
        await asyncio.gather(*arrival_tasks)
        pool.shutdown(wait=True)

    if executor:
        # Drive the frozen recovery persistence window after the arrival
        # workload drains. Cleanup samples are not serving latency samples.
        for _ in range(int(cfg["preflight"]["recovery_cleanup_samples"])):
            _update_scheduler(scheduler, states, None)
            _controller_sample(model, executor, coordinator, scheduler, states, None, origin, mode, system, events, force="recovery" if preflight else None)
            add_system(None, "controller_cleanup")
            await asyncio.sleep(0)
        for layer in list(getattr(model, "layer_transfer_events", {})):
            executor.copier.wait_current(layer)
        torch.cuda.synchronize()
    records.sort(key=lambda row: row["request_id"])
    return records, scheduler


def _build_summary(records, system, events, model, executor, cfg, kind, trace_name, origin):
    valid = [r for r in records if r["error"] is None and not r["timeout"]]
    ttft = [r["ttft_s"] for r in valid if r["ttft_s"] is not None]
    tpot = [x for r in valid for x in r["tpot_intervals_s"]]
    queues = [r["queue_delay_s"] for r in valid if r["queue_delay_s"] is not None]
    violation = [r for r in records if r["ttft_s"] is not None and r["ttft_s"] > float(cfg["slo_ttft_s"])]
    duration = max((r["completion_s"] for r in records), default=0.0)
    def stats(values):
        return {f"p{q}": percentile(values, q / 100) for q in (50, 95, 99)}
    capacities = [int(s["kv_capacity_blocks"]) for s in system]
    used = [int(s["kv_used_blocks"]) for s in system]
    max_active = max((len(s["active_w4_layers"]) for s in system), default=0)
    quant_times = []
    for a, b in zip(system, system[1:]):
        if a["active_w4_layer_count"] > 0:
            quant_times.append(max(0.0, b["timestamp_s"] - a["timestamp_s"]))
    # Trapezoid time-weighted occupancy/capacity over recorded samples.
    weighted_cap = weighted_used = weighted_time = 0.0
    for a, b in zip(system, system[1:]):
        dt = max(0.0, b["timestamp_s"] - a["timestamp_s"])
        weighted_time += dt
        weighted_cap += dt * (a["kv_capacity_blocks"] + b["kv_capacity_blocks"]) / 2
        weighted_used += dt * (a["kv_used_blocks"] + b["kv_used_blocks"]) / 2
    morph_count = sum(1 for e in events if e["type"] == "pressure_trigger" and e["controller_event"]["command"]["layer_delta"] > 0)
    recovery_count = sum(1 for e in events if e["type"] == "recovery_trigger" and e["controller_event"]["command"]["layer_delta"] < 0)
    final_active = list(getattr(executor, "active_layers", [])) if executor else []
    final_pending = sorted(int(x) for x in getattr(model, "layer_transfer_events", {}))
    return {
        "schema_version": 2, "classification": cfg["classification"], "configuration": PRECISION_NAMES[kind],
        "trace": trace_name, "request_count": len(records), "completed_count": len(valid),
        "completion_rate": len(valid) / len(records) if records else 0.0,
        "error_count": sum(r["error"] is not None for r in records), "timeout_count": sum(r["timeout"] for r in records),
        "ttft_s": stats(ttft), "tpot_s": stats(tpot), "queue_delay_s": stats(queues),
        "ttft_over_2s_count": len(violation), "ttft_over_2s_fraction_submitted": len(violation) / len(records) if records else 0.0,
        "ttft_over_2s_fraction_completed": len(violation) / len(valid) if valid else None,
        "completed_request_throughput_s": len(valid) / duration if duration else 0.0,
        "output_token_throughput_s": sum(r["generated_token_count"] for r in valid) / duration if duration else 0.0,
        "duration_s": duration, "max_queue_depth": max((s["queue_depth"] for s in system), default=0),
        "peak_kv_capacity_blocks": max(capacities, default=0), "peak_kv_used_blocks": max(used, default=0),
        "peak_kv_usage_fraction": max((s["kv_usage_fraction"] for s in system), default=0.0),
        "time_weighted_kv_capacity_blocks": weighted_cap / weighted_time if weighted_time else None,
        "time_weighted_kv_used_blocks": weighted_used / weighted_time if weighted_time else None,
        "morph_count": morph_count, "recovery_count": recovery_count,
        "duration_with_one_or_more_quantized_layers_s": sum(quant_times),
        "maximum_simultaneous_quantized_layers": max_active,
        "scheduler_preemption_count": None, "scheduler_preemptions_measured": False,
        "final_active_w4_layers": final_active, "final_pending_layer_events": final_pending,
        "final_executor_poisoned": bool(getattr(executor, "poisoned", False)) if executor else False,
        "valid_for_comparison": all((r["error"] is not None and r["timeout"]) or (r["error"] is None and r["generated_token_count"] == 512) for r in records) and not final_active and not final_pending,
        "percentile_definition": "Hyndman-Fan type 7 linear interpolation; TTFT per request and TPOT per generated-token interval",
        "raw_accounting_definition": "every submitted request is retained exactly once, including errors/timeouts; invalid runs are excluded from comparison",
    }


def _write_json(path, value):
    Path(path).write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fp16-model", required=True); p.add_argument("--w4-model", required=True)
    p.add_argument("--config", required=True); p.add_argument("--payload", required=True)
    p.add_argument("--trace-manifest", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--kind", choices=("fp16", "static_w4", "morphserve-default"), required=True)
    p.add_argument("--mode", default="default", choices=("accuracy", "default", "performance"))
    p.add_argument("--preflight", action="store_true")
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    controller_path = Path(cfg["controller_config"])
    if not controller_path.is_absolute():
        controller_path = Path(args.config).resolve().parent.parent / controller_path
    cfg["controller_modes"] = json.loads(controller_path.read_text())["modes"]
    payload = json.loads(Path(args.payload).read_text())
    if payload["token_count"] != 1024 or len(payload["token_ids"]) != 1024:
        raise ValueError("frozen payload is not exactly 1024 tokens")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    specs = _load_specs(Path(args.trace_manifest), payload, args.preflight)
    if not args.preflight and any(spec["output_len"] != 512 for spec in specs):
        raise ValueError("benchmark requests must force exactly 512 output tokens")
    model, executor, setup = _load_model(args, cfg, args.kind, args.mode)
    if executor is not None:
        model.morph_executor = executor
    monitor = ServingMonitor(alpha=float(json.loads(controller_path.read_text())["monitor"]["ema_alpha_default"])) if executor else None
    coordinator = AdaptiveCoordinator(setup["profile_order"], executor, mode=args.mode, monitor=monitor) if executor else None
    # Warmup is before timed origin. The origin is reset after warmup below.
    pre_origin = time.perf_counter(); warm_events = []
    _warmup(model, executor, args.kind, payload, cfg, pre_origin, warm_events)
    torch.cuda.synchronize(); origin = time.perf_counter()
    events = list(warm_events); system = []
    records, scheduler = asyncio.run(_run_requests(model, executor, coordinator, specs, payload, cfg, args.kind, args.mode, args.preflight, origin, system, events))
    if executor:
        for layer in list(getattr(model, "layer_transfer_events", {})):
            executor.copier.wait_current(layer)
        torch.cuda.synchronize()
    summary = _build_summary(records, system, events, model, executor, cfg, args.kind, Path(args.trace_manifest).name, origin)
    # Final state and request accounting gates are written before any plot.
    final = {
        "active_w4_layers": list(getattr(executor, "active_layers", [])) if executor else [],
        "pending_layer_events": sorted(int(x) for x in getattr(model, "layer_transfer_events", {})),
        "executor_poisoned": bool(executor.poisoned) if executor else False,
        "controller_poisoned": bool(coordinator.poisoned) if coordinator else False,
        "kv_capacity_blocks": int(model.gpu_block_manager.num_blocks),
        "kv_free_blocks": _int(model.gpu_block_manager.num_free_blocks),
    }
    gate = {
        "all_submitted_ids_once": len(records) == len(specs) == len({r["request_id"] for r in records}),
        "all_completed_exact_512": all((r["error"] is not None and r["timeout"]) or (r["error"] is None and r["generated_token_count"] == (8 if args.preflight else 512)) for r in records),
        "raw_system_telemetry_present": bool(system),
        "controller_telemetry_present": (not executor) or any(e["type"] == "controller_sample" or e["type"] in ("pressure_trigger", "recovery_trigger") for e in events),
        "morphing_observed": (not executor) or any(e["type"] == "pressure_trigger" and e["controller_event"]["command"]["layer_delta"] > 0 for e in events),
        "recovery_observed": (not executor) or any(e["type"] == "recovery_trigger" and e["controller_event"]["command"]["layer_delta"] < 0 for e in events),
        "final_no_pending_layer_events": not final["pending_layer_events"],
        "final_executor_not_poisoned": not final["executor_poisoned"] and not final["controller_poisoned"],
        "summary_matches_record_count": summary["request_count"] == len(records),
    }
    if args.preflight and executor:
        gate.update({
            "preflight_real_w4_module_observed": any("WQLinear_GEMM" in classes for e in events for classes in e.get("real_w4_module_classes", {}).values()),
            "preflight_physical_kv_expansion_observed": any(any(item[0] == "expand" for item in e.get("executor_log_delta", [])) for e in events),
            "preflight_recovery_order_observed": any(any(item[0] == "shrink" for item in e.get("executor_log_delta", [])) for e in events),
            "preflight_clean_final_state": final["active_w4_layers"] == [] and final["kv_free_blocks"] == final["kv_capacity_blocks"],
        })
    metadata = {
        "schema_version": 2, "classification": cfg["classification"], "preflight": args.preflight,
        "configuration": {"kind": args.kind, "mode": args.mode, "config": cfg, "setup": {k: str(v) for k, v in setup.items() if k not in ("backups", "packed", "packed_gpu")}},
        "model": {"path": str(Path(args.fp16_model).resolve()), "snapshot": cfg["model_snapshot"], "w4_artifact": str(Path(args.w4_model).resolve())},
        "payload": {"path": str(Path(args.payload).resolve()), "sha256": payload["payload_sha256"], "token_count": payload["token_count"]},
        "trace_manifest": str(Path(args.trace_manifest).resolve()), "request_count": len(specs),
        "warmup_excluded": True, "generation_policy": {"prompt_tokens": 1024 if not args.preflight else 32, "output_tokens": 512 if not args.preflight else 8, "stop_on_eos": False},
        "summary": summary, "final": final, "gate": gate, "passed": all(gate.values()),
    }
    _write_json(out / "raw_requests.jsonl", {}) if False else None
    (out / "raw_requests.jsonl").write_text("".join(json.dumps(_jsonable(r), sort_keys=True) + "\n" for r in records))
    (out / "system_telemetry.jsonl").write_text("".join(json.dumps(_jsonable(s), sort_keys=True) + "\n" for s in system))
    _write_json(out / "controller-events.json", events)
    _write_json(out / "summary.json", summary)
    _write_json(out / "run-metadata.json", metadata)
    _write_json(out / "metrics.json", metadata)
    print(json.dumps({"passed": metadata["passed"], "gate": gate, "summary": summary, "final": final}, indent=2))
    return 0 if metadata["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
