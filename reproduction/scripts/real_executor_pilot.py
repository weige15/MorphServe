#!/usr/bin/env python3
"""Transactional real-GPU executor/controller pilot."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from morphserve.autoawq_adapter import backup_fp16_layer, load_packed_layer
from morphserve.controller import ServingMonitor
from morphserve.integration import AdaptiveCoordinator, queue_state
from morphserve.real_executor import RealMorphingExecutor


def metrics(kv, delay):
    return {
        "gpu_memory_usage": 0.7,
        "kv_usage": kv,
        "queue_depth": 2,
        "queue_delay_s": delay,
        "throughput_tokens_s": 100,
        "ttft_s": 0.2,
        "tpot_s": 0.02,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16-model", required=True)
    parser.add_argument("--w4-model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import swiftllm_c
    from swiftllm.engine_config import EngineConfig
    from swiftllm.worker.model import LlamaModel

    layers = [25, 24, 26]
    config = EngineConfig(
        org_model_path=args.fp16_model,
        use_dummy=False,
        block_size=4,
        num_cpu_blocks=0,
        max_seqs_in_block_table=8,
        max_blocks_per_seq=32,
        max_batch_size=8,
        max_tokens_in_batch=2048,
    )
    model = LlamaModel(config, args.fp16_model)
    model.load_weights()
    model.init_kvcache_and_swap(4)
    manager = model.gpu_block_manager

    def allocator_state():
        free = manager.num_free_blocks
        return {
            "num_blocks": int(manager.num_blocks),
            "num_free_blocks": int(free.item()) if isinstance(free, torch.Tensor) else int(free),
            "is_block_free": manager.is_block_free.cpu().tolist(),
            "k_cache_new_len": len(model.k_cache_new),
            "v_cache_new_len": len(model.v_cache_new),
            "kv_cache_new_block_size": int(model.kv_cache_new_block_size),
        }

    initial_allocator = allocator_state()
    backups = {layer: backup_fp16_layer(model, layer, swiftllm_c) for layer in layers}
    packed = {layer: load_packed_layer(args.w4_model, layer, swiftllm_c) for layer in layers}
    tokenizer = AutoTokenizer.from_pretrained(args.fp16_model, local_files_only=True)
    ids = tokenizer("Life blooms like a flower, far away", return_tensors="pt").input_ids[0].tolist()
    baseline = model.forward([ids], [0], [], ignore_kvcache=True, return_logits=True)[0].detach().cpu()

    executor = RealMorphingExecutor(model, swiftllm_c, backups, packed, fail_expand_calls={2})
    scheduler = SimpleNamespace(waiting_q=[3, 4], running_q=[1, 2], swapped_q=[5])
    initial_queues = queue_state(scheduler)

    morph_started = executor.morph_to_w4(layers[:2])
    expand_completed = executor.expand_kv(layers[:2])
    restore_completed = executor.restore_fp16(list(reversed(layers[:2])))
    rollback_logits = model.forward([ids], [0], [], ignore_kvcache=True, return_logits=True)[0].detach().cpu()
    rollback_region_exact = {
        str(layer): bool(torch.equal(swiftllm_c.get_layer_memory_org_gpu(layer)[:backups[layer]["size"]].cpu(), backups[layer]["buffer"]))
        for layer in layers[:2]
    }
    after_failure = {
        "success": morph_started and expand_completed,
        "restore_success": restore_completed,
        "executor_layers": list(executor.active_layers),
        "kv_groups": len(executor.kv_groups),
        "num_blocks": model.gpu_block_manager.num_blocks,
        "allocator": allocator_state(),
        "queues_unchanged": queue_state(scheduler) == initial_queues,
        "fp16_logits_exact": bool(torch.equal(rollback_logits, baseline)),
        "fp16_region_bytes_exact": rollback_region_exact,
        "pending_layer_events": sorted(model.layer_transfer_events),
        "partial_group_was_acquired": any(row[0] == "expand" for row in executor.log),
    }
    executor.fail_expand_calls.clear()
    coordinator = AdaptiveCoordinator(layers, executor, mode="accuracy", monitor=ServingMonitor(alpha=1))
    timestamp = 0.0

    def pressure_action():
        nonlocal timestamp
        event = None
        for _ in range(3):
            event = coordinator.step(scheduler, timestamp, metrics(0.9, 0.11))
            timestamp += 0.05
        return event

    successful_events = [pressure_action() for _ in layers]
    active_state = {
        "coordinator_layers": list(coordinator.active_layers),
        "executor_layers": list(executor.active_layers),
        "group_layers": [group["layer"] for group in executor.kv_groups],
        "group_blocks": [int(group["k"].shape[0]) for group in executor.kv_groups],
        "num_blocks": model.gpu_block_manager.num_blocks,
        "num_free_blocks": model.gpu_block_manager.num_free_blocks,
        "explicit_kv_regions": model.explicit_kv_regions,
        "module_classes": {
            str(layer): sorted({type(module).__name__ for module in executor.quant_objects[layer][1]})
            for layer in layers
        },
        "k_addresses": [group["k"].data_ptr() for group in executor.kv_groups],
    }

    recovery_events = []
    for _ in layers:
        event = None
        for _ in range(5):
            event = coordinator.step(scheduler, timestamp, metrics(0.5, 0.01))
            timestamp += 0.05
        recovery_events.append(event)
    final = model.forward([ids], [0], [], ignore_kvcache=True, return_logits=True)[0].detach().cpu()
    final_exact = bool(torch.equal(final, baseline))
    final_region_exact = {
        str(layer): bool(torch.equal(swiftllm_c.get_layer_memory_org_gpu(layer)[:backups[layer]["size"]].cpu(), backups[layer]["buffer"]))
        for layer in layers
    }
    final_state = {
        "controller_layers": coordinator.controller.quantized_layers,
        "coordinator_layers": list(coordinator.active_layers),
        "executor_layers": list(executor.active_layers),
        "kv_groups": len(executor.kv_groups),
        "num_blocks": model.gpu_block_manager.num_blocks,
        "num_free_blocks": model.gpu_block_manager.num_free_blocks,
        "allocator": allocator_state(),
        "explicit_kv_regions": model.explicit_kv_regions,
        "queues_unchanged": queue_state(scheduler) == initial_queues,
        "fp16_logits_exact": final_exact,
        "fp16_region_bytes_exact": final_region_exact,
        "pending_layer_events": sorted(model.layer_transfer_events),
    }
    gate = {
        "injected_failure_rolled_back": not after_failure["success"] and after_failure["restore_success"] and after_failure["partial_group_was_acquired"] and after_failure["executor_layers"] == [] and after_failure["kv_groups"] == 0 and after_failure["allocator"] == initial_allocator and after_failure["fp16_logits_exact"] and all(after_failure["fp16_region_bytes_exact"].values()) and not after_failure["pending_layer_events"],
        "profile_order_active": active_state["executor_layers"] == layers == active_state["group_layers"],
        "real_w4_modules": all(classes == ["WQLinear_GEMM"] for classes in active_state["module_classes"].values()),
        "capacity_physically_expanded": active_state["num_blocks"] > 4 and active_state["num_free_blocks"] == active_state["num_blocks"],
        "explicit_mapping_selected": active_state["explicit_kv_regions"] is True,
        "recovery_lifo": [event["selected_layers"] for event in recovery_events] == [[26], [24], [25]],
        "final_state_restored": final_state["controller_layers"] == 0 and final_state["executor_layers"] == [] and final_state["kv_groups"] == 0 and final_state["allocator"] == initial_allocator,
        "fcfs_unchanged": after_failure["queues_unchanged"] and final_state["queues_unchanged"],
        "final_fp16_logits_exact": final_exact,
        "final_fp16_region_bytes_exact": all(final_region_exact.values()),
        "no_pending_layer_events": not final_state["pending_layer_events"],
    }
    payload = {
        "schema_version": 1,
        "profile_order": layers,
        "initial_num_blocks": 4,
        "initial_allocator": initial_allocator,
        "after_injected_failure": after_failure,
        "successful_events": successful_events,
        "active_state": active_state,
        "recovery_events": recovery_events,
        "final_state": final_state,
        "executor_log": executor.log,
        "gate": gate,
        "passed": all(gate.values()),
    }
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True, default=lambda value: list(value) if isinstance(value, tuple) else str(value)) + "\n")
    print(json.dumps({"gate": gate, "after_failure": after_failure, "active": active_state, "final": final_state}, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
