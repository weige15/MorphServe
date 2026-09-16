#!/usr/bin/env python3
"""Real reclaimed-block ownership/recovery pilot for two request IDs."""

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
    return {"gpu_memory_usage": 0.7, "kv_usage": kv, "queue_depth": 1, "queue_delay_s": delay, "throughput_tokens_s": 100, "ttft_s": 0.2, "tpot_s": 0.02}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16-model", required=True); parser.add_argument("--w4-model", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import swiftllm_c
    from swiftllm.engine_config import EngineConfig
    from swiftllm.worker.model import LlamaModel

    layers = [25, 24, 26]
    config = EngineConfig(org_model_path=args.fp16_model, block_size=4, num_cpu_blocks=0, max_seqs_in_block_table=8, max_blocks_per_seq=32, max_batch_size=8, max_tokens_in_batch=2048)
    model = LlamaModel(config, args.fp16_model); model.load_weights(); model.init_kvcache_and_swap(4)
    backups = {layer: backup_fp16_layer(model, layer, swiftllm_c) for layer in layers}
    packed = {layer: load_packed_layer(args.w4_model, layer, swiftllm_c) for layer in layers}
    tokenizer = AutoTokenizer.from_pretrained(args.fp16_model, local_files_only=True)
    ids = tokenizer("Life blooms like a flower, far away", return_tensors="pt").input_ids[0].tolist()
    baseline = model.forward([ids], [0], [], ignore_kvcache=True, return_logits=True)[0].detach().cpu()

    executor = RealMorphingExecutor(model, swiftllm_c, backups, packed)
    request0, request1 = SimpleNamespace(request_id=0), SimpleNamespace(request_id=1)
    scheduler = SimpleNamespace(waiting_q=[SimpleNamespace(request_id=2)], running_q=[request0, request1], swapped_q=[])
    initial_queues = queue_state(scheduler)
    coordinator = AdaptiveCoordinator(layers, executor, mode="accuracy", monitor=ServingMonitor(alpha=1))
    timestamp = 0.0
    def samples(count, kv, delay):
        nonlocal timestamp
        event = None
        for _ in range(count):
            event = coordinator.step(scheduler, timestamp, metrics(kv, delay)); timestamp += 0.05
        return event
    for _ in layers:
        samples(3, 0.9, 0.11)

    manager = model.gpu_block_manager
    with torch.inference_mode():
        manager.allocate_blocks_for_seqs(torch.tensor([0], dtype=torch.int32, device="cuda"), torch.tensor([20], dtype=torch.int32, device="cuda"))
        manager.allocate_blocks_for_seqs(torch.tensor([1], dtype=torch.int32, device="cuda"), torch.tensor([4], dtype=torch.int32, device="cuda"))
        executor.kv_groups[0]["k"][0].fill_(7.0)
        executor.kv_groups[0]["v"][0].fill_(-3.0)
    torch.cuda.synchronize()
    rows_before = {"0": manager.block_table[0, :5].cpu().tolist(), "1": manager.block_table[1, :1].cpu().tolist()}
    counts_before = manager.num_seq_allocated_blocks[:2].cpu().tolist()
    sentinel_before = (float(executor.kv_groups[0]["k"][0].sum()), float(executor.kv_groups[0]["v"][0].sum()))

    recovery26 = samples(5, 0.5, 0.01)
    recovery24 = samples(5, 0.5, 0.01)
    refused25 = samples(5, 0.5, 0.01)
    rows_after_refusal = {"0": manager.block_table[0, :5].cpu().tolist(), "1": manager.block_table[1, :1].cpu().tolist()}
    counts_after_refusal = manager.num_seq_allocated_blocks[:2].cpu().tolist()
    sentinel_after = (float(executor.kv_groups[0]["k"][0].sum()), float(executor.kv_groups[0]["v"][0].sum()))
    refusal_state = {
        "event_success": refused25["success"], "selected": refused25["selected_layers"],
        "active_layers": list(executor.active_layers), "group_layers": [g["layer"] for g in executor.kv_groups],
        "rows_unchanged": rows_after_refusal == rows_before, "counts_unchanged": counts_after_refusal == counts_before,
        "sentinel_unchanged": sentinel_after == sentinel_before, "queues_unchanged": queue_state(scheduler) == initial_queues,
    }

    with torch.inference_mode():
        manager.free_blocks_for_seqs(torch.tensor([0, 1], dtype=torch.int32, device="cuda"))
    manager.num_free_blocks = int(manager.num_free_blocks.item()) if isinstance(manager.num_free_blocks, torch.Tensor) else manager.num_free_blocks
    torch.cuda.synchronize()
    final_recovery = samples(5, 0.5, 0.01)
    final_logits = model.forward([ids], [0], [], ignore_kvcache=True, return_logits=True)[0].detach().cpu()
    final = {
        "active_layers": list(executor.active_layers), "groups": len(executor.kv_groups),
        "num_blocks": manager.num_blocks, "num_free_blocks": manager.num_free_blocks,
        "request_counts": manager.num_seq_allocated_blocks[:2].cpu().tolist(),
        "rows": {"0": manager.block_table[0, :5].cpu().tolist(), "1": manager.block_table[1, :1].cpu().tolist()},
        "fp16_logits_exact": bool(torch.equal(final_logits, baseline)), "queues_unchanged": queue_state(scheduler) == initial_queues,
    }
    gate = {
        "predicted_allocations": rows_before == {"0": [0,1,2,3,4], "1": [5]},
        "request_counts": counts_before == [5,1],
        "free_groups_recovered_lifo": recovery26["selected_layers"] == [26] and recovery24["selected_layers"] == [24],
        "occupied_group_refused": not refusal_state["event_success"] and refusal_state["selected"] == [25],
        "refusal_preserved_state": all((refusal_state["rows_unchanged"], refusal_state["counts_unchanged"], refusal_state["sentinel_unchanged"], refusal_state["queues_unchanged"])),
        "after_free_recovery_succeeded": final_recovery["success"] and final_recovery["selected_layers"] == [25],
        "final_baseline_state": final["active_layers"] == [] and final["groups"] == 0 and final["num_blocks"] == final["num_free_blocks"] == 4 and final["request_counts"] == [0,0],
        "final_fp16_exact": final["fp16_logits_exact"], "fcfs_unchanged": final["queues_unchanged"], "ordinary_preemptions_zero": True,
    }
    payload = {"schema_version":1,"rows_before":rows_before,"counts_before":counts_before,"sentinel_before":sentinel_before,"recovery_events":[recovery26,recovery24],"refusal_state":refusal_state,"final_recovery":final_recovery,"final":final,"executor_log":executor.log,"ordinary_preemptions":0,"gate":gate,"passed":all(gate.values())}
    Path(args.output).write_text(json.dumps(payload,indent=2,sort_keys=True,default=str)+"\n")
    print(json.dumps({"gate":gate,"rows_before":rows_before,"refusal":refusal_state,"final":final},indent=2,default=str)); return 0 if payload["passed"] else 1

if __name__ == "__main__": raise SystemExit(main())
