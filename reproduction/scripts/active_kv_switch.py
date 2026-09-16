#!/usr/bin/env python3
"""Active-request same-history FP16/W4/FP16 state-preservation pilot."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from morphserve.autoawq_adapter import (
    backup_fp16_layer,
    install_autoawq_layer,
    load_packed_layer,
    materialize_packed_tensors,
    release_fp16_layer,
    restore_fp16_layer,
)


def compare(reference, candidate):
    reference = reference.float()
    candidate = candidate.float()
    delta = candidate - reference
    return {
        "exact": bool(torch.equal(reference, candidate)),
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(delta.norm() / reference.norm().clamp_min(1e-12)),
        "top1_reference": int(reference.flatten().argmax()) if reference.ndim == 1 else None,
        "top1_candidate": int(candidate.flatten().argmax()) if candidate.ndim == 1 else None,
    }


def snapshot_state(model):
    manager = model.gpu_block_manager
    return {
        "k": model.k_cache.clone(),
        "v": model.v_cache.clone(),
        "block_table": manager.block_table.clone(),
        "num_seq": manager.num_seq_allocated_blocks.clone(),
        "is_free": manager.is_block_free.clone(),
        "num_free": manager.num_free_blocks,
        "num_blocks": manager.num_blocks,
        "num_blocks_org": manager.num_blocks_org,
    }


def restore_state(model, state):
    with torch.inference_mode():
        model.k_cache.copy_(state["k"])
        model.v_cache.copy_(state["v"])
    manager = model.gpu_block_manager
    manager.block_table = state["block_table"].clone()
    manager.num_seq_allocated_blocks = state["num_seq"].clone()
    manager.is_block_free = state["is_free"].clone()
    manager.num_free_blocks = state["num_free"]
    manager.num_blocks = state["num_blocks"]
    manager.num_blocks_org = state["num_blocks_org"]
    model.k_cache_new = []
    model.v_cache_new = []
    model.kv_cache_new_block_size = 0


def maps_exact(tensors, tensor_map, buffer):
    rows = []
    for tensor, entry in zip(tensors, tensor_map):
        name, info = next(iter(entry.items()))
        actual = tensor.detach().view(torch.uint8).reshape(-1).cpu()
        expected = buffer[info["offset"]:info["offset"] + info["size"]]
        rows.append({"name": name, "exact": bool(torch.equal(actual, expected))})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16-model", required=True)
    parser.add_argument("--w4-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer", type=int, default=31)
    parser.add_argument("--prompt", default="Life blooms like a flower, far away")
    args = parser.parse_args()

    import swiftllm_c
    from swiftllm.engine_config import EngineConfig
    from swiftllm.worker.model import LlamaModel

    tokenizer = AutoTokenizer.from_pretrained(args.fp16_model, local_files_only=True)
    prompt_ids = tokenizer(args.prompt, return_tensors="pt").input_ids[0].tolist()
    config = EngineConfig(
        org_model_path=args.fp16_model,
        use_dummy=False,
        block_size=4,
        num_cpu_blocks=0,
        max_seqs_in_block_table=4,
        max_blocks_per_seq=32,
        max_batch_size=4,
        max_tokens_in_batch=2048,
    )
    model = LlamaModel(config, args.fp16_model)
    model.load_weights()
    model.init_kvcache_and_swap(4)

    def forward(input_token_ids, decoding_length=None):
        if decoding_length is None:
            value = model.forward([input_token_ids], [0], [], return_logits=True)[0]
        else:
            value = model.forward([[input_token_ids[-1]]], [0], [decoding_length], return_logits=True)[0]
        torch.cuda.synchronize()
        return value.detach().cpu()

    prefill_logits = forward(prompt_ids)
    first_token = int(prefill_logits.argmax())
    prefill_snapshot = snapshot_state(model)
    original_wrapper = model.transformer_layers[args.layer]
    backup = backup_fp16_layer(model, args.layer, swiftllm_c)
    packed = load_packed_layer(args.w4_model, args.layer, swiftllm_c)
    model.org_layer_param_size = backup["size"] // model.k_cache.element_size()

    # Same-history reference: separately allocated W4 tensors, original KV block 3.
    reference_tensors = materialize_packed_tensors(packed)
    reference_wrapper, reference_modules = install_autoawq_layer(
        model, args.layer, reference_tensors, packed["tensor_map"]
    )
    reference_inputs = [first_token]
    reference_logits = []
    reference_outputs = []
    for step in range(4):
        logits = forward(reference_inputs, len(prompt_ids) + step + 1)
        reference_logits.append(logits)
        token = int(logits.argmax())
        reference_outputs.append(token)
        reference_inputs.append(token)
    reference_block_k = model.k_cache[3].clone()
    reference_block_v = model.v_cache[3].clone()
    reference_block_table = model.gpu_block_manager.block_table[0, :4].clone()
    model.transformer_layers[args.layer] = original_wrapper
    del reference_wrapper, reference_modules, reference_tensors
    gc.collect()
    torch.cuda.empty_cache()
    final_reference_logits = forward(reference_inputs, len(prompt_ids) + 5)

    # Reset exactly to post-prefill state, then use in-place W4 + reclaimed KV.
    restore_state(model, prefill_snapshot)
    model.transformer_layers[args.layer] = original_wrapper
    prompt_k_before = model.k_cache[:3].clone()
    prompt_v_before = model.v_cache[:3].clone()
    manager = model.gpu_block_manager
    manager.num_blocks_org = 3
    manager.num_blocks = 3
    manager.num_free_blocks = 0
    manager.num_blocks_org = 3
    manager.num_free_blocks_org = 3
    manager.is_block_free = manager.is_block_free[:3].clone()

    release_fp16_layer(model, args.layer)
    quant_tensors = swiftllm_c.replace_layer_org2quant(args.layer)
    quant_audit = maps_exact(quant_tensors, packed["tensor_map"], packed["buffer"])
    system_wrapper, system_modules = install_autoawq_layer(
        model, args.layer, quant_tensors, packed["tensor_map"]
    )
    model.layer_quant_list = [args.layer]
    model.is_layer_quant_list[args.layer] = True
    k_new, v_new = swiftllm_c.acquire_new_kvcache(args.layer)
    model.k_cache_new = [k_new]
    model.v_cache_new = [v_new]
    model.kv_cache_new_block_size = k_new.shape[0]
    manager.is_block_free = torch.cat((
        manager.is_block_free,
        torch.ones(k_new.shape[0], dtype=torch.bool, device="cuda"),
    ))
    manager.num_blocks = 3 + k_new.shape[0]
    manager.num_free_blocks = k_new.shape[0]
    torch.cuda.synchronize()
    prompt_cache_unchanged_at_switch = bool(
        torch.equal(model.k_cache[:3], prompt_k_before)
        and torch.equal(model.v_cache[:3], prompt_v_before)
    )

    system_logits = []
    per_step = []
    for step, forced_input in enumerate(reference_inputs[:4]):
        logits = forward([forced_input], len(prompt_ids) + step + 1)
        system_logits.append(logits)
        item = compare(reference_logits[step], logits)
        item["forced_input"] = forced_input
        item["reference_output"] = reference_outputs[step]
        per_step.append(item)
    system_block_id = int(manager.block_table[0, 3].item())
    reclaimed_group_blocks = int(k_new.shape[0])
    block_k_comparison = compare(reference_block_k, k_new[0])
    block_v_comparison = compare(reference_block_v, v_new[0])

    # Preserve the occupied reclaimed block by migrating it to physical original block 3.
    migrated_k = k_new[0].clone()
    migrated_v = v_new[0].clone()
    with torch.inference_mode():
        model.k_cache[3].copy_(k_new[0], non_blocking=True)
        model.v_cache[3].copy_(v_new[0], non_blocking=True)
    swiftllm_c.record_layer_memory_use(args.layer)
    manager.num_blocks_org = 4
    manager.num_blocks = 4
    manager.num_free_blocks = 0
    manager.is_block_free = torch.zeros(4, dtype=torch.bool, device="cuda")
    model.k_cache_new = []
    model.v_cache_new = []
    model.kv_cache_new_block_size = 0
    model.transformer_layers[args.layer] = None
    model.layer_quant_list = []
    model.is_layer_quant_list[args.layer] = False
    del system_wrapper, system_modules, quant_tensors, k_new, v_new
    gc.collect()
    restored_tensors = swiftllm_c.replace_layer_quant2org(args.layer)
    restore_audit = maps_exact(restored_tensors, backup["tensor_map"], backup["buffer"])
    restore_fp16_layer(model, args.layer, restored_tensors, backup["tensor_map"])
    torch.cuda.synchronize()
    migration_exact = bool(
        torch.equal(model.k_cache[3], migrated_k)
        and torch.equal(model.v_cache[3], migrated_v)
    )
    final_system_logits = forward([reference_inputs[4]], len(prompt_ids) + 5)
    final_comparison = compare(final_reference_logits, final_system_logits)

    gate = {
        "prompt_cache_unchanged_at_switch": prompt_cache_unchanged_at_switch,
        "quant_source_exact": all(row["exact"] for row in quant_audit),
        "reclaimed_block_allocated": system_block_id == 3 and reclaimed_group_blocks > 0,
        "w4_steps_top1_match": all(row["top1_reference"] == row["top1_candidate"] for row in per_step),
        "w4_steps_within_envelope": all(row["relative_l2"] < 0.005 for row in per_step),
        "reclaimed_k_within_envelope": block_k_comparison["relative_l2"] < 0.005,
        "reclaimed_v_within_envelope": block_v_comparison["relative_l2"] < 0.005,
        "migration_exact": migration_exact,
        "block_table_preserved": int(manager.block_table[0, 3].item()) == 3,
        "restore_source_exact": all(row["exact"] for row in restore_audit),
        "final_top1_match": final_comparison["top1_reference"] == final_comparison["top1_candidate"],
        "final_within_envelope": final_comparison["relative_l2"] < 0.01,
    }
    payload = {
        "schema_version": 1,
        "classification": "modified-condition-active-kv-same-history",
        "prompt_ids": prompt_ids,
        "precision_schedule": ["FP16 prefill", "W4 layer 31 x4 decode", "FP16 x1 decode"],
        "forced_history": [first_token] + reference_outputs,
        "reference_block_table_after_w4": reference_block_table.cpu().tolist(),
        "system_block_table_after_migration": manager.block_table[0, :4].cpu().tolist(),
        "reclaimed_group_blocks": reclaimed_group_blocks,
        "system_allocated_block_id": system_block_id,
        "per_w4_step": per_step,
        "reclaimed_block_k": block_k_comparison,
        "reclaimed_block_v": block_v_comparison,
        "final_fp16": final_comparison,
        "quant_tensor_audit": quant_audit,
        "restore_tensor_audit": restore_audit,
        "gate": gate,
        "passed": all(gate.values()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gate": gate, "per_w4_step": per_step, "reclaimed_k": block_k_comparison, "reclaimed_v": block_v_comparison, "final": final_comparison}, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
