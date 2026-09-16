#!/usr/bin/env python3
"""Switch one normalized candidate layer FP16 -> real AutoAWQ W4 -> FP16."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from morphserve.autoawq_adapter import (
    backup_fp16_layer,
    install_autoawq_layer,
    load_packed_layer,
    release_fp16_layer,
    restore_fp16_layer,
)


def comparison(reference, candidate):
    reference = reference.float()
    candidate = candidate.float()
    delta = candidate - reference
    return {
        "exact": bool(torch.equal(reference, candidate)),
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "relative_l2": float(delta.norm() / reference.norm().clamp_min(1e-12)),
        "top1_reference": int(reference.argmax()),
        "top1_candidate": int(candidate.argmax()),
        "top5_reference": [int(x) for x in reference.topk(5).indices],
        "top5_candidate": [int(x) for x in candidate.topk(5).indices],
    }


def maps_exact(tensors, tensor_map, buffer):
    checks = []
    for tensor, entry in zip(tensors, tensor_map):
        name, info = next(iter(entry.items()))
        actual = tensor.detach().view(torch.uint8).reshape(-1).cpu()
        expected = buffer[info["offset"]:info["offset"] + info["size"]]
        checks.append({"name": name, "bytes": info["size"], "exact": bool(torch.equal(actual, expected))})
    return checks


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
    token_ids = tokenizer(args.prompt, return_tensors="pt").input_ids[0]
    config = EngineConfig(
        org_model_path=args.fp16_model,
        use_dummy=False,
        num_cpu_blocks=0,
        max_seqs_in_block_table=8,
        max_blocks_per_seq=2048,
        max_batch_size=8,
        max_tokens_in_batch=2048,
    )
    model = LlamaModel(config, args.fp16_model)
    model.load_weights()

    def logits():
        result = model.forward([token_ids.tolist()], [0], [], ignore_kvcache=True, return_logits=True)[0]
        torch.cuda.synchronize()
        return result.detach().cpu()

    fp16_initial = logits()
    fp16_repeat = logits()
    initial_repeat = comparison(fp16_initial, fp16_repeat)
    gpu_bytes_before = torch.cuda.memory_allocated()

    backup = backup_fp16_layer(model, args.layer, swiftllm_c)
    packed = load_packed_layer(args.w4_model, args.layer, swiftllm_c)
    if packed["size"] > backup["size"]:
        raise RuntimeError(f"packed layer {packed['size']} exceeds FP16 region {backup['size']}")
    release_fp16_layer(model, args.layer)
    copy_start = time.perf_counter()
    quant_tensors = swiftllm_c.replace_layer_org2quant(args.layer)
    quant_copy_wall_ms = (time.perf_counter() - copy_start) * 1000
    torch.cuda.synchronize()
    quant_checks = maps_exact(quant_tensors, packed["tensor_map"], packed["buffer"])
    quant_base = min(tensor.data_ptr() for tensor in quant_tensors)
    quant_end = max(tensor.data_ptr() + tensor.numel() * tensor.element_size() for tensor in quant_tensors)
    wrapper, modules = install_autoawq_layer(
        model, args.layer, quant_tensors, packed["tensor_map"]
    )
    real_module_count = len(modules)
    real_module_classes = [type(module).__name__ for module in modules]
    model.layer_quant_list = [args.layer]
    model.is_layer_quant_list[args.layer] = True
    gpu_bytes_after_install = torch.cuda.memory_allocated()

    mixed_first = logits()
    mixed_second = logits()
    mixed_repeat = comparison(mixed_first, mixed_second)
    mixed_vs_fp16 = comparison(fp16_initial, mixed_second)

    model.transformer_layers[args.layer] = None
    model.layer_quant_list = []
    model.is_layer_quant_list[args.layer] = False
    del wrapper, modules, quant_tensors
    gc.collect()
    torch.cuda.synchronize()
    restore_start = time.perf_counter()
    restored_tensors = swiftllm_c.replace_layer_quant2org(args.layer)
    restore_copy_wall_ms = (time.perf_counter() - restore_start) * 1000
    torch.cuda.synchronize()
    restore_checks = maps_exact(restored_tensors, backup["tensor_map"], backup["buffer"])
    restored_base = min(tensor.data_ptr() for tensor in restored_tensors)
    restore_fp16_layer(model, args.layer, restored_tensors, backup["tensor_map"])
    restored_logits = logits()
    restored_vs_initial = comparison(fp16_initial, restored_logits)

    packed_storage = {}
    for tensor, entry in zip(quant_checks, packed["tensor_map"]):
        name, info = next(iter(entry.items()))
        suffix = name.rsplit(".", 1)[-1]
        packed_storage[suffix] = packed_storage.get(suffix, 0) + info["size"]
    gate = {
        "initial_fp16_repeat_exact": initial_repeat["exact"],
        "quant_source_exact": all(row["exact"] for row in quant_checks),
        "quant_within_fp16_region": quant_base == backup["base"] and quant_end <= backup["base"] + backup["size"],
        "real_wqlinear_modules": real_module_count == 7 and set(real_module_classes) == {"WQLinear_GEMM"},
        "mixed_repeat_within_envelope": mixed_repeat["relative_l2"] < 0.005,
        "mixed_topk_stable": mixed_repeat["top1_reference"] == mixed_repeat["top1_candidate"] and mixed_repeat["top5_reference"] == mixed_repeat["top5_candidate"],
        "mixed_differs_from_fp16": not mixed_vs_fp16["exact"],
        "restore_source_exact": all(row["exact"] for row in restore_checks),
        "restore_base_stable": restored_base == backup["base"],
        "restored_logits_exact": restored_vs_initial["exact"],
    }
    payload = {
        "schema_version": 1,
        "classification": "modified-condition-one-layer-autoawq-switch",
        "layer_id": args.layer,
        "prompt": args.prompt,
        "token_ids": token_ids.tolist(),
        "device": torch.cuda.get_device_name(),
        "storage": {
            "fp16_region_bytes": backup["size"],
            "packed_layer_bytes_including_metadata_and_norms": packed["size"],
            "reclaimed_bytes": backup["size"] - packed["size"],
            "packed_by_suffix": packed_storage,
            "gpu_allocated_before": gpu_bytes_before,
            "gpu_allocated_after_install": gpu_bytes_after_install,
            "gpu_allocator_delta": gpu_bytes_after_install - gpu_bytes_before,
            "fp16_base": backup["base"],
            "quant_base": quant_base,
            "quant_end": quant_end,
            "restored_base": restored_base,
        },
        "copies": {"quant_wall_ms": quant_copy_wall_ms, "restore_wall_ms": restore_copy_wall_ms},
        "quant_tensor_audit": {"checked": len(quant_checks), "mismatches": [row for row in quant_checks if not row["exact"]]},
        "restore_tensor_audit": {"checked": len(restore_checks), "mismatches": [row for row in restore_checks if not row["exact"]]},
        "initial_fp16_repeat": initial_repeat,
        "mixed_repeat": mixed_repeat,
        "mixed_vs_fp16": mixed_vs_fp16,
        "restored_vs_initial": restored_vs_initial,
        "gate": gate,
        "passed": all(gate.values()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gate": gate, "storage": payload["storage"], "mixed_repeat": mixed_repeat, "mixed_vs_fp16": mixed_vs_fp16, "restored_vs_initial": restored_vs_initial}, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
