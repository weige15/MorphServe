#!/usr/bin/env python3
"""Three-layer real-W4 WikiText-2 MorphServe LIS pilot."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from morphserve.autoawq_adapter import (
    backup_fp16_layer,
    install_autoawq_layer,
    load_packed_layer,
    release_fp16_layer,
    restore_fp16_layer,
)
from morphserve.profiling import rank_layers


def cosine(left, right):
    return float(torch.nn.functional.cosine_similarity(left.float().flatten(), right.float().flatten(), dim=0))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16-model", required=True)
    parser.add_argument("--w4-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", default="29,30,31")
    args = parser.parse_args()
    layers = [int(value) for value in args.layers.split(",")]

    import swiftllm_c
    from swiftllm.engine_config import EngineConfig
    from swiftllm.worker.model import LlamaModel

    tokenizer = AutoTokenizer.from_pretrained(args.fp16_model, local_files_only=True)
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="train",
        download_mode="reuse_dataset_if_exists",
    )
    row_ids, texts = [], []
    for row_id, row in enumerate(dataset):
        text = row["text"].strip()
        if not text:
            continue
        row_ids.append(row_id)
        texts.append(text)
        joined = "\n\n".join(texts)
        ids = tokenizer(joined, add_special_tokens=True).input_ids
        if len(ids) >= 2048:
            token_ids = ids[:2048]
            break
    else:
        raise RuntimeError("WikiText-2 train text did not yield 2048 tokens")
    calibration_text = "\n\n".join(texts)

    config = EngineConfig(
        org_model_path=args.fp16_model,
        use_dummy=False,
        num_cpu_blocks=0,
        max_seqs_in_block_table=4,
        max_blocks_per_seq=4096,
        max_batch_size=4,
        max_tokens_in_batch=2048,
    )
    model = LlamaModel(config, args.fp16_model)
    model.load_weights()
    backups = {layer: backup_fp16_layer(model, layer, swiftllm_c) for layer in layers}
    packed = {layer: load_packed_layer(args.w4_model, layer, swiftllm_c) for layer in layers}
    current = set()
    active = {}
    copy_records = []
    restore_exact = []

    def set_precision(target):
        nonlocal current
        target = set(target)
        for layer in sorted(current - target, reverse=True):
            model.transformer_layers[layer] = None
            tensors_before = active.pop(layer)
            del tensors_before
            gc.collect()
            tensors = swiftllm_c.replace_layer_quant2org(layer)
            checks = []
            for tensor, entry in zip(tensors, backups[layer]["tensor_map"]):
                _, info = next(iter(entry.items()))
                actual = tensor.detach().view(torch.uint8).reshape(-1).cpu()
                expected = backups[layer]["buffer"][info["offset"]:info["offset"] + info["size"]]
                checks.append(bool(torch.equal(actual, expected)))
            restore_exact.extend(checks)
            restore_fp16_layer(model, layer, tensors, backups[layer]["tensor_map"])
        for layer in sorted(target - current):
            release_fp16_layer(model, layer)
            started = time.perf_counter()
            tensors = swiftllm_c.replace_layer_org2quant(layer)
            wall_ms = (time.perf_counter() - started) * 1000
            wrapper, modules = install_autoawq_layer(model, layer, tensors, packed[layer]["tensor_map"])
            active[layer] = (wrapper, modules, tensors)
            copy_records.append({"layer": layer, "wall_ms": wall_ms})
        current = target
        model.layer_quant_list = sorted(current)
        model.is_layer_quant_list = [index in current for index in range(model.model_config.num_layers)]

    def run(capture=()):
        captured = {}
        originals = {}
        for layer in capture:
            wrapper = model.transformer_layers[layer]
            original = wrapper.forward
            originals[layer] = original
            def traced(*call_args, _layer=layer, _original=original, **kwargs):
                input_embds, residual = call_args[0], call_args[1]
                before = (input_embds + residual).detach().cpu()
                result = _original(*call_args, **kwargs)
                after = (result + residual).detach().cpu()
                captured[_layer] = {"input": before, "output": after}
                return result
            wrapper.forward = traced
        try:
            logits = model.forward([token_ids], [0], [], ignore_kvcache=True, return_logits=True)[0]
            torch.cuda.synchronize()
            return logits.detach().cpu(), captured
        finally:
            for layer, original in originals.items():
                model.transformer_layers[layer].forward = original

    started = time.perf_counter()
    baseline_logits, baseline_states = run(layers)
    cache = {(): baseline_logits}
    quant_states = {}
    evaluated_sets = []
    for layer in layers:
        set_precision({layer})
        logits, states = run([layer])
        cache[(layer,)] = logits
        quant_states[layer] = states[layer]
        evaluated_sets.append([layer])
    lts = {layer: cosine(baseline_states[layer]["input"], baseline_states[layer]["output"]) for layer in layers}
    lrs = {layer: cosine(baseline_states[layer]["output"], quant_states[layer]["output"]) for layer in layers}
    mds_calls = []

    def mds(layer, quantized):
        before_key = tuple(sorted(quantized))
        after_key = tuple(sorted((*quantized, layer)))
        if after_key not in cache:
            set_precision(after_key)
            cache[after_key], _ = run()
            evaluated_sets.append(list(after_key))
        value = cosine(cache[before_key], cache[after_key])
        mds_calls.append({"layer": layer, "quantized_before": list(quantized), "set": list(after_key), "value": value})
        return value

    profile = rank_layers(lts, lrs, mds)
    set_precision(set())
    final_logits, _ = run()
    elapsed = time.perf_counter() - started
    final_exact = bool(torch.equal(final_logits, baseline_logits))
    gate = {
        "sequence_length_2048": len(token_ids) == 2048,
        "local_metrics_complete": set(lts) == set(layers) == set(lrs),
        "conditioned_mds_calls_6": len(mds_calls) == 6,
        "all_unique_sets_evaluated": len({tuple(item) for item in evaluated_sets}) == 6,
        "profile_steps_3": len(profile["steps"]) == 3 and sorted(profile["order"]) == sorted(layers),
        "all_restores_exact": all(restore_exact),
        "final_fp16_logits_exact": final_exact,
    }
    payload = {
        "schema_version": 1,
        "classification": "modified-condition-three-layer-real-lis-pilot",
        "calibration": {
            "dataset": "Salesforce/wikitext wikitext-2-raw-v1 train",
            "selection": "first 2048 tokenizer tokens from ordered non-empty rows",
            "seed": None,
            "row_ids": row_ids,
            "text_sha256": sha256_bytes(calibration_text.encode()),
            "token_ids": token_ids,
            "token_ids_sha256": sha256_bytes(json.dumps(token_ids).encode()),
            "tokenizer_snapshot": Path(args.fp16_model).resolve().name,
            "sequence_length": len(token_ids),
        },
        "representation": {
            "layer": "flattened float32 logical residual stream before/after full decoder layer",
            "model": "flattened float32 last-token vocabulary logits",
            "cosine_axis": "single flattened vector",
            "layer_indexing": "zero-based",
            "tie_break": "lowest_layer_index",
        },
        "candidate_layers": layers,
        "lts": lts,
        "lrs": lrs,
        "mds_calls": mds_calls,
        "evaluated_sets": evaluated_sets,
        "profile": profile,
        "copy_records": copy_records,
        "elapsed_seconds_including_model_forwards_and_copies": elapsed,
        "restored_tensor_checks": len(restore_exact),
        "gate": gate,
        "passed": all(gate.values()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"order": profile["order"], "lts": lts, "lrs": lrs, "mds_calls": mds_calls, "elapsed": elapsed, "gate": gate}, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
