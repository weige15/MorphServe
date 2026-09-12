"""Probe one static condition's memory, KV capacity, and short generation."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer

from swiftllm.engine_config import EngineConfig
from swiftllm.worker.model import LlamaModel


def run(
    model_path: Path,
    quantized_layer_count: int,
    prompt: str,
    max_new_tokens: int,
    max_batch_size: int,
    max_tokens_in_batch: int,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompt_ids = tokenizer(prompt, return_attention_mask=False)["input_ids"]
    config = EngineConfig(
        model_path=str(model_path),
        use_dummy=False,
        block_size=16,
        gpu_mem_utilization=0.99,
        num_cpu_blocks=1,
        max_seqs_in_block_table=4,
        max_blocks_per_seq=3072,
        max_batch_size=max_batch_size,
        max_tokens_in_batch=max_tokens_in_batch,
        quantized_layer_count=quantized_layer_count,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    model = LlamaModel(config)
    model.load_weights()
    torch.cuda.synchronize()
    load_ms = (time.perf_counter() - load_start) * 1000
    resident_weight_bytes = torch.cuda.memory_allocated()
    resident_reserved_bytes = torch.cuda.memory_reserved()

    profile_start = time.perf_counter()
    num_gpu_blocks = model.profile_num_blocks()
    torch.cuda.synchronize()
    profile_ms = (time.perf_counter() - profile_start) * 1000
    profile_peak_bytes = torch.cuda.max_memory_allocated()
    after_profile_bytes = torch.cuda.memory_allocated()

    model.init_kvcache_and_swap(num_gpu_blocks)
    with torch.inference_mode():
        logits = model.forward([prompt_ids], [0], [], return_logits=True)[0]
        generated = [int(torch.argmax(logits).item())]
        for _ in range(1, max_new_tokens):
            if generated[-1] in (2, 128001):
                break
            logits = model.forward(
                [[generated[-1]]],
                [0],
                [len(prompt_ids) + len(generated)],
                return_logits=True,
            )[0]
            generated.append(int(torch.argmax(logits).item()))
    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    model.free_seqs_resources([0])
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "schema_version": 1,
        "quantized_layer_count": quantized_layer_count,
        "quantization": "fp16" if quantized_layer_count == 0 else "nf4_bitsandbytes_w4",
        "gpu": torch.cuda.get_device_name(),
        "engine_config": {
            "max_batch_size": max_batch_size,
            "max_tokens_in_batch": max_tokens_in_batch,
            "block_size": 16,
        },
        "load_ms": load_ms,
        "profile_ms": profile_ms,
        "resident_weight_bytes_after_load": resident_weight_bytes,
        "resident_reserved_bytes_after_load": resident_reserved_bytes,
        "profile_peak_allocated_bytes": profile_peak_bytes,
        "allocated_bytes_after_profile": after_profile_bytes,
        "num_gpu_blocks": num_gpu_blocks,
        "gpu_kv_token_slots": 16 * num_gpu_blocks,
        "prompt": prompt,
        "prompt_token_ids": prompt_ids,
        "output_token_ids": generated,
        "decoded": decoded,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--quantized-layer-count", type=int, choices=(0, 8, 16, 32), required=True)
    parser.add_argument("--prompt", default="The capital of France is Paris. The capital of Germany is")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--max-tokens-in-batch", type=int, default=49152)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.model_path,
        args.quantized_layer_count,
        args.prompt,
        args.max_new_tokens,
        args.max_batch_size,
        args.max_tokens_in_batch,
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
