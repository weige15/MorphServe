"""Measure model-level FP16/W4 prefill and one-token decode latency."""

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


def timed(fn, repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return samples


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((p / 100) * len(ordered)))]


def run(
    model_path: Path,
    quantized_layer_count: int,
    prompt_token_count: int,
    repeats: int,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_ids = tokenizer(
        "The capital of France is Paris. The capital of Germany is",
        return_attention_mask=False,
    )["input_ids"]
    prompt_ids = (base_ids * ((prompt_token_count + len(base_ids) - 1) // len(base_ids)))[:prompt_token_count]
    config = EngineConfig(
        model_path=str(model_path),
        use_dummy=False,
        block_size=16,
        gpu_mem_utilization=0.90,
        num_cpu_blocks=1,
        max_seqs_in_block_table=4,
        max_blocks_per_seq=3072,
        max_batch_size=1,
        max_tokens_in_batch=prompt_token_count,
        quantized_layer_count=quantized_layer_count,
    )
    torch.cuda.empty_cache()
    model = LlamaModel(config)
    model.load_weights()
    num_gpu_blocks = model.profile_num_blocks()
    model.init_kvcache_and_swap(num_gpu_blocks)

    def prefill() -> None:
        model.forward([prompt_ids], [0], [])

    prefill()
    model.free_seqs_resources([0])

    def decode() -> None:
        # Prefill allocates sequence 0; the final token is deliberately fixed
        # so every condition measures the same one-token decode operation.
        model.forward([[prompt_ids[-1]]], [0], [prompt_token_count + 1])

    prefill()
    decode()
    model.free_seqs_resources([0])
    prefill_samples = []
    for _ in range(repeats):
        prefill_samples.extend(timed(prefill, 1))
        model.free_seqs_resources([0])
    decode_samples = []
    for _ in range(repeats):
        prefill()
        decode_samples.extend(timed(decode, 1))
        model.free_seqs_resources([0])
    result = {
        "schema_version": 1,
        "gpu": torch.cuda.get_device_name(),
        "quantized_layer_count": quantized_layer_count,
        "quantization": "fp16" if quantized_layer_count == 0 else "nf4_bitsandbytes_w4",
        "prompt_token_count": prompt_token_count,
        "repeats": repeats,
        "num_gpu_blocks": num_gpu_blocks,
        "prefill_ms": prefill_samples,
        "decode_one_token_ms": decode_samples,
        "prefill_median_ms": percentile(prefill_samples, 50),
        "decode_one_token_median_ms": percentile(decode_samples, 50),
        "prefill_p95_ms": percentile(prefill_samples, 95),
        "decode_one_token_p95_ms": percentile(decode_samples, 95),
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--quantized-layer-count", type=int, choices=(0, 8, 16, 32), required=True)
    parser.add_argument("--prompt-token-count", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.model_path, args.quantized_layer_count, args.prompt_token_count, args.repeats)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
