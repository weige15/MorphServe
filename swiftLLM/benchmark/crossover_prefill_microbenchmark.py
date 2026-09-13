"""Benchmark batched prefill across the pre-registered crossover GEMM-M range."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import time

import torch
from transformers import AutoTokenizer

from swiftllm.engine_config import EngineConfig
from swiftllm.worker.model import LlamaModel

from .metrics import percentile


DEFAULT_SEQUENCE_COUNTS = (1, 3, 4, 6, 7, 8, 9, 16, 24, 32)


def exact_prompt_ids(tokenizer, token_count: int) -> list[int]:
    seed = tokenizer(
        "The capital of France is Paris. The capital of Germany is",
        return_attention_mask=False,
    )["input_ids"]
    return (seed * ((token_count + len(seed) - 1) // len(seed)))[:token_count]


def run(args: argparse.Namespace) -> dict:
    if args.quantized_layer_count not in (0, 16):
        raise ValueError("crossover diagnosis only permits FP16 or AWQ W4-16")
    if args.quantized_layer_count == 16 and (
        args.quantization_backend != "awq_marlin" or args.quantized_model_path is None
    ):
        raise ValueError("W4-16 requires the validated AWQ-Marlin checkpoint")
    if args.quantized_layer_count == 0 and args.quantization_backend != "nf4_bitsandbytes":
        raise ValueError("FP16 must use the unchanged default control configuration")
    if args.prompt_token_count != 1024:
        raise ValueError("the crossover protocol fixes each prompt at 1024 tokens")
    if tuple(args.sequence_counts) != tuple(sorted(set(args.sequence_counts))):
        raise ValueError("sequence counts must be unique and ascending")
    if args.sequence_counts[-1] > 32:
        raise ValueError("sequence count exceeds the unchanged serving max batch size")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    prompt_ids = exact_prompt_ids(tokenizer, args.prompt_token_count)
    config = EngineConfig(
        model_path=str(args.model_path),
        use_dummy=False,
        block_size=16,
        gpu_mem_utilization=0.99,
        num_cpu_blocks=1,
        max_seqs_in_block_table=32,
        max_blocks_per_seq=3072,
        max_batch_size=32,
        max_tokens_in_batch=49152,
        quantized_layer_count=args.quantized_layer_count,
        quantization_backend=args.quantization_backend,
        quantized_model_path=(str(args.quantized_model_path) if args.quantized_model_path else None),
    )
    torch.cuda.empty_cache()
    model = LlamaModel(config)
    model.load_weights()
    resident_bytes = int(torch.cuda.memory_allocated())

    rows = []
    for sequence_count in args.sequence_counts:
        input_ids = [prompt_ids] * sequence_count
        seq_ids = list(range(sequence_count))

        def prefill() -> None:
            model.forward(input_ids, seq_ids, [], ignore_kvcache=True)

        for _ in range(args.warmup):
            prefill()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            start_ns = time.perf_counter_ns()
            prefill()
            torch.cuda.synchronize()
            samples.append((time.perf_counter_ns() - start_ns) / 1_000_000)
        rows.append(
            {
                "sequence_count": sequence_count,
                "prompt_tokens_per_sequence": args.prompt_token_count,
                "total_prefill_tokens": sequence_count * args.prompt_token_count,
                "effective_gemm_m": sequence_count * args.prompt_token_count,
                "warmup_count": args.warmup,
                "sample_count": len(samples),
                "samples_ms": samples,
                "mean_ms": statistics.fmean(samples),
                "median_ms": percentile(samples, 50),
                "p95_ms": percentile(samples, 95),
                "min_ms": min(samples),
                "max_ms": max(samples),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            }
        )

    result = {
        "schema_version": 1,
        "condition": "fp16_0" if args.quantized_layer_count == 0 else "awq_w4_16",
        "precision_state": "FP16" if args.quantized_layer_count == 0 else "AWQ-Marlin W4-16",
        "gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "model_path": str(args.model_path),
        "quantized_model_path": str(args.quantized_model_path) if args.quantized_model_path else None,
        "quantization_backend": args.quantization_backend,
        "quantized_layer_count": args.quantized_layer_count,
        "resident_weight_bytes": resident_bytes,
        "measurement_scope": (
            "compute-only model forward with ignore_kvcache=True; identical decoder path and "
            "multi-sequence 1024-token batches, excluding KV-capacity effects"
        ),
        "serving_limits": {"max_batch_size": 32, "max_tokens_in_batch": 49152},
        "expected_trace_burst_sequence_counts": [1, 3, 4, 6, 7, 8, 9],
        "pre_registered_sequence_counts": list(args.sequence_counts),
        "results": rows,
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--quantized-layer-count", type=int, choices=(0, 16), required=True)
    parser.add_argument(
        "--quantization-backend",
        choices=("nf4_bitsandbytes", "awq_marlin"),
        default="nf4_bitsandbytes",
    )
    parser.add_argument("--quantized-model-path", type=Path)
    parser.add_argument("--prompt-token-count", type=int, default=1024)
    parser.add_argument("--sequence-counts", type=int, nargs="+", default=DEFAULT_SEQUENCE_COUNTS)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
