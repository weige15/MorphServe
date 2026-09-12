"""Validate W4 matrix numerics and low-bit prefill/decode dispatch."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import torch

from swiftllm.worker.kernels.linear import linear
from swiftllm.worker.weight import quantize_matrix


REPRESENTATIVE_SHAPES = {
    "q_or_o_projection": (4096, 4096),
    "k_or_v_projection": (1024, 4096),
    "up_or_gate_projection": (14336, 4096),
    "down_projection": (4096, 14336),
}


def _timed(fn, repeats: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / repeats


def check_matrix(name: str, shape: tuple[int, int], repeats: int) -> dict[str, Any]:
    out_features, in_features = shape
    weight = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.02
    quantized = quantize_matrix(weight, torch.float16)
    rows: dict[str, Any] = {}
    for label, batch in (("decode", 1), ("prefill", 32)):
        inputs = torch.randn(batch, in_features, device="cuda", dtype=torch.float16)
        reference = torch.nn.functional.linear(inputs, weight)
        actual = linear(inputs, quantized)
        finite = bool(torch.isfinite(actual).all().item())
        relative_l2 = float(
            (actual.float() - reference.float()).norm()
            / reference.float().norm().clamp_min(1e-8)
        )
        max_abs = float((actual.float() - reference.float()).abs().max())
        fp16_ms = _timed(
            lambda: torch.nn.functional.linear(inputs, weight), repeats, warmup=2
        )
        w4_ms = _timed(lambda: linear(inputs, quantized), repeats, warmup=2)
        rows[label] = {
            "batch_rows": batch,
            "finite": finite,
            "relative_l2_error": relative_l2,
            "max_abs_error": max_abs,
            "fp16_ms": fp16_ms,
            "w4_ms": w4_ms,
            "dispatch": "bitsandbytes.gemv_4bit" if batch == 1 else "bitsandbytes.matmul_4bit_prefill_fallback",
        }
    del weight, quantized
    torch.cuda.empty_cache()
    return {"shape": list(shape), "checks": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("W4 sanity requires CUDA")
    result = {
        "schema_version": 1,
        "device": torch.cuda.get_device_name(),
        "bits": 4,
        "quant_type": "nf4",
        "blocksize": 64,
        "matrices": [
            check_matrix(name, shape, args.repeats)
            | {"name": name}
            for name, shape in REPRESENTATIVE_SHAPES.items()
        ],
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
