from functools import cache

import torch


@cache
def _awq_marlin_runtime():
    # Keep vLLM's compiled extension lazy so the validated FP16/NF4
    # environment does not need to import its torch-2.9 ABI.
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_awq_marlin_linear,
    )
    from vllm.scalar_type import scalar_types

    return apply_awq_marlin_linear, scalar_types.uint4


def linear(
    a: torch.Tensor,  # [a, b]
    w,                # [c, b] FP16 tensor or QuantizedMatrix
) -> torch.Tensor:   # [a, c]
    # pylint: disable=not-callable
    if getattr(w, "backend", None) == "awq_marlin":
        apply_awq_marlin_linear, quant_type = _awq_marlin_runtime()
        return apply_awq_marlin_linear(
            input=a,
            weight=w.qweight,
            weight_scale=w.scales,
            weight_zp=w.qzeros,
            g_idx=w.g_idx,
            g_idx_sort_indices=w.g_idx_sort_indices,
            workspace=w.workspace,
            quant_type=quant_type,
            output_size_per_partition=w.shape[0],
            input_size_per_partition=w.shape[1],
        )
    if hasattr(w, "gemv_quant_state"):
        # bitsandbytes' source-layout NF4 GEMV is the serving-critical
        # one-token path. It consumes packed bytes and scales directly; it
        # never materializes the complete FP16 matrix per generated token.
        import bitsandbytes as bnb
        import bitsandbytes.functional as bnb_functional

        if a.numel() == a.shape[-1]:
            return bnb_functional.gemv_4bit(
                a,
                w.gemv_data,
                state=w.gemv_quant_state,
            )
        # bitsandbytes 0.49's batched wrapper is the available prefill
        # fallback. Its multi-row implementation dequantizes internally; this
        # is intentionally reported by the microbenchmark and is not used for
        # single-token decode. A future fused low-bit GEMM can replace only
        # this branch without changing the model loader or decode gate.
        return bnb.matmul_4bit(
            a,
            w.matmul_data,
            bias=None,
            quant_state=w.matmul_quant_state,
        )
    # torch.nn.functional.linear selects the appropriate GEMM/GEMV path for
    # the FP16 weights used by the zero-layer control.
    return torch.nn.functional.linear(a, w)
