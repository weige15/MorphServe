import torch


def linear(
    a: torch.Tensor,  # [a, b]
    w,                # [c, b] FP16 tensor or QuantizedMatrix
) -> torch.Tensor:   # [a, c]
    # pylint: disable=not-callable
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
