import torch


def linear(
    a: torch.Tensor,  # [a, b]
    w,                # [c, b] FP16 tensor or QuantizedMatrix
) -> torch.Tensor:   # [a, c]
    # pylint: disable=not-callable
    if hasattr(w, "quant_state"):
        # Import lazily so the unchanged FP16 path has no bitsandbytes import
        # requirement. The same NF4 W4 path is used for all selected layers.
        import bitsandbytes as bnb

        # bitsandbytes 0.49's GEMV fast path assumes the opposite matrix
        # orientation for a one-row input. Decode and call F.linear explicitly
        # in that case; batched prefill uses the normal low-bit matmul path.
        if a.numel() == a.shape[-1]:
            import bitsandbytes.functional as bnb_functional

            dequantized = bnb_functional.dequantize_4bit(
                w.data,
                w.quant_state,
            ).to(dtype=a.dtype)
            return torch.nn.functional.linear(a, dequantized.t())
        return bnb.matmul_4bit(
            a,
            w.data,
            bias=None,
            quant_state=w.quant_state,
        )
    # torch.nn.functional.linear selects the appropriate GEMM/GEMV path for
    # the FP16 weights used by the zero-layer control.
    return torch.nn.functional.linear(a, w)
