# One-layer real AutoAWQ in-place switch result

## Result

All predeclared gates passed for layer 31 on local Llama 3.1 8B:

- 23 packed source tensors were byte-exact after copy into the FP16 layer region;
- seven attention/MLP modules executed as real AutoAWQ `WQLinear_GEMM`;
- quantized and restored views reused the exact FP16 base address;
- GPU allocator delta at install was 0 bytes (no duplicate GPU model/layer allocation);
- two W4 forwards had stable top-1/top-5 and repeat relative L2 0.001061, below the independently fixed 0.005 split-K envelope;
- mixed logits differed materially from FP16 (relative L2 0.04193), proving low-bit execution affected computation;
- 8/8 FP16 reconstructed tensors were exact, and restored logits were bit-exact to initial FP16.

## Actual storage

- FP16 region: 436,224,000 bytes.
- Packed layer including norms/qweight/qzeros/scales: 113,328,128 bytes.
- Physically reusable tail: 322,895,872 bytes (74.02%).
- Metadata was counted: qzeros 851,968 bytes and scales 3,407,872 bytes.

## Timing evidence and deviation

Blocking wall time around the candidate C++ operation was 15.67 ms for the 113.3-MB W4 copy/reconstruction tensors and 58.55 ms for 436.2-MB FP16 restore. This is substantially slower than the paper's approximate Llama 2 7B PCIe Gen4 examples (4/16 ms transfer, ~6 ms complete W4 swap), on different RTX 3090 hardware/model with stream creation and synchronization exposed. It is a modified-condition negative timing comparison, not a reproduction of paper latency or overlap.

## Classification

**Approximate/modified-condition mechanism reproduction.** Real packed one-layer W4 switching/restoration and physical tail reclamation are demonstrated. This adapter is independent and uses the local AutoAWQ checkpoint; it is not execution of the candidate's mismatched llm-awq loader. KV state, mid-request history, concurrency/overlap, multiple layers, and controller behavior remain open.

## Evidence

- `results/metrics.json`, `verification.log`
- `results/run.{stdout,stderr}.log`, `run.exitcode`
- `runtime/morphserve/autoawq_adapter.py`
