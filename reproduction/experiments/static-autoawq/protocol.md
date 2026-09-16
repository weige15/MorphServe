# Protocol: local real packed static AWQ W4 baseline

Status: pre-registered before dependency installation/model execution.

## Question

Can the available local Llama 3.1 8B AutoAWQ W4 G128 zero-point checkpoint execute real packed INT4 GEMMs on RTX 3090 and provide a deterministic fixed-W4 numerical baseline against the now-validated FP16 path?

## Frozen inputs

- FP16 snapshot `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`.
- W4 asset `/nfs/home/s314511048/.cache/morphserve/llama31-8b-autoawq-w4-g128-zp`, whose config declares AutoAWQ, 4 bits, group size 128, zero point, GEMM.
- AutoAWQ 0.2.9, Transformers 4.51.3, PyTorch 2.4.0+cu121; use AutoAWQ's Triton fallback if no compiled `awq_ext` is available.
- Same nine-token prompt/IDs as the FP16 baseline.
- No candidate morphing, KV cache, controller, or serving trace.

## Gate

1. Loaded linear modules are `WQLinear_GEMM` with INT32 `qweight/qzeros` and FP16 scales; no fake-quantized FP16 weights are presented as W4.
2. Config and actual packed tensor storage (including scales/zeros) are recorded.
3. Two W4 forwards produce exactly identical logits/top-k.
4. Logits are finite and their shape matches FP16.
5. Report W4-vs-FP16 top-1/top-5, relative L2, and max absolute error without requiring equality.
6. Process exits 0; model/config hashes and package versions are saved.

## Boundaries

This is a modified-condition static W4 mechanism baseline. It does not validate the candidate's unreleased llm-awq interface, in-place layer reconstruction, mixed precision, paper checkpoint, task quality, throughput, or latency. First-call compilation/warmup is excluded from performance claims.

## Pre-registered diagnostic follow-up after exact-repeat failure

The first run met every mechanism/numerical gate except bit-exact repeat logits. Source inspection shows AutoAWQ 0.2.9's small-input Triton path hard-codes `split_k_iters=8` and combines splits with `tl.atomic_add`, a plausible nondeterminism source. Keep the exact-repeat gate unchanged, run five W4 forwards, and record per-run max/relative difference and top-k stability. This follow-up quantifies the negative result; it does not convert the failed confirmatory gate into a pass.
