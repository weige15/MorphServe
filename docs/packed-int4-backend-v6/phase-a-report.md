# Packed INT4 backend v6 — Phase A feasibility

Status: **PASS — proceed to minimal SwiftLLM integration**.

The candidate is the pre-registered proxy: AutoAWQ-compatible W4, group size
128, asymmetric zero point, FP16 scales/activations, converted once to the
installed vLLM 0.11.2 AWQ-Marlin representation. Tests used physical RTX 3090
GPU 5 (SM 8.6, 82 SMs) and real layer-0 tensors from the exact v5
Llama-3.1-8B checkpoint. The cheap packer applies AutoAWQ's final asymmetric
quantize/packing formula directly; activation-aware calibration is represented
by the separately generated full offline checkpoint, not claimed by this
matrix-only numerical test.

## Installed path and environment finding

The installed source paths and hashes are in
`benchmark-results/packed-int4-backend-v6/phase-a/backend-inventory.json`.
Static inspection verified:

- ordinary vLLM AWQ uses packed GEMM below 256 flattened tokens but explicitly
  dequantizes the complete weight before `torch.matmul` at 256 or more;
- AWQ-Triton is present and consumes the AutoAWQ layout, but is not needed as
  the one allowed fallback because Marlin passed;
- AWQ-Marlin accepts only the relevant runtime-zero-point `uint4` case here,
  group sizes `{-1,32,64,128}`, FP16/BF16 activations, and SM >= 8.0;
- Marlin requires `N % 64 == 0`, `K % 128 == 0`, and local K divisible by the
  group size. Every actual TP=1 Llama-3.1-8B shape passes, including fused
  up+gate `(K,N)=(4096,28672)`;
- active workspace is 82 int32 elements, or 328 bytes per matrix on this GPU;
- runtime dispatch is `apply_awq_marlin_linear -> gptq_marlin_gemm`; neither
  `awq_dequantize` nor `torch.matmul` occurs in that Python hot path.

The validated v5 environment itself has torch 2.5.1, while its newly installed
vLLM 0.11.2 wheel was built for torch 2.9. Importing `vllm._C` there fails with
an undefined c10 symbol. A clean isolated environment with torch 2.9.0+cu128,
vLLM 0.11.2, Transformers 4.51.3, AutoAWQ 0.2.9, and bitsandbytes 0.49.2 loads
both Marlin compiled ops successfully. SwiftLLM's small C++ extension also
builds against torch 2.9. This environment transition must be covered by a
matched FP16 parity/smoke check before serving evidence is trusted; it is not
hidden as part of the speed result.

## Representation and numerics

All values below come from
`phase-a/awq-marlin-kernel-feasibility.json`; full long-form rows are in
`phase-a/kernel-results.csv` and `phase-a/storage-results.csv`.

| matrix `[out,in]` | FP16 MiB | Marlin MiB | fraction | NF4 dual-layout MiB | AWQ rel-L2 range | AWQ cosine range |
|---|---:|---:|---:|---:|---:|---:|
| q/o `[4096,4096]` | 32.00 | 8.31 | 0.260 | 18.00 | 0.118–0.126 | 0.9921–0.9931 |
| k/v `[1024,4096]` | 8.00 | 2.08 | 0.260 | 4.50 | 0.123–0.129 | 0.9917–0.9927 |
| up/gate `[14336,4096]` | 112.00 | 29.09 | 0.260 | 63.00 | 0.103 | 0.9947–0.9948 |
| down `[4096,14336]` | 112.00 | 29.09 | 0.260 | 63.00 | 0.104–0.105 | 0.9945–0.9946 |
| fused up+gate `[28672,4096]` | 224.00 | 58.19 | 0.260 | 126.00 | 0.103–0.104 | 0.9946–0.9947 |

Every output was finite for `M={1,16,128,1024}`. The worst relative L2 error
was 0.1293 and minimum cosine similarity was 0.9917, passing the frozen 0.20
and 0.98 gates. Marlin storage was at most 25.98% of FP16, including scales,
zero points, and workspace. Its maximum unexplained measured call allocation
was 5.13 MiB, far below the 64 MiB gate and unlike NF4's one-full-weight
multi-row allocations (8–224 MiB per tested matrix, accumulating to the v5
profile pathology).

## Kernel latency

Entries are median/P95 milliseconds from 20 samples after 10 warmups.

| matrix | M | FP16 | AWQ-Marlin | NF4 |
|---|---:|---:|---:|---:|
| q/o | 1 | 0.068/0.077 | **0.052/0.057** | 0.063/0.067 |
| q/o | 128 | **0.104/0.108** | 0.113/0.116 | 0.208/0.217 |
| q/o | 1024 | 0.663/0.667 | **0.567/0.576** | 0.690/0.708 |
| k/v | 1 | **0.034/0.039** | 0.060/0.069 | 0.057/0.065 |
| k/v | 128 | **0.046/0.049** | 0.089/0.096 | 0.158/0.171 |
| k/v | 1024 | **0.185/0.193** | 0.188/0.206 | 0.291/0.310 |
| up/gate | 1 | 0.180/0.190 | **0.072/0.090** | 0.150/0.154 |
| up/gate | 128 | 0.361/0.369 | **0.275/0.290** | 0.585/0.602 |
| up/gate | 1024 | 2.024/2.041 | **1.987/1.993** | 2.124/2.139 |
| down | 1 | 0.173/0.187 | **0.070/0.082** | 0.106/0.122 |
| down | 128 | 0.303/0.311 | **0.273/0.281** | 0.534/0.544 |
| down | 1024 | **1.842/1.849** | 2.042/2.066 | 2.092/2.123 |
| fused up+gate | 1 | 0.315/0.326 | **0.105/0.109** | 0.152/0.157 |
| fused up+gate | 128 | 0.481/0.488 | **0.477/0.479** | 0.887/0.894 |
| fused up+gate | 1024 | **4.064/4.098** | 4.066/4.137 | 4.227/4.257 |

K/V is a small-N exception: Marlin is 1.79× FP16 at one row and 1.93× at 128
rows. It is only 1.02× at 1024 rows. The large MLP matrices compensate. A
simple sum of the seven SwiftLLM projection medians predicts separate AWQ
up/gate at 0.63× FP16 for M=1, 1.13× for M=128, and 0.99× for M=1024. A
combined up+gate representation predicts 0.58×, 1.06×, and 1.00× respectively.
These are kernel-only mechanism estimates, not serving claims.

## Phase A decision

The candidate is numerically sane, uses a true packed INT4 hot path, supports
every exact decoder shape, uses one representation for decode and prefill, and
removes the full-weight/multi-GiB workspace mechanism. It is not clearly
regressive overall. Therefore AWQ-Marlin passes the cheap gate and AWQ-Triton
is not measured. Proceed to minimal integration, explicitly resolving and
validating the torch 2.9 runtime environment. No static GO claim is made until
full-shape resource profiles and repeated frozen-load serving pass.
