# Bounded Marlin tuning follow-ups

Both pre-registered follow-ups stopped without changing the measured default
backend.

## Reduction controls

For W4-16, seven-sample integrated model timings were:

| setting | 1024 prefill median/P95 ms | decode median/P95 ms | prefill change | decode change | decision |
|---|---:|---:|---:|---:|---|
| default FP32 reduce | 258.71 / 264.06 | 21.79 / 22.12 | reference | reference | keep |
| FP16 reduce | 257.67 / 266.08 | 22.43 / 22.68 | 0.40% faster | 2.93% slower | reject |
| FP16 reduce + atomic add | 259.85 / 272.32 | 21.90 / 22.01 | 0.44% slower | 0.47% slower | reject |

Neither alternative reached the frozen 3% prefill-improvement threshold. No
full-shape or serving rerun was allowed.

## Packed QKV fusion

Concatenating actual AutoAWQ Q/K/V output columns before the same Marlin repack
reduced summed kernel medians by 68.7% at M=1, 3.72% at M=1024, and 1.46% at
M=8192. Worst fused-vs-separate relative L2 difference was 0.0116, and packed
storage did not increase, so it advanced to the integrated gate.

At W4-8, integrated 1024 prefill improved from 263.54 to 256.50 ms (2.67%) and
decode improved from 22.15 to 21.37 ms. However, SwiftLLM's rotary/KV kernels
require contiguous Q/K/V. Materializing contiguous slices changed the
full-shape allocation pattern: temporary allocated bytes remained 4.145 GiB,
but physically safe blocks fell from 3,034 to 2,938 (3.16%) in the unchanged
`mem_get_info` capacity profiler. The pre-registered gate prohibited any safe
capacity regression, so fused QKV was rejected and the source reverted.

Raw evidence is in `phase-c/tuning/`. No further flags, fusions, kernels, or
load search were attempted.
