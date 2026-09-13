# Pre-serving multi-M prefill diagnosis

Status: complete before any AWQ or intermediate-load serving run.

The pre-registered compute-only integrated prefill benchmark used the exact
FP16 and AWQ-Marlin W4-16 states, 1024 tokens per sequence, two warmups, seven
synchronized measurements, unchanged max batch/token limits, and
`ignore_kvcache=True` to remove the known capacity difference from this kernel/
compute comparison.

| effective M | sequences | FP16 median ms | AWQ median ms | AWQ / FP16 |
|---:|---:|---:|---:|---:|
| 1,024 | 1 | 255.892 | 268.244 | 1.048 |
| 3,072 | 3 | 711.662 | 764.134 | 1.074 |
| 4,096 | 4 | 934.980 | 1,021.669 | 1.093 |
| 6,144 | 6 | 1,397.239 | 1,520.299 | 1.088 |
| 7,168 | 7 | 1,639.068 | 1,786.359 | 1.090 |
| 8,192 | 8 | 1,867.219 | 2,023.364 | 1.084 |
| 9,216 | 9 | 2,130.754 | 2,291.113 | 1.075 |
| 16,384 | 16 | 3,745.371 | 4,034.849 | 1.077 |
| 24,576 | 24 | 5,676.154 | 6,103.363 | 1.075 |
| 32,768 | 32 | 7,559.001 | 8,125.430 | 1.075 |

AWQ is slower at every fixed M, by 4.8--9.3% at the measured medians. This
confirms a pure large-M prefill compute regression exists before queue/KV
effects are introduced. The trace's largest synchronized nine-request cohort
(M=9,216) lies near the existing two-second reference boundary even in FP16,
while AWQ adds about 160 ms at the same M.

The first scale-6 FP16 endpoint run was used to validate batch instrumentation.
Its observed prefill M values were 1,024, 2,048, 3,072, 4,096, 6,144, 7,168,
8,192, and 9,216. All lie inside the pre-registered 1,024--32,768 range. M=2,048
was not one of the original ten exact points, so a bounded matched supplemental
measurement was added before any AWQ or intermediate serving result: FP16
484.713 ms, AWQ 504.611 ms, ratio 1.041. This addition is transparent,
diagnostic only, and changes neither the load grid nor decision rule.

Raw samples and peak allocations are retained under
`benchmark-results/fp16-awq-crossover-v7/microbenchmark/`. No kernel,
quantization, fusion, scheduler, KV, state, or load was changed in response.
