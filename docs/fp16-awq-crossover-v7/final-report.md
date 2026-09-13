# FP16/AWQ-Marlin W4-16 crossover characterization v7

Status: **complete — crossover evidence GO; no controller implemented**.

This experiment tests only the validated static state pair `{FP16,
AWQ-W4-16}`. It preserves the unchanged Llama-3.1-8B checkpoint, offline AWQ
artifact, BurstGPT/DuReader request contents, strict-FCFS scheduler, KV
semantics, engine limits, decoding, RTX 3090 GPU 5, and all v2/v4/v5/v6
artifacts. The fixed nine-load protocol and 36-run order were committed as
`72da2d5` before new intermediate-load results. The earlier v6 scale-4 result
is used only as exploratory motivation.

All new evidence is under `benchmark-results/fp16-awq-crossover-v7/`. No
runtime adaptation, controller, alternate backend, scheduler policy, KV change,
or kernel optimization was implemented.

## Protocol validity and evidence volume

- Nine fixed scales: `6, 5.75, 5.5, 5.25, 5, 4.75, 4.5, 4.25, 4`, or nominal
  offered rates `0.667--1.000 requests/s`.
- The scale-6 and scale-4 inputs are byte-identical to v5/v6. Every point has
  the same source sequences 42--105 and common request-content hash
  `2219f131...e7d038c`.
- Two counterbalanced, fresh-process repeats per state/load: 36/36 valid runs.
- 2,304/2,304 requests completed, each with exactly 1,024 prompt tokens and 512
  output tokens (1,179,648 generated tokens total).
- 83,041 forward-batch events, including 421 prefill batches, plus 15,847
  periodic pressure samples. Every event passed required-field, monotonic-index,
  effective-M, safe-block, and range checks.
- Safe blocks reproduced exactly: FP16 1,768 (28,288 token slots), AWQ-W4-16
  4,286 (68,576 slots, +142.4%).
- All runs used one NVIDIA GeForce RTX 3090 through `CUDA_VISIBLE_DEVICES=5`,
  torch 2.9.0+cu128, vLLM 0.11.2, Transformers 4.51.3, max batch 32, max tokens
  49,152, block size 16, utilization 0.99, and one excluded 8-token warmup.

The exact per-run wide table is
`analysis/serving_runs.csv`; it contains P50/P95/P99 for every requested
latency/queue/batch metric, safe and used blocks, queue distributions, swaps,
HBM-relevant capacity, and throughput. Raw sources are each run's
`metadata.json`, `requests.jsonl`, `telemetry.jsonl`, and `batches.jsonl`.

## Fixed-M prefill diagnosis before the matrix

The compute-only integrated benchmark uses separate 1,024-token sequences and
the identical model forward with KV writes disabled. This isolates model/
Marlin compute from capacity and scheduling.

| effective M | sequences | FP16 median ms | AWQ median ms | AWQ/FP16 |
|---:|---:|---:|---:|---:|
| 1,024 | 1 | 255.892 | 268.244 | 1.048 |
| 2,048 | 2 | 484.713 | 504.611 | 1.041 |
| 3,072 | 3 | 711.662 | 764.134 | 1.074 |
| 4,096 | 4 | 934.980 | 1,021.669 | 1.093 |
| 6,144 | 6 | 1,397.239 | 1,520.299 | 1.088 |
| 7,168 | 7 | 1,639.068 | 1,786.359 | 1.090 |
| 8,192 | 8 | 1,867.219 | 2,023.364 | 1.084 |
| 9,216 | 9 | 2,130.754 | 2,291.113 | 1.075 |
| 16,384 | 16 | 3,745.371 | 4,034.849 | 1.077 |
| 24,576 | 24 | 5,676.154 | 6,103.363 | 1.075 |
| 32,768 | 32 | 7,559.001 | 8,125.430 | 1.075 |

M=2,048 is a transparently labeled bounded supplement after instrumentation
validation observed a two-request partial batch; all original pre-registered
points remain intact. In serving events at shared M=1,024--9,216, AWQ median
forward duration is also consistently slower, by 1.3--6.8%. Thus a real AWQ
large-M prefill tax exists; any end-to-end advantage must come from admission
and queue/preemption relief, not a faster Marlin prefill kernel.

## Every serving run: raw-derived results

`prefill→output` is first-prefill to first model output. TPOT is client-stream
TPOT. `wait/run/swap max` and `KV max` are periodic pressure observations;
`analysis/serving_runs.csv` additionally reports their P50/P95/P99 and exact
safe/used-block distributions. Prefill sequence and M columns are event-level.

| scale | state | rep | TTFT P50/P95/P99 s | >2s | queue P50/P95/P99 s | prefill→output P50/P95/P99 s | TPOT P50/P95/P99 ms | done RPS | preempt | wait/run/swap max | KV max | prefill seq P50/P95 | M P50/P95 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | FP16 | 0 | 1.898/2.175/5.308 | 32.8% | 0.011/0.040/4.869 | 1.878/2.144/2.144 | 42.8/50.0/51.0 | 0.523 | 4 | 2/23/2 | 1.000 | 6.0/9.0 | 6144/9216 |
| 6 | FP16 | 1 | 1.940/2.219/8.518 | 32.8% | 0.024/0.040/8.079 | 1.898/2.162/2.162 | 49.8/54.1/57.2 | 0.509 | 4 | 2/23/2 | 0.998 | 6.0/9.0 | 6144/9216 |
| 6 | W4-16 | 0 | 2.075/2.355/2.355 | 67.2% | 0.028/0.034/0.038 | 2.029/2.303/2.303 | 46.7/53.3/53.3 | 0.536 | 0 | 7/24/0 | 0.477 | 7.5/9.0 | 7680/9216 |
| 6 | W4-16 | 1 | 2.085/2.329/2.329 | 67.2% | 0.008/0.039/0.041 | 2.028/2.295/2.295 | 49.2/53.7/53.7 | 0.534 | 0 | 0/24/0 | 0.473 | 7.5/9.0 | 7680/9216 |
| 5.75 | FP16 | 0 | 1.942/2.206/8.473 | 45.3% | 0.013/0.028/7.944 | 1.904/2.184/2.184 | 48.7/54.2/57.9 | 0.525 | 6 | 2/24/4 | 0.998 | 6.0/9.0 | 6144/9216 |
| 5.75 | FP16 | 1 | 1.930/2.212/8.104 | 32.8% | 0.020/0.026/7.668 | 1.899/2.167/2.167 | 48.1/51.4/55.9 | 0.530 | 4 | 9/23/2 | 0.998 | 6.0/9.0 | 6144/9216 |
| 5.75 | W4-16 | 0 | 2.061/2.338/2.489 | 67.2% | 0.016/0.046/0.936 | 2.035/2.293/2.293 | 52.8/59.3/59.3 | 0.544 | 0 | 1/32/0 | 0.618 | 7.0/9.0 | 7168/9216 |
| 5.75 | W4-16 | 1 | 2.091/2.363/2.363 | 67.2% | 0.007/0.049/0.049 | 2.038/2.302/2.302 | 49.8/54.6/54.6 | 0.551 | 0 | 0/24/0 | 0.480 | 7.5/9.0 | 7680/9216 |
| 5.5 | FP16 | 0 | 1.920/2.752/6.314 | 43.8% | 0.010/0.347/3.909 | 1.903/2.384/2.384 | 50.3/55.8/62.1 | 0.539 | 7 | 10/24/4 | 0.999 | 7.0/9.6 | 7168/9779 |
| 5.5 | FP16 | 1 | 1.950/2.238/4.780 | 43.8% | 0.028/0.039/3.288 | 1.908/2.185/2.185 | 46.7/54.6/60.6 | 0.544 | 7 | 8/24/3 | 1.000 | 7.0/9.0 | 7168/9216 |
| 5.5 | W4-16 | 0 | 2.074/2.335/2.434 | 67.2% | 0.022/0.033/0.877 | 2.046/2.303/2.303 | 50.3/56.5/56.5 | 0.571 | 0 | 1/32/0 | 0.618 | 7.0/9.0 | 7168/9216 |
| 5.5 | W4-16 | 1 | 2.072/2.358/3.125 | 67.2% | 0.029/0.055/1.566 | 2.038/2.296/2.296 | 51.9/60.0/60.0 | 0.559 | 0 | 1/32/0 | 0.627 | 7.0/9.0 | 7168/9216 |
| 5.25 | FP16 | 0 | 1.938/4.028/7.428 | 43.8% | 0.024/1.622/5.021 | 1.906/2.389/2.389 | 49.4/56.7/62.8 | 0.550 | 8 | 10/24/4 | 1.000 | 7.0/9.6 | 7168/9779 |
| 5.25 | FP16 | 1 | 1.947/3.085/6.485 | 43.8% | 0.019/0.685/4.085 | 1.908/2.379/2.379 | 48.5/55.9/61.7 | 0.555 | 7 | 10/24/4 | 0.999 | 7.0/9.6 | 7168/9779 |
| 5.25 | W4-16 | 0 | 2.074/2.346/3.626 | 67.2% | 0.020/0.039/2.063 | 2.035/2.294/2.294 | 55.3/60.2/60.2 | 0.568 | 0 | 8/32/0 | 0.633 | 7.0/9.0 | 7168/9216 |
| 5.25 | W4-16 | 1 | 2.072/2.341/3.258 | 67.2% | 0.018/0.031/1.695 | 2.040/2.297/2.297 | 51.2/58.3/58.3 | 0.578 | 0 | 8/32/0 | 0.629 | 7.0/9.0 | 7168/9216 |
| 5 | FP16 | 0 | 2.195/4.780/8.017 | 51.6% | 0.021/2.367/5.605 | 1.901/2.391/2.391 | 49.2/56.2/62.0 | 0.569 | 7 | 10/24/4 | 0.998 | 5.5/9.4 | 5632/9677 |
| 5 | FP16 | 1 | 2.182/4.799/8.036 | 53.1% | 0.028/2.393/5.631 | 1.899/2.384/2.384 | 48.9/55.4/60.6 | 0.574 | 8 | 10/24/4 | 1.000 | 5.5/9.4 | 5632/9677 |
| 5 | W4-16 | 0 | 2.063/2.344/3.841 | 67.2% | 0.019/0.045/2.283 | 2.034/2.293/2.293 | 52.5/58.7/58.7 | 0.591 | 0 | 9/32/0 | 0.636 | 7.0/9.0 | 7168/9216 |
| 5 | W4-16 | 1 | 2.096/2.344/3.938 | 67.2% | 0.031/0.042/2.379 | 2.039/2.295/2.295 | 52.5/59.3/59.3 | 0.590 | 0 | 8/32/0 | 0.638 | 7.0/9.0 | 7168/9216 |
| 4.75 | FP16 | 0 | 1.948/4.606/7.681 | 43.8% | 0.023/2.199/5.275 | 1.906/2.389/2.389 | 42.8/54.9/61.2 | 0.594 | 8 | 10/24/4 | 1.000 | 7.0/9.6 | 7168/9779 |
| 4.75 | FP16 | 1 | 1.954/2.419/5.523 | 43.8% | 0.026/0.039/3.138 | 1.897/2.376/2.376 | 42.6/51.7/57.9 | 0.606 | 7 | 1/24/4 | 0.999 | 7.0/9.6 | 7168/9779 |
| 4.75 | W4-16 | 0 | 2.087/2.375/2.794 | 67.2% | 0.022/0.040/1.214 | 2.054/2.315/2.315 | 41.8/50.5/50.5 | 0.657 | 0 | 8/32/0 | 0.618 | 7.0/9.0 | 7168/9216 |
| 4.75 | W4-16 | 1 | 2.071/2.343/3.778 | 67.2% | 0.016/0.041/2.219 | 2.037/2.294/2.294 | 51.1/55.9/55.9 | 0.616 | 0 | 7/32/0 | 0.635 | 7.0/9.0 | 7168/9216 |
| 4.5 | FP16 | 0 | 1.939/10.432/15.101 | 32.8% | 0.030/8.235/14.104 | 1.575/2.175/2.175 | 50.8/58.8/65.2 | 0.598 | 12 | 10/24/3 | 1.000 | 4.0/8.3 | 4096/8550 |
| 4.5 | FP16 | 1 | 1.964/10.173/15.232 | 32.8% | 0.041/7.976/14.232 | 1.582/2.179/2.179 | 50.7/59.0/65.1 | 0.602 | 12 | 10/24/3 | 1.000 | 4.0/8.3 | 4096/8550 |
| 4.5 | W4-16 | 0 | 2.090/2.338/4.447 | 67.2% | 0.030/0.047/2.895 | 2.037/2.299/2.299 | 52.1/56.8/56.8 | 0.621 | 0 | 1/32/0 | 0.644 | 7.0/9.0 | 7168/9216 |
| 4.5 | W4-16 | 1 | 2.097/2.385/4.984 | 67.2% | 0.029/0.067/3.414 | 2.047/2.306/2.306 | 53.7/59.5/59.5 | 0.618 | 0 | 1/32/0 | 0.651 | 7.0/9.0 | 7168/9216 |
| 4.25 | FP16 | 0 | 1.936/12.631/16.354 | 32.8% | 0.028/10.681/15.299 | 1.444/1.931/1.931 | 51.2/59.9/67.0 | 0.620 | 13 | 10/24/3 | 1.000 | 3.5/8.0 | 3584/8192 |
| 4.25 | FP16 | 1 | 1.943/11.923/16.109 | 43.8% | 0.028/9.984/15.059 | 1.574/2.151/2.151 | 49.4/60.5/67.1 | 0.625 | 12 | 10/24/3 | 1.000 | 4.0/8.4 | 4096/8602 |
| 4.25 | W4-16 | 0 | 2.080/2.337/5.140 | 56.2% | 0.020/0.035/4.785 | 2.029/2.301/2.301 | 51.6/57.9/59.2 | 0.646 | 0 | 8/32/0 | 0.653 | 6.5/9.0 | 6656/9216 |
| 4.25 | W4-16 | 1 | 1.945/2.813/5.910 | 50.0% | 0.019/2.321/5.579 | 1.808/2.291/2.291 | 56.4/60.0/61.0 | 0.636 | 0 | 8/32/0 | 0.655 | 4.5/9.0 | 4608/9216 |
| 4 | FP16 | 0 | 2.181/17.528/18.986 | 59.4% | 0.040/15.331/17.496 | 1.921/2.178/2.178 | 52.0/63.0/68.5 | 0.617 | 16 | 12/24/3 | 1.000 | 4.0/9.0 | 4096/9216 |
| 4 | FP16 | 1 | 2.189/16.699/18.334 | 59.4% | 0.035/14.513/16.850 | 1.888/2.166/2.166 | 51.7/63.9/66.7 | 0.629 | 14 | 11/24/3 | 1.000 | 4.0/9.0 | 4096/9216 |
| 4 | W4-16 | 0 | 1.942/5.883/8.703 | 50.0% | 0.027/5.441/8.326 | 1.805/2.289/2.289 | 59.3/61.9/63.2 | 0.639 | 0 | 2/32/0 | 0.662 | 4.5/9.0 | 4608/9216 |
| 4 | W4-16 | 1 | 1.941/3.682/7.136 | 50.0% | 0.029/3.143/6.845 | 1.797/2.287/2.287 | 54.9/60.8/61.0 | 0.649 | 0 | 4/32/0 | 0.662 | 4.5/9.0 | 4608/9216 |

## Pre-registered point classification

Positive advantage is `FP16 P95 - W4 P95`. A point is AWQ-preferred only if
both repeat advantages exceed the fixed run-noise floor and queue-area,
queue-P95, KV-dwell, preemption, throughput, and validity gates all pass.

| RPS | scale | classification | FP16 P95 r0/r1 s | W4 P95 r0/r1 s | advantage r0/r1 s | noise s |
|---:|---:|---|---:|---:|---:|---:|
| .667 | 6.00 | FP16-default/non-inferior | 2.175/2.219 | 2.355/2.329 | -.181/-.109 | .111 |
| .696 | 5.75 | FP16-default/non-inferior | 2.206/2.212 | 2.338/2.363 | -.132/-.151 | .111 |
| .727 | 5.50 | FP16-default/non-inferior | 2.752/2.238 | 2.335/2.358 | .418/-.120 | .515 |
| .762 | 5.25 | ambiguous | 4.028/3.085 | 2.346/2.341 | 1.682/.744 | .943 |
| .800 | 5.00 | **AWQ-preferred** | 4.780/4.799 | 2.344/2.344 | 2.436/2.454 | .239 |
| .842 | 4.75 | ambiguous | 4.606/2.419 | 2.375/2.343 | 2.231/.077 | 2.186 |
| .889 | 4.50 | **AWQ-preferred** | 10.432/10.173 | 2.338/2.385 | 8.094/7.788 | .515 |
| .941 | 4.25 | **AWQ-preferred** | 12.631/11.923 | 2.337/2.813 | 10.294/9.109 | .709 |
| 1.000 | 4.00 | **AWQ-preferred** | 17.528/16.699 | 5.883/3.682 | 11.645/13.018 | 2.202 |

The predeclared bracket is therefore **0.727--0.800 requests/s** (scale
5.5--5.0). Scale 5.25 is inside that bracket and remains ambiguous rather than
being used to move it. Scale 4.75 is also ambiguous because FP16 was unstable:
one repeat showed a 2.231-second W4 advantage and one only 0.077 second. It is
not relabeled. The more robust sustained high-pressure region is scale
4.5--4.0 (0.889--1.000 requests/s), where three adjacent points pass in both
repeats.

## Why the crossover occurs

### 1. Pure AWQ-Marlin large-M regression

Fixed-M microbenchmarks show AWQ slower at every M by 4.1--9.3%. Serving
matched-M events independently show a 1.3--6.8% AWQ penalty over M=1,024--9,216.
This explains the low-load ordering and the near-constant W4 first-prefill P95
around 2.29--2.31 seconds. It also confirms that the W4 wins are not kernel
speedups.

### 2. Additional capacity changes admitted prefill batches

W4 does admit larger, less-fragmented prefill batches at several important
points. At scale 6 its mean prefill batch is 6.40 sequences in 10 batches versus
FP16's 5.33 in 12. At the repeated high-pressure scale 4.5 it is 5.82 in 11
batches versus 4.57 in 14. Scale 5.0 is 5.82 in 11 versus 5.33 in 12. The shift
is not universal—at scale 4.0 W4's mean is 4.57 versus FP16's 4.75—so capacity
alone does not explain all latency. `mechanism_by_load.csv` and
`batch_histograms.json` retain every distribution.

### 3. Queue/preemption relief dominates under sustained pressure

At AWQ-preferred scales 5.0/4.5/4.25/4.0, mean W4 waiting-queue area is only
21.2%/6.2%/11.2%/14.4% of FP16. FP16 averages 7.5/12/12.5/15 preemptions;
W4 has zero in every run. Completed throughput rises from
`.571/.600/.623/.623` to `.591/.619/.641/.644` requests/s. W4 has no dwell at
native logical KV >=.95, while FP16 repeatedly reaches approximately 1.0.

FP16's first-prefill-to-output P95 can actually become shorter at scale 4.5 and
4.25 because KV admission fragments work into smaller batches. End-to-end TTFT
still explodes because requests wait 8--15 seconds. That separation is direct
evidence that queue/preemption relief—not a faster W4 prefill—is the dominant
high-pressure mechanism.

### 4. Net effect is a combination

Below sustained pressure, the fixed-M W4 tax dominates and FP16 is better or
within run noise. Near the transition, stochastic timing of synchronized
cohorts causes partial admission and ambiguous repeats. Under sustained
pressure, W4's 2.42x safe-block capacity admits work with fewer splits and
eliminates preemption; the saved queue time is much larger than the roughly
5--9% prefill compute penalty. The result is a combination of (1), (2), and
(3), with (3) dominant after saturation.

## Strict 2-second paper-reference metric

The strict metric is preserved unchanged but does not define the crossover.
For example, at scale 5.0 W4 cuts P95 TTFT from 4.780/4.799 to 2.344/2.344
seconds and eliminates 7/8 preemptions, yet its violation rate is 67.2% versus
FP16's 51.6/53.1%. Many synchronized W4 first prefills land just above two
seconds. Conversely, at scale 4.5 FP16 has only 32.8% strict violations while
its P95 TTFT exceeds ten seconds. This validates the pre-registration warning:
a count around an unloaded service-time boundary is not an adequate state
criterion.

## Runtime-observable regime signals

The triggers use only causal 0.25-second queue/KV/preemption history. Offered
rate, planned future arrivals, request completions, and final latency are not
features.

| candidate | high FP16 fires | low/default false fires | qualifies | matched W4 release delay |
|---|---:|---:|---|---:|
| capacity margin + queue integral | 8/8 | 0/6 | yes | 27.6--51.5 s |
| sustained compound pressure | 8/8 | 0/6 | yes | 27.2--51.5 s |
| preemption + queue | 8/8 | 4/6 | no | 28.6--49.5 s |

The qualifying **sustained compound pressure** entry is: over the trailing
three seconds, FP16 KV utilization >=.95 for >=80% of elapsed time, waiting
>=4 for >=50%, and current waiting >=4. It first fires at 44.7--80.4 seconds
in every AWQ-preferred FP16 run and never at the three FP16-default points.

The qualifying **capacity-margin + queue-integral** entry is: fewer than 65
free FP16 blocks continuously for at least one second and trailing-five-second
waiting area >=15 request-seconds. It fires at 45.2--80.1 seconds in the same
8/8 high-regime runs and 0/6 low-regime runs. The preemption-only candidate is
rejected because scale-6/5.75 FP16 can preempt without the sustained queue
pressure needed to overcome W4 compute cost.

For a next controller experiment, use these exact entry inputs as two small
candidate rules. Preserve release hysteresis using current used blocks divided
by the known FP16 capacity (1,768), not W4's larger denominator: require a
causal ten-second window with waiting mean <=0.5, no preemption, and
FP16-equivalent utilization <=0.70 for >=90% of the window. Aligned W4 traces
would not satisfy this until 27--51 seconds after matched FP16 entry, so the
rule does not immediately release simply because W4 lowered native KV
utilization. Transition cost and actual closed-loop stability remain for the
next experiment; no controller is present here.

## CONFIRMED FINDINGS

- The exact static pair is `{FP16, AWQ-W4-16}` with 1,768 versus 4,286 safe
  blocks on the same RTX 3090. All 36 runs and 2,304 requests are complete and
  protocol-valid.
- AWQ-Marlin W4-16 has a real matched-M prefill compute regression: 4.1--9.3%
  in compute-only multi-M tests and 1.3--6.8% in serving batches.
- FP16 is the default/non-inferior choice at scales 6/5.75/5.5
  (0.667/0.696/0.727 requests/s). At the two lowest loads it has lower P95 in
  both repeats.
- AWQ-W4-16 passes every predeclared latency/noise/queue/KV/preemption/
  throughput gate at scale 5.0 and at the three adjacent high-pressure scales
  4.5/4.25/4.0. At scale 4.5, repeated P95 falls from 10.432/10.173 seconds to
  2.338/2.385 seconds; at scale 4 it falls from 17.528/16.699 to 5.883/3.682.
- High-pressure gains coincide with 79--94% lower waiting area, zero W4
  preemptions versus 7--16 FP16 preemptions/run, lower KV pressure, and higher
  completed throughput. They are not explained by a W4 kernel speedup.
- Two fixed causal signals separate all eight AWQ-preferred FP16 runs from all
  six FP16-default runs without future workload knowledge.

## SUPPORTED BUT UNCERTAIN FINDINGS

- The measured crossover bracket is scale 5.5--5.0, or
  **0.727--0.800 requests/s**, under this exact trace/hardware/protocol. Scale
  5.25 is ambiguous, so the precise threshold inside the bracket is unknown.
- Scale 4.75 is also ambiguous because one FP16 repeat queued while the other
  did not. The conditional ordering is strongest at the contiguous
  0.889--1.000 requests/s high-pressure region, not at every individual point
  above the first win.
- Additional W4 capacity often increases admitted prefill batch size and
  reduces fragmentation, but this shift is not monotonic at every load. Queue
  and preemption relief are the more consistent high-pressure explanation.
- The two qualifying trigger rules separate these open-loop traces. Their
  release behavior is assessed by aligned static W4 traces, not an actual
  state transition, so closed-loop hysteresis remains unvalidated.

## BLOCKED QUESTIONS

- Actual FP16-to-AWQ transition latency, temporary memory, KV migration/state
  compatibility, and controller stability are outside this goal and require a
  separate controller experiment.
- Generalization to different trace segments, request lengths, GPUs, scheduler
  policies, checkpoints, or tensor parallelism is unmeasured.
- Exact MorphServe AWQ parameters and hidden workload details remain
  unavailable; the validated local AutoAWQ W4/G128/asymmetric-ZP state remains
  a documented proxy.
- No new quality run was required or used for the latency crossover; this goal
  preserves the existing dataset and validated execution states rather than
  claiming new AWQ task-quality evidence.

## REMAINING UNCERTAINTY

- Two independent runs per state/load satisfy the frozen protocol but leave
  substantial run-level uncertainty near synchronized admission boundaries.
  Scales 5.25 and 4.75 demonstrate this directly and are not resolved post hoc.
- The pre-registered noise floor is deliberately conservative: large FP16
  repeat ranges can classify a useful mean difference as ambiguous.
- Batch telemetry is observational. The mechanism decomposition separates
  matched-M compute, admitted-M distributions, and queue/preemption behavior,
  but it is not a randomized causal mediation experiment.
- Entry signals are validated only for this fixed open-loop trace. The
  recommended release hysteresis uses FP16-equivalent block pressure to avoid
  immediate denominator-driven release, but needs closed-loop testing before
  deployment.

## CROSSOVER DECISION: **GO**

The pre-registered rule passes for the exact state pair **`{FP16,
AWQ-W4-16}`**. Use FP16 below the measured **0.727--0.800 requests/s** bracket;
AWQ-W4-16 is repeatedly preferable once sustained queue/KV pressure is present,
with the strongest contiguous evidence at **0.889--1.000 requests/s**. The next
controller experiment should test the fixed sustained-compound and
capacity-margin/queue-integral entry signals plus the ten-second
FP16-equivalent-KV release hysteresis above. This result authorizes that future
experiment only; it does not implement or validate a runtime controller.
