# MorphServe validated static frontier v5

Status: **complete — DYNAMIC ADAPTATION DECISION: NO-GO**.

This experiment used commit `9b3814a7` as the validated inference substrate
and the subsequently committed v5 benchmark/analysis harness. It did not
implement dynamic adaptation, runtime layer swapping, a controller, or a new
scheduler. The v2 and v4 namespaces remain preserved. All new evidence is
under `benchmark-results/static-frontier-v5/`.

## Experimental separation and scale

Quality and serving latency were measured separately:

- **Quality:** 106 fixed answer-bearing DuReader requests per state, launched
  sequentially, 1024-token prompts, exactly 512 greedy output steps, one
  excluded 8-token warmup, and no request overlap.
- **Serving:** FP16-only calibration first; then 64 fixed requests per run at
  three frozen loads for all four states. Every state was repeated at the
  frozen near-knee load. One request contributes 1.5625 percentage points to
  the SLO rate.
- **Mechanism:** two fresh full-shape resource profiles and one seven-sample
  model microbenchmark per state.

All conditions used the same Llama-3.1-8B checkpoint/tokenizer, prompts,
references, fixed front-to-back layer order, NF4 quantizer, greedy decoding,
output policy, scheduler, and engine configuration. FP16/Hugging-Face parity
was inherited from the already-passed v4 gate and was not reopened.

## FP16-only load calibration

The first four full-protocol FP16 calibration points had invariant 42.1875%
SLO violations even though P95 queueing stayed below 0.03 s. The fixed
BurstGPT timestamps launch groups of requests simultaneously, and their
multi-request prefill straddles the 2-second SLO even without sustained
queueing. Treating SLO alone as a saturation signal would therefore select a
false knee. Before any v5 W4 run, FP16 calibration was extended to 0.6667 and
1.0 nominal requests/s.

| nominal RPS | time scale | P95 TTFT (s) | TTFT >2 s | P95 queue (s) | peak KV | peak waiting | completed RPS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1667 | 24 | 2.093 | 42.2% | 0.017 | 0.489 | 0 | 0.160 |
| 0.2500 | 16 | 2.083 | 42.2% | 0.018 | 0.489 | 7 | 0.234 |
| 0.3333 | 12 | 2.118 | 42.2% | 0.029 | 0.760 | 0 | 0.305 |
| 0.5000 | 8 | 2.108 | 42.2% | 0.025 | 0.826 | 1 | 0.432 |
| 0.6667 | 6 | 2.118 | 32.8% | 0.043 | 0.998 | 7 | 0.534 |
| 1.0000 | 4 | 12.241 | 32.8% | 10.413 | 1.000 | 10 | 0.662 |

The predeclared revised rule selected the first load with peak logical KV at
least 0.85 or P95 queueing at least 2 s. Thus scale 6 / 0.6667 RPS is the
near-knee point, with adjacent scale 8 / 0.5 and scale 4 / 1.0 RPS frozen
before W4 measurement. The decision and raw-derived calibration are in
`calibration/calibration_decision.json`, `calibration/calibration_table.csv`,
and `calibration/calibration_fp16_knee.png`.

## QUALITY CHARACTERIZATION

### Primary DuReader F1

The 95% intervals are 10,000-resample request-level bootstrap intervals. W4
intervals for the delta use paired resampling of each fixed request against
FP16. The full 106-row paired table is
`analysis/quality_paired_differences.csv`.

| state | W4 layers | N | F1 % (95% CI) | paired delta vs FP16, pp (95% CI) | positive / zero / negative deltas |
|---|---:|---:|---:|---:|---:|
| FP16 | 0 | 106 | 14.471 [11.361, 17.880] | 0.000 [0.000, 0.000] | 0 / 106 / 0 |
| W4-8 | 8 | 106 | 14.230 [11.153, 17.524] | -0.241 [-2.506, 1.880] | 38 / 38 / 30 |
| W4-16 | 16 | 106 | 14.825 [11.626, 18.353] | +0.354 [-2.024, 2.531] | 48 / 32 / 26 |
| W4-32 | 32 | 106 | 14.977 [11.741, 18.444] | +0.507 [-1.481, 2.427] | 44 / 30 / 32 |

The F1 means remain non-monotonic; no paired interval excludes zero. Therefore
this experiment does **not** claim that NF4 improves or degrades DuReader F1.
No examples were discarded or retuned. The larger corrected sample changes the
v4 interpretation: the apparent 16-request F1 increase is not statistically
resolved on the fixed 106-request set.

### Secondary quantization-distortion evidence

This was predeclared as mechanism evidence, not a replacement quality metric.
It compares the 512 generated token IDs request by request with FP16.

| state | position-aligned token agreement (95% CI) | exact 512-token matches | mean FP16 common prefix (tokens, 95% CI) |
|---|---:|---:|---:|
| W4-8 | 0.389 [0.303, 0.476] | 36/106 (34.0%) | 186.3 [141.7, 232.8] |
| W4-16 | 0.343 [0.263, 0.427] | 29/106 (27.4%) | 153.1 [111.6, 196.6] |
| W4-32 | 0.307 [0.227, 0.391] | 26/106 (24.5%) | 138.7 [97.1, 181.7] |

Quantization clearly changes model outputs even though this Chinese/base-model
QA F1 sample cannot resolve a quality direction. Mean token agreement and
common-prefix length decrease with W4 coverage, but their bootstrap intervals
overlap, so the monotonic trend is supported rather than definitive.
`analysis/quality_characterization.png` shows both the unresolved F1 result and
the independent distortion signal.

## SERVING-LATENCY CHARACTERIZATION

### All request-latency runs

`prefill` below is the observed first-prefill-to-first-model-output interval for
the serving batch; it depends on batch composition. The isolated 1024-token
single-request prefill appears in the mechanism table. `decode` is request
TPOT over the remaining 511 output steps.

| run | RPS | done | TTFT P50/P95/P99 (s) | SLO % | queue P50/P95/P99 (s) | prefill P50/P95 (ms) | decode P50/P95 (ms) | done RPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `serv-fp16_0-scale-8-rep0` | 0.500 | 64/64 | 1.883/2.162/2.162 | 42.2 | 0.022/0.041/0.041 | 1854.8/2100.6 | 43.0/44.4 | 0.432 |
| `serv-w4_8-scale-8-rep0` | 0.500 | 64/64 | 139.117/290.934/315.338 | 93.8 | 138.480/290.421/314.827 | 500.6/930.9 | 48.0/69.1 | 0.138 |
| `serv-w4_16-scale-8-rep0` | 0.500 | 64/64 | 65.782/144.069/146.484 | 89.1 | 64.302/142.960/145.483 | 1459.0/1704.1 | 56.7/100.4 | 0.214 |
| `serv-w4_32-scale-8-rep0` | 0.500 | 64/64 | 2.110/4.220/4.220 | 68.8 | 0.047/1.888/1.888 | 2039.7/2314.3 | 77.9/81.5 | 0.383 |
| `serv-fp16_0-scale-6-rep0` | 0.667 | 64/64 | 1.843/2.102/5.295 | 32.8 | 0.018/0.048/4.873 | 1812.4/2071.6 | 44.8/47.9 | 0.534 |
| `serv-fp16_0-scale-6-rep1` | 0.667 | 64/64 | 1.874/2.123/5.363 | 32.8 | 0.023/0.036/4.940 | 1829.1/2088.5 | 44.3/48.2 | 0.532 |
| `serv-w4_8-scale-6-rep0` | 0.667 | 64/64 | 159.383/328.034/353.639 | 93.8 | 158.747/327.523/353.128 | 498.6/930.2 | 48.5/70.4 | 0.135 |
| `serv-w4_8-scale-6-rep1` | 0.667 | 64/64 | 154.119/324.858/349.846 | 93.8 | 153.481/324.339/349.326 | 500.1/930.8 | 48.3/70.1 | 0.137 |
| `serv-w4_16-scale-6-rep0` | 0.667 | 64/64 | 78.405/174.080/177.031 | 87.5 | 76.910/172.994/176.018 | 1474.1/1721.2 | 56.7/103.4 | 0.213 |
| `serv-w4_16-scale-6-rep1` | 0.667 | 64/64 | 80.706/176.537/179.407 | 89.1 | 79.243/175.468/178.410 | 1450.3/1695.3 | 56.6/103.3 | 0.212 |
| `serv-w4_32-scale-6-rep0` | 0.667 | 64/64 | 2.396/16.628/28.442 | 68.8 | 0.060/16.283/27.884 | 2044.8/2335.4 | 83.4/97.1 | 0.402 |
| `serv-w4_32-scale-6-rep1` | 0.667 | 64/64 | 2.338/14.442/24.382 | 76.6 | 0.033/13.099/24.081 | 2021.1/2287.4 | 81.7/93.3 | 0.412 |
| `serv-fp16_0-scale-4-rep0` | 1.000 | 64/64 | 1.846/11.774/15.314 | 32.8 | 0.020/9.935/14.317 | 1381.8/1817.2 | 47.1/56.0 | 0.667 |
| `serv-w4_8-scale-4-rep0` | 1.000 | 64/64 | 180.748/362.933/387.699 | 93.8 | 180.350/362.412/387.182 | 497.3/928.7 | 48.7/71.6 | 0.135 |
| `serv-w4_16-scale-4-rep0` | 1.000 | 64/64 | 97.878/214.575/218.144 | 87.5 | 96.389/213.498/217.141 | 1465.1/1715.5 | 58.0/105.9 | 0.206 |
| `serv-w4_32-scale-4-rep0` | 1.000 | 64/64 | 10.520/42.151/49.284 | 70.3 | 9.474/41.850/48.732 | 1285.5/2578.0 | 84.1/99.1 | 0.432 |

Every row above is regenerated from raw request timestamps. The complete
machine-readable table also includes arrival-to-scheduler timing and token
throughput: `analysis/serving_runs.csv`.

### Queue, KV, events, and HBM for every run

| run | waiting P50/P95/max | running P50/P95/max | swapped P50/P95/max | KV P50/P95/peak | blocks/slots | swap in/out | HBM peak/free-min GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `serv-fp16_0-scale-8-rep0` | 0/0/0 | 9/17/17 | 0/0/0 | .423/.778/.826 | 1768/28288 | 0/0 | 20.73/2.83 |
| `serv-w4_8-scale-8-rep0` | 19/40/43 | 3/4/4 | 0/1/1 | .869/.987/1.000 | 312/4992 | 28/28 | 15.80/7.76 |
| `serv-w4_16-scale-8-rep0` | 11/30/31 | 6/8/8 | 1/2/2 | .933/.993/1.000 | 568/9088 | 25/25 | 15.73/7.83 |
| `serv-w4_32-scale-8-rep0` | 0/0/9 | 17/24/25 | 0/0/1 | .675/.968/.993 | 2040/32640 | 1/1 | 16.22/7.34 |
| `serv-fp16_0-scale-6-rep0` | 0/1/2 | 11/21/23 | 0/1/2 | .502/.977/.997 | 1768/28288 | 4/4 | 20.73/2.83 |
| `serv-fp16_0-scale-6-rep1` | 0/1/7 | 11/21/23 | 0/1/2 | .512/.977/.998 | 1768/28288 | 4/4 | 20.73/2.83 |
| `serv-w4_8-scale-6-rep0` | 22/46/50 | 3/4/4 | 0/1/1 | .869/.987/1.000 | 312/4992 | 28/28 | 15.91/7.65 |
| `serv-w4_8-scale-6-rep1` | 20/46/47 | 3/4/4 | 0/1/1 | .869/.987/1.000 | 312/4992 | 28/28 | 15.80/7.76 |
| `serv-w4_16-scale-6-rep0` | 17/37/37 | 6/8/8 | 1/2/2 | .938/.993/1.000 | 568/9088 | 24/24 | 15.73/7.83 |
| `serv-w4_16-scale-6-rep1` | 17/37/37 | 6/8/8 | 1/2/2 | .937/.995/1.000 | 568/9088 | 24/24 | 15.73/7.83 |
| `serv-w4_32-scale-6-rep0` | 0/7/9 | 21/26/27 | 0/2/3 | .821/.996/1.000 | 2040/32640 | 14/14 | 16.22/7.34 |
| `serv-w4_32-scale-6-rep1` | 0/7/8 | 21/26/26 | 0/2/3 | .799/.994/1.000 | 2040/32640 | 13/13 | 16.22/7.34 |
| `serv-fp16_0-scale-4-rep0` | 0/10/10 | 19/23/24 | 0/2/3 | .844/.994/1.000 | 1768/28288 | 12/12 | 20.24/3.32 |
| `serv-w4_8-scale-4-rep0` | 24/50/53 | 3/4/4 | 0/1/1 | .869/.987/1.000 | 312/4992 | 28/28 | 15.80/7.76 |
| `serv-w4_16-scale-4-rep0` | 17/43/44 | 6/8/8 | 1/2/3 | .937/.992/1.000 | 568/9088 | 26/26 | 15.68/7.88 |
| `serv-w4_32-scale-4-rep0` | 3/17/26 | 24/27/27 | 0/2/4 | .970/.997/1.000 | 2040/32640 | 22/22 | 16.27/7.29 |

Physical HBM is supporting evidence only: SwiftLLM preallocates the profiled KV
cache. Logical KV occupancy, queues, and swap/preemption events identify the
active pressure mechanism.

### Near-knee repeatability and noise

The P95 ordering was identical in both near-knee repetitions:

`FP16 < W4-32 < W4-16 < W4-8`.

FP16 P95 was 2.102/2.123 s. W4-32 P95 was 16.628/14.442 s. Thus W4-32's mean
P95 **penalty** was 13.423 s, while the larger within-state run range was only
2.186 s. The result is opposite the required advantage and much larger than
ordinary observed run variation. W4-16 and W4-8 penalties were still larger.
Because ordering did not change materially, the predeclared third-repeat rule
was not triggered.

The load curves are `analysis/latency_slo_vs_load.png`; queue/KV curves are
`analysis/queue_kv_vs_load.png`; the combined quality–near-knee plot is
`analysis/quality_vs_near_knee_latency.png`.

## MECHANISM / STATE ELIGIBILITY

### Resource and model-level timing

Persistent allocation is CUDA allocation after model load (weights plus stable
model state). Temporary workspace is `profile peak - persistent allocation`
for the same 32-sequence/49,152-token profile used by serving. Each profile was
repeated twice; all block/workspace coefficients of variation were 0%.

| state | persistent GiB | temporary workspace GiB | safe blocks | KV token slots | 1024 prefill median/P95 ms | 1-token decode median/P95 ms | serving saturation |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 14.989 | 4.145 | 1768 | 28288 | 253.34/259.81 | 22.09/26.23 | 0.667 RPS |
| W4-8 | 13.575 | 6.395 | 312 | 4992 | 257.38/258.81 | 23.52/27.29 | at/below 0.5 RPS |
| W4-16 | 12.163 | 6.395 | 568 | 9088 | 263.19/271.30 | 27.93/41.03 | at/below 0.5 RPS |
| W4-32 | 9.339 | 6.395 | 2040 | 32640 | 269.28/275.07 | 32.93/37.64 | at/below 0.5 RPS |

Persistent memory decreases monotonically, but every W4 state pays a 6.395-GiB
profile workspace versus 4.145 GiB for FP16. W4-8 and W4-16 therefore expose
82.4% and 67.9% fewer safe KV blocks than FP16. They are pathological
pressure-relief states despite their lower resident weights.

W4-32 is different: it exposes 2040 blocks, 15.4% more than FP16. However, its
isolated prefill is 6.3% slower and its one-token decode median is 49.1% slower.
At 0.5 RPS it already reaches 0.993 peak KV and one swap, versus FP16's 0.826
and zero swaps. Longer service time consumes the modest additional capacity,
so the mechanism does not turn into latency relief.

### Eligibility table

| state | classification | resource relief | near-knee P95 better twice | SLO better twice | exceeds noise | mechanism aligns | reason |
|---|---|---:|---:|---:|---:|---:|---|
| FP16 | **ELIGIBLE** | reference | reference | reference | reference | reference | stable controller base/reference state |
| W4-8 | **INELIGIBLE** | no | no | no | no | no | temporary workspace leaves only 312 blocks; severe queue/swap pressure |
| W4-16 | **INELIGIBLE** | no | no | no | no | no | temporary workspace leaves only 568 blocks; severe queue/swap pressure |
| W4-32 | **INELIGIBLE** | yes | no | no | no | no | 15.4% block gain is outweighed by slower prefill/decode and earlier pressure |

No quantized state is recommended for runtime adaptation. The recommended
state set is therefore empty rather than `{FP16, W4-32}`.

## CONFIRMED FINDINGS

- All 424 sequential quality generations and all 1024 final serving requests
  completed at the fixed 1024/512 protocol.
- The 106-request paired F1 analysis finds no statistically resolved W4-vs-FP16
  difference; every paired 95% interval includes zero.
- Output-token agreement independently verifies substantial quantization
  distortion and decreases in mean from W4-8 to W4-32.
- Resident allocation falls with W4 coverage, but W4-8/W4-16 safe KV capacity
  is far below FP16 because the multi-row profile workspace is larger.
- W4-32 provides 272 additional blocks but is slower in isolated prefill and
  decode and is worse than FP16 at every frozen serving load.
- Both near-knee repeats reproduce the same full ordering. Every W4 state has
  worse P95 TTFT and SLO rate than FP16; no favorable outlier is used.

## SUPPORTED BUT UNCERTAIN FINDINGS

- Increasing W4 coverage appears to increase output distortion: mean token
  agreement falls 0.389 -> 0.343 -> 0.307. The intervals overlap, so the
  ordering itself is suggestive rather than conclusive.
- The small positive W4-16/W4-32 F1 point estimates may reflect metric
  insensitivity, output-length/lexical effects, or chance. Their paired
  intervals do not support a quality-improvement claim.
- The exact saturation thresholds are local to this fixed burst shape and RTX
  3090 substrate. The qualitative failure of W4 is much stronger than the
  observed repeat noise, but numerical knees should not be exported to other
  traces or hardware.

## BLOCKED QUESTIONS

- This remains an NF4/front-to-back proxy, not selective AWQ with the paper's
  unpublished LIS order and fused kernels.
- The paper's English-translated DuReader subset, hidden BurstGPT interval,
  Llama-3 checkpoint, and L4 hardware remain unavailable.
- The current experiment cannot determine how a different packed W4 backend
  would perform; it only establishes that the present bitsandbytes path is not
  a usable pressure-relief substrate.

## REMAINING UNCERTAINTY

- Quality uncertainty is dominated by the proxy dataset/model mismatch and
  heterogeneous 106-request F1 distribution, not queueing.
- Only the near-knee point has two runs per state; below/above points have one.
  This is sufficient for the predeclared gate because the near-knee ordering is
  identical and effect sizes are large, but it is not a general performance
  model.
- The synchronized trace groups make the strict 2-second violation rate
  non-monotonic for FP16. P95 queueing and logical KV evidence, not SLO alone,
  define the knee.

## DYNAMIC ADAPTATION DECISION: **NO-GO**

The GO rule requires a quantized state with a repeatable same-workload
latency/SLO advantage attributable to resource relief, plus quality distinction
or distortion evidence. The quality/distortion side passes, but no W4 state
passes the serving gate:

- W4-8 and W4-16 expose fewer safe KV blocks than FP16 and are pathological.
- W4-32 exposes more blocks but has a 13.423-s near-knee mean P95 penalty,
  versus only 2.186 s of observed run range, and its SLO rate is worse in both
  repeats.

The smallest blocker is the current W4 execution backend: its multi-row
prefill dequantizes into a 6.395-GiB temporary workspace, and its packed
single-token decode is slower than FP16. Before any controller work, replace it
with one packed W4 path that provides low-workspace prefill **and**
non-regressive decode, then rerun this static gate. Building a controller on
the current mechanism would not be scientifically meaningful.

## Regeneration and audit

Exact command templates and run IDs are in
`benchmark-results/static-frontier-v5/execution_commands.json`. The final
derivation command is:

```bash
cd /nfs/home/s314511048/MorphServe
VENV=/nfs/home/s314511048/.venv
PYTHONPATH="$PWD/swiftLLM" "$VENV/bin/python" -m benchmark.analyze_static_frontier \
  --manifest benchmark-results/static-frontier-v5/protocol_manifest.json \
  --quality-runs benchmark-results/static-frontier-v5/quality/runs/quality-{fp16_0,w4_8,w4_16,w4_32} \
  --serving-runs benchmark-results/static-frontier-v5/serving/runs/serv-* \
  --profile-files benchmark-results/static-frontier-v5/mechanism/profile-*.json \
  --microbenchmark-files benchmark-results/static-frontier-v5/mechanism/micro-{fp16_0,w4_8,w4_16,w4_32}.json \
  --output-dir benchmark-results/static-frontier-v5/analysis

PYTHONPATH="$PWD/swiftLLM" "$VENV/bin/python" -m benchmark.audit_static_frontier \
  --root benchmark-results/static-frontier-v5 \
  --output benchmark-results/static-frontier-v5/completion_audit.json
```

The machine audit performs 147 checks over raw files, protocol/workload hashes,
request counts, 1024/512 completion, telemetry fields, exact summary
regeneration, required load/repeat matrix, near-knee ordering, mechanism probe
matrix, derived deliverables, and the explicit decision. See
`docs/static-frontier-v5/completion-audit.md`.
