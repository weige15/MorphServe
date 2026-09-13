# MorphServe packed INT4 backend v6

Status: **complete — BACKEND DECISION: NO-GO**.

This experiment preserved all v2, v4, and v5 artifacts and replaced only the
candidate execution substrate. It did not implement dynamic adaptation, a
controller, runtime layer swapping, KVResizer, scheduler changes, or altered KV
semantics. All new evidence is under
`benchmark-results/packed-int4-backend-v6/`.

## Candidate and environment

The candidate is public AutoAWQ 0.2.9 W4 with group size 128, asymmetric zero
points, FP16 scales/activations, duo scaling and clipping, served by vLLM
0.11.2 AWQ-Marlin on an RTX 3090 (SM 8.6). MorphServe does not disclose group,
zero-point, or calibration details, so this frozen standard configuration is a
paper-parameter **proxy**, not an exact reproduction. The exact local
Llama-3.1-8B checkpoint and tokenizer are unchanged.

Offline quantization used the pinned `mit-han-lab/pile-val-backup` revision,
AutoAWQ shuffle seed 42, 128 accepted samples and 59 complete 512-token blocks.
The resulting 5,745,260,850-byte checkpoint was produced once; all file hashes
and exact calibration token IDs are recorded in
`phase-b/awq-checkpoint-manifest.json` and
`phase-b/calibration-selected.jsonl`.

The installed vLLM wheel cannot load in v5's torch-2.5 environment because its
`_C` extension has an unresolved c10 ABI symbol. A separate reproducible
execution environment uses torch 2.9.0+cu128, vLLM 0.11.2, Transformers 4.51.3,
AutoAWQ 0.2.9, and a SwiftLLM extension rebuilt against torch 2.9. Matched
SwiftLLM/Hugging-Face FP16 parity passed an exact 8-token greedy comparison in
this environment, so the environment transition is measured rather than
hidden.

## Phase A — cheap kernel feasibility

Every exact TP=1 decoder shape passed installed Marlin checks, including q/o,
k/v, up/gate, down, and combined up+gate. The Marlin hot path is
`gptq_marlin_gemm`; it does not call `awq_dequantize` or `torch.matmul` and uses
an 82-int32 (328-byte) workspace per retained matrix. One packed representation
serves every activation row count.

Across real layer-0 checkpoint matrices and `M={1,16,128,1024}`, every output
was finite, worst relative L2 error was 0.1293, minimum cosine similarity was
0.9917, and packed storage including scale/zero/workspace was at most 25.98% of
FP16. Maximum unexplained call allocation was 5.13 MiB. In contrast, the
existing dual-layout NF4 path materialized one full matrix for multi-row work.

Twenty-sample kernel medians showed Marlin generally faster for large MLP
matrices and non-regressive at M=1024, although small-N K/V remained slower.
A seven-projection sum predicted 0.63× FP16 at M=1, 1.13× at M=128, and 0.99×
at M=1024 with separate up/gate. Full results and P95s are in
`phase-a/kernel-results.csv` and the Phase A report. AWQ-Triton was not tested
because the preferred backend passed; ordinary AWQ's >=256-token full-weight
dequantization branch was explicitly excluded.

## Phase B — SwiftLLM integration and MLP fusion

Selected front-to-back decoder layers load AutoAWQ-scaled norms and packed
`qweight/qzeros/scales`; unselected layers and all non-decoder weights remain
on the exact base FP16 checkpoint. AutoAWQ layout is repacked once at model
load, after which source temporaries are released. Decode and prefill both use
the retained Marlin representation. All 224 matrix mappings / 672 checkpoint
components passed exact key, shape, and dtype checks. Marlin agreed with public
AutoAWQ dequantization at worst 0.000334 relative L2 error.

Separate quantized up/gate projections were numerically correct but recreated
the v5 capacity pathology through activation buffers rather than weight
dequantization: W4-8 full-shape temporary allocation was 6.387 GiB and only
1,711 blocks remained. Combining AutoAWQ output columns in exact `[up,gate]`
order before repack restored one projection/representation. Temporary
allocation fell to 4.145 GiB and blocks rose to 3,034. The fused mapping is
covered by the numerical audit.

All 0/8/16/32 eight-token smokes were finite and produced the same sane
continuation as FP16. Source and runtime checks find no retained dense matrix or
full-weight hot-path dequantization.

## Phase C — resource and model-performance gate

Two fresh 32-sequence/49,152-token profiles per state reproduced exactly:

| state | persistent GiB | packed GiB | temporary GiB | safe blocks | token slots | block gain | 1024 prefill median/P95 ms | decode median/P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 14.989 | 0 | 4.145 | 1768 / 1768 | 28,288 | reference | 260.78 / 266.80 | 22.18 / 23.77 |
| AWQ W4-8 | 12.585 | 0.844 | 4.145 | 3034 / 3034 | 48,544 | +71.6% | 263.54 / 267.82 | 22.15 / 22.38 |
| AWQ W4-16 | 10.181 | 1.689 | 4.145 | 4286 / 4286 | 68,576 | +142.4% | 264.54 / 271.24 | 22.25 / 23.02 |
| AWQ W4-32 | 5.374 | 3.377 | 4.142 | 6766 / 6766 | 108,256 | +282.7% | 263.91 / 281.00 | 21.30 / 21.94 |

All three candidates passed the resource/microbenchmark gate. Unlike NF4, safe
capacity is monotonic and even W4-8 is real pressure relief. Median prefill
regression is only 1.1–1.4%; decode is within +0.3% and improves 3.9% at W4-32.
Because capacity changed substantially, the already-defined adjacent v5 loads
were frozen for all states before any serving result.

## Phase D — frozen serving gate

Every row below contains 64/64 completed requests with 1024 prompt tokens and
512 greedy output steps. Scale 6 is the unchanged v5 FP16 near-knee workload;
it has two independent repeats per state. Scale 8/4 are the pre-existing
low/high neighbors. `prefill` is first-prefill-to-first-output and `decode` is
request TPOT. All values are regenerated from raw request timestamps.

| run | RPS | TTFT P50/P95/P99 s | >2 s | queue P95 s | prefill P50/P95 ms | decode P50/P95 ms | done RPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 s8 r0 | .500 | 1.963 / 2.218 / 2.218 | 42.2% | .026 | 1918.6 / 2177.4 | 45.2 / 58.5 | .428 |
| W4-8 s8 r0 | .500 | 2.005 / 2.310 / 2.310 | 54.7% | .046 | 1976.9 / 2243.1 | 45.3 / 47.1 | .427 |
| W4-16 s8 r0 | .500 | 2.078 / 2.373 / 2.373 | 67.2% | .035 | 2044.0 / 2316.7 | 45.9 / 49.3 | .424 |
| W4-32 s8 r0 | .500 | 2.173 / 2.446 / 2.446 | 67.2% | .044 | 2113.6 / 2385.9 | 47.3 / 48.7 | .425 |
| FP16 s6 r0 | .667 | 1.940 / 2.193 / 6.706 | 32.8% | .035 | 1896.5 / 2160.0 | 47.1 / 50.8 | .520 |
| FP16 s6 r1 | .667 | 1.923 / 2.232 / 7.621 | 32.8% | .041 | 1904.6 / 2175.2 | 48.4 / 52.3 | .516 |
| W4-8 s6 r0 | .667 | 2.029 / 2.313 / 2.313 | 54.7% | .045 | 1983.2 / 2249.3 | 48.0 / 52.4 | .537 |
| W4-8 s6 r1 | .667 | 2.056 / 2.306 / 2.306 | 54.7% | .047 | 1990.2 / 2254.7 | 43.6 / 52.7 | .537 |
| W4-16 s6 r0 | .667 | 2.112 / 2.351 / 2.351 | 67.2% | .046 | 2044.9 / 2314.8 | 47.4 / 53.4 | .535 |
| W4-16 s6 r1 | .667 | 2.085 / 2.341 / 2.341 | 67.2% | .028 | 2039.1 / 2311.1 | 49.4 / 52.7 | .536 |
| W4-32 s6 r0 | .667 | 2.197 / 2.478 / 2.478 | 67.2% | .044 | 2151.5 / 2428.2 | 48.7 / 52.6 | .537 |
| W4-32 s6 r1 | .667 | 2.201 / 2.477 / 2.477 | 67.2% | .046 | 2143.4 / 2428.1 | 49.3 / 52.7 | .534 |
| FP16 s4 r0 | 1.000 | 1.947 / 15.779 / 18.369 | 40.6% | 13.384 | 1900.1 / 2374.5 | 50.1 / 60.7 | .642 |
| W4-8 s4 r0 | 1.000 | 2.265 / 19.190 / 22.463 | 78.1% | 18.634 | 1360.2 / 2231.1 | 55.2 / 88.2 | .574 |
| W4-16 s4 r0 | 1.000 | 2.066 / 2.343 / 2.572 | 67.2% | .027 | 2042.2 / 2300.3 | 34.3 / 41.8 | .791 |
| W4-32 s4 r0 | 1.000 | 2.065 / 3.649 / 6.616 | 50.0% | 3.091 | 1915.2 / 2420.3 | 54.7 / 57.9 | .667 |

### Queue, KV, swaps, and HBM

Values are telemetry P50/P95/max. Safe blocks are 1768/3034/4286/6766 for
FP16/W4-8/16/32. Peak physical HBM stayed 20.46–20.79 GiB because SwiftLLM
preallocates KV; logical utilization and events identify pressure.

| run | waiting | running | swapped | KV utilization | swap in/out | HBM peak/free-min GiB |
|---|---:|---:|---:|---:|---:|---:|
| FP16 s8 | 0/0/9 | 10/17/19 | 0/0/0 | .456/.814/.880 | 0/0 | 20.73/2.83 |
| W4-8 s8 | 0/0/1 | 10/17/17 | 0/0/0 | .254/.459/.485 | 0/0 | 20.74/2.82 |
| W4-16 s8 | 0/0/0 | 10/17/17 | 0/0/0 | .176/.327/.345 | 0/0 | 20.71/2.85 |
| W4-32 s8 | 0/0/4 | 10/17/17 | 0/0/0 | .119/.209/.220 | 0/0 | 20.74/2.82 |
| FP16 s6 r0 | 0/1.5/7 | 11/22/23 | 0/1/2 | .512/.980/.999 | 4/4 | 20.67/2.88 |
| FP16 s6 r1 | 0/2/7 | 15/22/23 | 0/1/2 | .636/.981/1.000 | 4/4 | 20.73/2.83 |
| W4-8 s6 r0 | 0/0/8 | 15/24/24 | 0/0/0 | .379/.628/.666 | 0/0 | 20.68/2.88 |
| W4-8 s6 r1 | 0/0/8 | 11/24/24 | 0/0/0 | .282/.623/.657 | 0/0 | 20.61/2.94 |
| W4-16 s6 r0 | 0/0/8 | 11/24/24 | 0/0/0 | .211/.446/.473 | 0/0 | 20.66/2.90 |
| W4-16 s6 r1 | 0/0/0 | 15/24/24 | 0/0/0 | .273/.447/.475 | 0/0 | 20.71/2.85 |
| W4-32 s6 r0 | 0/0/7 | 15/24/24 | 0/0/0 | .173/.280/.295 | 0/0 | 20.66/2.90 |
| W4-32 s6 r1 | 0/0/7 | 15/24/24 | 0/0/0 | .179/.284/.302 | 0/0 | 20.74/2.82 |
| FP16 s4 | 0/10/11 | 20/23/24 | 0/2/3 | .928/.996/1.000 | 14/14 | 20.79/2.77 |
| W4-8 s4 | 0/15/19 | 21/32/32 | 0/0/0 | .569/.903/.986 | 0/0 | 20.70/2.86 |
| W4-16 s4 | 0/0/4 | 17/24/32 | 0/0/0 | .297/.470/.618 | 0/0 | 20.46/3.10 |
| W4-32 s4 | 0/2/4 | 18/32/32 | 0/0/0 | .216/.389/.417 | 0/0 | 20.57/2.99 |

### Frozen near-knee eligibility

Memory relief is real in every state and eliminates FP16's four preemptions per
near-knee run. It also improves near-knee P99 dramatically. But the fixed gate
requires P95 TTFT and/or strict SLO behavior to beat FP16 in **both** matched
repeats and exceed noise. No state does:

| state | P95 reduction vs matched FP16 (s) | mean | SLO reduction (pp) | P95/SLO better twice | eligible |
|---|---:|---:|---:|---:|---:|
| W4-8 | -0.120 / -0.074 | -0.097 | -21.875 / -21.875 | no / no | **no** |
| W4-16 | -0.157 / -0.109 | -0.133 | -34.375 / -34.375 | no / no | **no** |
| W4-32 | -0.285 / -0.245 | -0.265 | -34.375 / -34.375 | no / no | **no** |

FP16's near-knee P95 range is 0.038 s; the frozen noise floor is 0.111 s (5%
of FP16 mean). W4-8's smaller penalty is partly within that conservative
floor, but it is in the wrong direction in both repeats and its SLO is 21.875
points worse. The synchronized trace places many first-prefill completions near
the 2-second boundary. AWQ removes queue/preemption tails, but its serving
prefill P95 rises from FP16's 2.160/2.175 s to 2.249/2.255, 2.315/2.311, and
2.428/2.428 s as W4 coverage grows. That compute shift worsens P95/SLO even
with ample KV headroom.

### Adjacent high-load evidence (exploratory confirmation)

W4-16 is a real high-pressure sweet spot at the already-defined 1.0-RPS
neighbor. Because the first run was strongly favorable, a transparently
exploratory second FP16/W4-16 repeat checked mechanism repeatability without
changing eligibility:

| state | P95 TTFT reps (s) | P95 queue reps (s) | peak KV | preemptions | done RPS reps |
|---|---:|---:|---:|---:|---:|
| FP16 | 15.779 / 13.207 | 13.384 / 11.499 | 1.000 / 1.000 | 14 / 13 | .642 / .642 |
| W4-16 | 2.343 / 3.556 | .027 / 3.016 | .618 / .662 | 0 / 0 | .791 / .657 |

The high-load P95 advantage is large and mechanistically aligned in both runs,
but strict >2-second SLO remains worse (67.2%/50.0% versus 40.6%/42.2%) because
W4 first-prefill compute crosses the threshold. The protocol explicitly
forbids replacing the frozen scale-6 eligibility workload after seeing this
result. Therefore this finding is promising evidence for a future revised
workload/SLO study, not a GO under the current gate.

## Bounded tuning outcome

Two installed Marlin reduction variants failed a predeclared 3% prefill gain.
Packed QKV fusion improved isolated W4-8 prefill 2.67% and decode 3.56%, but
contiguous Q/K/V materialization reduced physically safe profile blocks from
3,034 to 2,938. It failed the no-capacity-regression rule and was reverted. No
additional flag, fusion, kernel, or workload search was attempted. See
`tuning-report.md` and `phase-c/tuning/`.

## Quality decision

The protocol requires the 106-request quality rerun only after at least one
state becomes eligible. No state passed the serving gate, so candidate quality
was correctly not run. The 0/8/16/32 short generations and packed numerical
checks prove execution sanity, not task quality. The prior v5 NF4 quality
result remains preserved but is not transferred to AWQ.

## CONFIRMED FINDINGS

- The exact installed AWQ-Marlin kernels support every Llama-3.1-8B decoder
  shape on the RTX 3090 when run in their required torch-2.9 ABI environment.
- The offline artifact is genuine activation-aware AWQ, and the serving hot
  path retains only packed INT4 qweights, FP16 scales, packed zero points, and
  tiny Marlin workspace; no full FP16 decoder copy or hot-path dequantization
  is present.
- Fused packed up+gate is necessary: without it, activation concatenation alone
  recreates v5's ~6.39-GiB profile pathology.
- With fusion, temporary profile allocation matches FP16 and safe KV increases
  monotonically by 71.6%, 142.4%, and 282.7% for W4-8/16/32.
- Isolated decode is non-regressive; the NF4 workspace/decode blockers are
  solved.
- At the frozen near-knee workload, every AWQ state has worse P95 TTFT and
  strict SLO in both repeats despite lower KV pressure and zero preemptions.
- W4-16 beats saturated FP16 P95 strongly in both exploratory 1.0-RPS repeats,
  showing that the resource-relief mechanism is real but activates too late
  for the fixed eligibility point.

## SUPPORTED BUT UNCERTAIN FINDINGS

- The remaining near-knee loss is primarily multi-request prefill compute,
  because queueing is near zero, isolated decode is non-regressive, and serving
  prefill-to-first-output rises monotonically with W4 coverage. Exact kernel
  versus batching/runtime contributions are not separated.
- W4-16 appears to be the best capacity/compute balance at 1.0 RPS; only two
  transparently labeled exploratory repeats support that claim, and one still
  queues for ~3 seconds at P95.
- Near-knee P99 improves because AWQ prevents preemption tails. P99 was not the
  predeclared eligibility metric and cannot substitute for failed P95/SLO.

## BLOCKED QUESTIONS

- Whether a different packed backend or a custom fused QKV/contiguous-output
  kernel can remove the remaining multi-request prefill tax without reducing
  safe capacity. Open-ended kernel development was out of scope.
- Whether a different predeclared trace shape, SLO threshold, or higher FP16
  saturation point would make W4-16 eligible. The current goal forbids choosing
  such a load after observing W4.
- AWQ quality/distortion on the 106-request set, because the eligibility
  prerequisite failed.
- Exact MorphServe AWQ parameters, LIS order, translated DuReader subset,
  hidden BurstGPT interval, Llama-3 checkpoint, and L4 hardware remain
  unavailable.

## REMAINING UNCERTAINTY

- Torch/vLLM had to move from the validated torch-2.5 environment to a matched
  torch-2.9 environment. FP16 parity and matched serving control this locally,
  but absolute v5 timing is not assumed invariant across environments.
- The strict 2-second SLO is highly sensitive to synchronized burst prefills;
  it can worsen while P99 queue/preemption tails improve.
- Adjacent loads have one confirmatory run per state; only FP16 and W4-16 at the
  high neighbor received an additional exploratory repeat.

## BACKEND DECISION: **NO-GO**

No AWQ state is eligible for future dynamic adaptation under the frozen gate.
AWQ-Marlin **does** solve the NF4 format, workspace, and decode problems and
creates real memory relief. The remaining blocker is multi-request **prefill
speed** relative to the 2-second synchronized-burst boundary: it makes
near-knee P95/SLO worse in both repeats for every 0/8/16/32 state.

The recommended runtime state set is empty. In particular, the encouraging
high-load W4-16 result does not justify `{FP16,W4-16}` because it fails the
predeclared near-knee rule. Do not implement a controller on this evidence.
