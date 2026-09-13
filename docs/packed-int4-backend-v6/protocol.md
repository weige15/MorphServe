# Packed INT4 backend v6 protocol

Status: **pre-registered before v6 GPU kernel measurements** on 2026-09-13 UTC.

## Question and boundaries

This experiment asks whether a genuinely packed, low-workspace INT4 execution
backend can replace the rejected bitsandbytes NF4 path and create at least one
static SwiftLLM pressure-relief state worth switching into later. The existing
v5 NF4 result remains a valid NO-GO and is not retuned or overwritten.

This protocol does **not** implement a controller, runtime layer swapping,
KVResizer, scheduler changes, dynamic adaptation, or a different FP16 path. It
preserves the v2, v4, and v5 namespaces. New evidence is written only under
`benchmark-results/packed-int4-backend-v6/` and
`docs/packed-int4-backend-v6/`.

FP16/Hugging-Face correctness is inherited from
`benchmark-results/table5-substrate-v4/substrate-evidence/fp16-hf-parity.json`
unless new evidence falsifies it.

## Frozen candidate and proxy status

The first candidate is the AWQ-Marlin path shipped in installed vLLM 0.11.2.
The MorphServe paper identifies AWQ INT4 but does not disclose group size,
zero-point convention, calibration corpus, sample count, or packing details.
The following common AutoAWQ configuration is therefore frozen as a
**paper-parameter proxy**, not an exact paper reproduction:

- quantizer/exporter: public AutoAWQ 0.2.9;
- weight bits: 4;
- group size: 128 along the input dimension;
- asymmetric integer zero point: enabled;
- checkpoint dialect: AutoAWQ GEMM (`qweight`, `qzeros`, FP16 `scales`);
- serving backend: vLLM 0.11.2 AWQ-to-Marlin repack and Marlin GEMM;
- activation and output dtype: FP16;
- calibration: AutoAWQ's standard `mit-han-lab/pile-val-backup` validation
  corpus at frozen revision `2f5e46ae6a69cf0dce4b12f78241c408936ca0e4`,
  shuffled with its fixed seed 42, at most 128 accepted samples and 512 tokens
  per packed calibration block;
- `duo_scaling=true` and clipping enabled;
- memory-only execution controls: one calibration sample per forward partition
  and a 512 MiB scale-search chunk ceiling (these do not change the accepted
  samples or quantization objective);
- embeddings, norms, and LM head remain FP16;
- decoder layers are selected front-to-back in the fixed states 0/8/16/32.

The base checkpoint remains the exact local Llama-3.1-8B snapshot used by v5.
Quantization is performed offline once. The packed artifact is stored outside
Git because of its size; its path, per-file hashes, quantization metadata, and
source-checkpoint hashes will be recorded under the v6 namespace. Serving must
not quantize or repeatedly repack from FP16 at startup. A one-time conversion
from AutoAWQ layout to a versioned offline Marlin artifact is permitted only if
its schema and hashes are recorded and the runtime loads that packed artifact
directly.

If AutoAWQ 0.2.9 cannot run in a compatible isolated tool environment, an
activation-aware AWQ artifact produced by public llm-compressor may replace the
exporter only before any full serving measurement; the W4/group/zero/calibration
choices above remain fixed and the tooling substitution must be recorded.

## Phase A: cheap backend gate

First record the exact installed vLLM AWQ, AWQ-Marlin, and AWQ-Triton source
paths, package versions, GPU model/capability, compiled-op availability, and
shape checks. At tensor parallel size one, test every distinct Llama-3.1-8B
decoder projection:

| class | source `[out,in]` | Marlin `(K,N)` |
|---|---:|---:|
| q/o | `[4096,4096]` | `(4096,4096)` |
| k/v | `[1024,4096]` | `(4096,1024)` |
| up/gate | `[14336,4096]` | `(4096,14336)` |
| down | `[4096,14336]` | `(14336,4096)` |
| fused up+gate feasibility | `[28672,4096]` | `(4096,28672)` |

For each unfused class, measure activation row counts `M={1,16,128,1024}`.
Use ten warmups and at least twenty timed samples for kernel-level latency.
Compare AWQ-Marlin with FP16 and the current NF4 path on identical inputs. Use
CUDA events or explicit device synchronization around every sample.

Record for each matrix and row count:

- finite outputs, relative L2 error, maximum absolute error, and cosine
  similarity against the same FP16 operation;
- qweight, scales, zero-point, workspace, and total persistent bytes;
- peak allocated bytes above persistent inputs/outputs during the operation;
- latency median and P95 for FP16, NF4, and AWQ-Marlin;
- exact dispatch/API and whether full FP16 weight materialization appears in
  source or measured hot-path allocation.

Phase A passes only if all real Llama shapes pass the installed Marlin checks,
all outputs are finite with relative L2 error below 0.20 and cosine similarity
above 0.98, one packed representation serves all M values, packed-side storage
is below 35% of the FP16 matrix bytes, and unexplained hot-path temporary
allocation stays below 64 MiB per matrix call. This gate primarily rejects the
multi-GiB dequantization pathology; performance ratios are evidence for the
integration decision rather than a license to alter the frozen config.

If AWQ-Marlin is unsupported or clearly unable to provide a non-regressive
path, evaluate at most one fallback: installed vLLM AWQ-Triton using the same
AutoAWQ layout/config. The ordinary vLLM AWQ large-M dequantize-to-FP16 branch
is not a viable fallback. No custom CUDA/Triton kernel project is allowed.

## Phase B: minimal SwiftLLM integration

Proceed only after Phase A passes. Add only the backend/config and packed-load
seams needed to:

- leave state 0 and every unselected decoder layer on the unchanged FP16 path;
- load selected layer matrices from the offline AWQ/Marlin artifact;
- retain qweight, scales, zero points, and only required Marlin workspace;
- avoid a hidden full FP16 copy and avoid hot-path full-weight dequantization;
- use one serving representation per matrix for decode and prefill.

The FP16 fused `[up,gate]` projection is unchanged. Initially measure separate
AWQ up and gate launches. If this overhead materially blocks an otherwise
non-regressive decode path, create a correctly ordered combined packed
`[up,gate]` representation, prove numerical equivalence, and use it for both
one-row and multi-row execution. Do not change attention, scheduler, KV
allocator semantics, timing definitions, or FP16 execution to favor W4.

Targeted tests must prove FP16 behavior is unchanged, packed components map to
the correct selected matrix, unselected layers remain FP16, 0/8/16/32 short
generations are finite and sane, no full-weight dequantization appears on
prefill/decode hot paths, and backend/config/artifact hashes are in run
metadata.

## Phase C: resource and model-performance gate

Before expensive serving, run two fresh full-shape profiles for each of
0/8/16/32 with v5's `max_batch_size=32`, `max_tokens_in_batch=49152`, block
size 16, and GPU utilization 0.99. Run the same 1024-token isolated prefill and
one-token decode model microbenchmark with at least seven measured samples
after warmup.

Record persistent allocation, packed representation bytes, profile peak and
temporary workspace, safe GPU KV blocks/token slots, 1024-token prefill
median/P95, and one-token decode median/P95. Investigate non-monotonic safe KV.
A W4 state with no real relief is excluded from serving but its evidence is
retained. W4-32 is always measured if it passes the resource and microbenchmark
gate.

A state is viable for Phase D only if profiles are repeatable within 5%, safe
KV blocks exceed FP16 by more than 5%, no hidden multi-GiB workspace appears,
and the measured compute regression is not already large enough to erase the
capacity mechanism. The final judgment uses the integrated serving telemetry,
not resident bytes alone.

## Phase D: frozen static gate

For every viable W4 state, rerun FP16 and the candidate using the exact v5
frozen 64-request scale-6 near-knee workload with two independent repeats per
state. Candidate results do not select a new favorable load. If capacity or
the curve needs context, also run the already-frozen neighboring scale-8 and
scale-4 workloads identically.

Regenerate the v5 raw-derived metrics: TTFT P50/P95/P99, strict >2 s SLO rate,
queueing, throughput, observed prefill/decode timing, waiting/running/swapped
queues, logical KV utilization, safe block count, swap/preemption, physical
HBM, request counts, and repeat variation.

A candidate is **ELIGIBLE** only if resource relief is repeatable; P95 TTFT
and/or SLO behavior is better than FP16 in both matched near-knee repeats; the
advantage exceeds the larger observed within-state run range (or 5% of FP16
mean when both ranges are zero); queue/KV/preemption telemetry supports the
relief mechanism; and no unreported dequantization/workspace cost exists.

Only if at least one state is eligible, rerun the frozen 106-request sequential
quality characterization and report paired DuReader F1 with 10,000-resample
bootstrap uncertainty plus token agreement/exact match/common-prefix
distortion against the matched FP16 run.

## Final decision

The report ends with `CONFIRMED FINDINGS`, `SUPPORTED BUT UNCERTAIN FINDINGS`,
`BLOCKED QUESTIONS`, `REMAINING UNCERTAINTY`, and `BACKEND DECISION: GO` or
`NO-GO`.

GO names exact eligible states, backend/config/packed representation, safe KV,
compute cost, repeated serving advantage, quality/distortion evidence, and the
minimal later runtime state set. If only W4-32 passes, recommend
`{FP16,W4-32}`. NO-GO names the smallest measured blocker and stops without
controller work.
