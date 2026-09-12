# MorphServe Table 5 substrate rerun (v4)

Status: **substrate validation complete; exact Table 5 reproduction not claimed**.

This artifact namespace is separate from the existing `v2` static artifacts. It
runs only static `0/8/16/32` conditions. MorphServe dynamic adaptation,
layer swapping, and scheduler changes were not implemented.

## FP16 reference parity

The local checkpoint was inspected before choosing the loader behavior:

- checkpoint: local Llama 3.1 8B snapshot, exposed through the stable local
  alias `/tmp/morphserve-llama31-stable`;
- config: 32 layers, hidden size 4096, vocabulary 128256, `rope_theta=500000`,
  `rope_scaling.rope_type=llama3`, factor 8, and `tie_word_embeddings=false`;
- indexed weight keys: 291; both `model.embed_tokens.weight` and explicit
  `lm_head.weight` are present;
- resolved output head: `lm_head.weight`.

The previous loader classified any dict-valued `rope_scaling` as `llama3.2`
and consequently selected `model.embed_tokens.weight`. The old RoPE code also
used low/high frequency factors as position divisors rather than the Llama 3.1
frequency interpolation. The loader now resolves the head from actual keys and
the RoPE cache follows the Transformers Llama-3 formula. The CPU regression
test compares the inverse-frequency formula against the installed Transformers
implementation (zero observed difference).

`benchmark-results/table5-substrate-v4/substrate-evidence/fp16-hf-parity.json`
records a same-tokenizer, same-checkpoint comparison:

- first token: exact match (`20437`, decoded as ` Berlin`);
- first-step top-k overlap: 9/10;
- 8-token greedy continuation: exact match;
- decoded continuation: ` Berlin. The capital of Italy is Rome`.

This replaces the earlier repeated-token behavior as the quality-evaluation
substrate. The runnable check is:

```bash
PYTHONPATH=swiftLLM:swiftLLM/csrc CUDA_VISIBLE_DEVICES=GPU \
  /nfs/home/s314511048/.venv/bin/python -m benchmark.inference_parity \
  --model-path /tmp/morphserve-llama31-stable --max-new-tokens 8 --top-k 10
```

## W4 execution validation

Selected decoder matrices are NF4 weight-only quantized with blocksize 64.
The source-layout packed view is used by `bitsandbytes.gemv_4bit` for the
single-row decode path. The final implementation does **not** call
`dequantize_4bit` in `linear.py`, and the regression test checks that the
single-token path remains a low-bit GEMV.

The installed bitsandbytes 0.49.2 `matmul_4bit` multi-row wrapper internally
dequantizes its packed matrix before its fallback GEMM. That is explicitly
reported rather than called a fused low-bit prefill kernel. A Triton on-the-fly
NF4 GEMM candidate was tested and retained as
`w4-triton-candidate-rejected.json`; it was numerically finite but was roughly
10--31 ms for representative batch-32 matrices versus 0.1--0.4 ms for the
available prefill fallback, making it impractical for the 49,152-token profile.
The narrow practical choice is therefore genuine low-bit GEMV for decode plus
the explicitly labeled bitsandbytes prefill fallback.

`w4-matrix-sanity.json` records representative Q/O, K/V, up/gate, and down
matrices. All outputs are finite. Relative output L2 error against FP16 is
approximately 9.2% for both decode and prefill matrix checks. The same short
prompt generated a finite, sane continuation for model conditions 0, 8, 16,
and 32; those outputs are in `profile-generation-*.json`.

The model-level microbenchmark (`microbenchmark-*.json`, 1024-token prefill,
three samples) is:

| W4 layers | prefill median (ms) | one-token decode median (ms) |
|---:|---:|---:|
| 0 | 252.25 | 21.57 |
| 8 | 261.46 | 22.89 |
| 16 | 264.98 | 25.86 |
| 32 | 269.65 | 25.08 |

These numbers measure the actual data-plane calls and show that this W4 proxy
is not automatically faster end-to-end. Decode is executing the low-bit GEMV;
prefill remains the documented fallback.

## Resident memory and profiled KV capacity

The probe uses the same 32-batch/49,152-token profile shape as the static
runner. Resident model allocation after loading decreases monotonically, while
profile peak and resulting KV capacity do not:

| W4 layers | resident weights (GiB) | profile peak (GiB) | profiled GPU blocks | KV token slots |
|---:|---:|---:|---:|---:|
| 0 | 14.99 | 19.13 | 1768 | 28288 |
| 8 | 13.58 | 19.97 | 312 | 4992 |
| 16 | 12.16 | 18.56 | 568 | 9088 |
| 32 | 9.34 | 15.73 | 2040 | 32640 |

The W4-8 capacity regression is explained by temporary multi-row fallback
workspace during the profile: persistent weights are smaller, but the profile
peak is about 0.84 GiB higher than FP16. This is a profiling/workspace effect,
not evidence that W4-8 has larger resident weights. The raw probe files are
`profile-generation-*.json`; the v4 serving metadata independently records
1768/312/568/2040 blocks.

## FP16-only workload calibration

Calibration was performed before the v4 static comparison and used only FP16
runs. Raw runs and the derived summary are under
`benchmark-results/substrate-v3/calibration*` and
`benchmark-results/substrate-v3/calibration_summary.{json,csv}`. The successful
512-output run at 0.06 RPS had p50 TTFT 0.304 s and mean TTFT 2.108 s; the
shorter-output sweep showed the transition to queueing at 0.4--0.8 RPS. The
frozen v4 workload keeps the v2 prompt/reference mapping for the first 16
requests and multiplies v2 arrival offsets by 3.0. This choice was made from
FP16 evidence before inspecting any v4 W4 result; it avoids the earlier
catastrophic 106-request burst while retaining a bounded burst near the local
transition.

The calibration is local and is not a claim about the paper's hidden trace
scaling. The selected workload, hashes, and formula were frozen in
`table5-substrate-v4/protocol_manifest.json` before the four v4 runs.

## Fresh static results

All four v4 conditions use the same 16-request workload, 1024-token prompts,
512 requested output tokens, greedy decoding, scheduler, model alias, GPU
class, and measurement definitions. There was one completed run per condition;
all 16/16 requests completed.

| setting | W4 layers | F1 (%) | P95 TTFT (s) | TTFT >2 s |
|---|---:|---:|---:|---:|
| FP16 | 0 | 13.490 | 10.222 | 12.5% |
| W4 | 8 | 18.125 | 24.981 | 25.0% |
| W4 | 16 | 18.479 | 4.833 | 12.5% |
| W4 | 32 | 19.463 | 8.210 | 12.5% |

Raw request, telemetry, metadata, logs, and derived files are under
`benchmark-results/table5-substrate-v4/`. The table and plot were regenerated
with:

```bash
PYTHONPATH=swiftLLM \
  /nfs/home/s314511048/.venv/bin/python -m benchmark.analyze_static_quantization_quality_latency \
  --manifest benchmark-results/table5-substrate-v4/protocol_manifest.json \
  --runs benchmark-results/table5-substrate-v4/runs/v4-fp16_0 \
    benchmark-results/table5-substrate-v4/runs/v4-w4_8 \
    benchmark-results/table5-substrate-v4/runs/v4-w4_16 \
    benchmark-results/table5-substrate-v4/runs/v4-w4_32 \
  --output-dir benchmark-results/table5-substrate-v4
```

Paper Table 5 values remain reference markers only. The v4 numbers are not
numerically comparable to them because this is Llama 3.1 on an RTX 3090, uses
the Chinese public DuReader demo subset, uses a substituted BurstGPT interval,
and uses a uniform NF4 proxy rather than MorphServe's selective AWQ path.

## CONFIRMED FINDINGS

- SwiftLLM FP16 now matches Hugging Face Transformers on the same checkpoint:
  first token, top-k neighborhood, and the full recorded 8-token greedy
  continuation agree.
- The checkpoint's explicit `lm_head.weight` and Llama 3.1 RoPE parameters
  are detected from actual config/index evidence; they are no longer inferred
  from dict-valued `rope_scaling`.
- The FP16 output is non-degenerate on the parity prompt.
- One-token W4 decode dispatches to `bitsandbytes.gemv_4bit`; no full-weight
  dequantization call remains in that hot path.
- Representative W4 outputs are finite with bounded, recorded quantization
  error.
- 0/8/16/32 model-level short generations complete and are sane.
- Persistent loaded-weight allocation decreases with quantized-layer count.
- Fresh raw-derived 0/8/16/32 F1, P95 TTFT, and strict `TTFT > 2.0 s`
  results exist in a new v4 namespace; the old v2 artifacts were not changed.

## SUPPORTED BUT UNCERTAIN FINDINGS

- On this local workload, W4-16 has the lowest P95 TTFT, while W4-8 is the
  slowest because its reduced persistent weights do not offset prefill fallback
  workspace and KV pressure. W4-32 is between W4-16 and FP16 in P95 TTFT.
- F1 rises from the FP16 control to the W4 proxy conditions in this 16-request
  Chinese/base-model sample. This is non-monotonic with the paper's expected
  quality direction and must not be interpreted as a general quantization
  result.
- The v4 run is near the local FP16 transition rather than the old
  all-configurations-violate burst, but the selected 16-request sample and
  one repetition make the frontier directional evidence only.

## BLOCKED QUESTIONS

- The paper's exact Llama 3 8B checkpoint was not available; Llama 3.1 8B is a
  validated proxy.
- The English-translated DuReader subset is unavailable; the public Chinese
  demo records are used.
- The paper does not publish the raw 72-second BurstGPT interval; v4 uses the
  frozen first-16 subset of the existing selected segment with a documented
  scale of 3.0 relative to v2.
- The paper's LIS ordering is not published; layers are selected in fixed
  front-to-back order.
- Selective AWQ execution and its fused kernel are unavailable in this
  SwiftLLM environment. W4 results are uniform NF4 proxy results.
- Hardware is an RTX 3090 rather than the paper's L4.
- Installed bitsandbytes does not provide a fused low-bit multi-row GEMM in
  the used wrapper; only the decode GEMV gate is a genuine low-bit hot path.

## REMAINING UNCERTAINTY

The substrate is now suitable for a subsequent runtime-adaptation experiment:
FP16 correctness is no longer a confounder and single-token W4 execution has a
mechanism-valid low-bit path. Calling this an exact Table 5 reproduction would
still be unjustified because model/checkpoint, dataset language, trace interval,
LIS order, AWQ kernel, hardware, workload size, and repetition count differ.
The largest remaining systems uncertainty is the performance/quality effect of
replacing MorphServe selective AWQ and its prefill kernel with the uniform NF4
proxy and bitsandbytes prefill fallback.
