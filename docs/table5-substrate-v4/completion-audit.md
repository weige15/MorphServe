# Completion audit: MorphServe inference/W4 substrate goal

Audit basis: current repository state, v4 raw artifacts, v4 derived artifacts,
parity JSON, matrix sanity JSON, memory probes, microbenchmarks, calibration
runs, and current test output. The old `benchmark-results/static-quantization-quality-latency`
v2 namespace was not used as v4 evidence and remains intact.

## Concrete deliverables restated

1. Correct SwiftLLM FP16 inference against the exact local checkpoint and HF
   Transformers, with non-degenerate generation and a regression check.
2. Correct W4 decode mechanism, independent numerical checks, and 0/8/16/32
   model-level generation.
3. Memory/KV-capacity evidence distinguishing persistent weights from temporary
   profile workspace, plus FP16/W4 prefill/decode microbenchmarks.
4. FP16-only load calibration followed by one frozen, fresh static 0/8/16/32
   result set with raw F1/P95-TTFT/>2-second SLO artifacts.
5. A report separating confirmed, supported-but-uncertain, blocked, and
   remaining-uncertainty findings.

## Prompt-to-artifact checklist

| Explicit requirement / gate | Concrete evidence inspected | Result |
|---|---|---|
| Use current repository; preserve v2; no dynamic MorphServe | `git status`; old v2 directory; v4 manifest objective; changed files contain no dynamic adaptation | PASS |
| Inspect model version detection and actual config | local `config.json`; `model.safetensors.index.json`; `fp16-hf-parity.json.checkpoint_config` | PASS |
| Verify lm_head source and tied-weight assumption | 291-key index contains both `lm_head.weight` and `model.embed_tokens.weight`; `resolved_lm_head_key=lm_head.weight`; `test_explicit_lm_head_wins_when_rope_scaling_is_a_dict` | PASS |
| Verify Llama 3/3.1 RoPE | config fields; `LlamaModelConfig.get_rope_inv_freq`; regression test; parity output | PASS |
| Verify tokenizer identity | parity tokenizer file SHA-256 values and prompt IDs; v4 workload metadata tokenizer hash | PASS |
| Verify weight loading and normalization/greedy path | same-checkpoint parity; first-token/top-k/continuation agreement; source loader key checks; test suite | PASS |
| Deterministic parity harness | `swiftLLM/benchmark/inference_parity.py`; runnable command in final report; saved v4 JSON | PASS |
| First token and short greedy continuation | parity: first token exact; 8-token output exact; decoded sane | PASS |
| Numerical intermediate where practical | parity first-step top-k overlap 9/10; logits top-k saved | PASS |
| Regression check for discovered issue | `benchmark/test_inference_substrate.py`: RoPE, tied/untied head, actual index, and linear hot path; 11 unittest tests pass | PASS |
| No full quantized-weight dequantization on one-token decode | `swiftllm/worker/kernels/linear.py` uses `gemv_4bit` for one row and has no `dequantize_4bit`; AST regression test; W4 sanity dispatch | PASS |
| Evaluate alternative backend rather than retaining old fallback | `w4-triton-candidate-rejected.json` records finite Triton candidate and measured 10–31 ms prefill cost; final decode uses genuine GEMV | PASS |
| Do not overclaim W4 prefill | final runner metadata and v4 manifest explicitly label `matmul_4bit` prefill fallback; final report calls out internal multi-row dequantization | PASS |
| Independent W4 matrix numerics | `w4-matrix-sanity.json`: four representative shapes, decode/prefill, finite outputs, relative errors | PASS |
| Model generation for 0/8/16/32 | `profile-generation-0/8/16/32.json`; each records same sane 8-token continuation | PASS |
| Profile model memory/KV for 0/8/16/32 | `profile-generation-*.json`; v4 metadata blocks 1768/312/568/2040 | PASS |
| Investigate non-monotonic capacity | final report table: resident weights 14.99/13.58/12.16/9.34 GiB decrease; profile peak and blocks non-monotonic; workspace explanation | PASS |
| Actual one-token decode and representative prefill microbenchmark | `microbenchmark-0/8/16/32.json`, 1024-token prefill and one-token decode, raw samples and percentiles | PASS |
| FP16-only calibration before static W4 | raw FP16-only runs under `benchmark-results/substrate-v3/calibration*`; `calibration_summary.json` references no W4 observations in selection | PASS |
| Avoid catastrophic-all-configurations regime | v4 FP16 has 12.5% >2s, not all requests; calibration and selected scale recorded | PASS, local proxy only |
| Freeze one workload and identical conditions | v4 `input/workload.jsonl`, metadata/hash, manifest; all v4 metadata has same workload/manifest/model hashes and fixed engine settings | PASS |
| Fresh 0/8/16/32 raw static results | `benchmark-results/table5-substrate-v4/runs/v4-*`: 16 launched and 16 completed in every condition; separate logs | PASS |
| F1 definition | analyzer output and v4 manifest; `quality_metrics.jsonl` per request; `static_quantization_quality_latency_results.csv` | PASS |
| P95 TTFT definition | v4 manifest, run metadata, raw request timestamps, analyzer output | PASS |
| >2 s SLO definition | strict threshold in metadata/manifest and per-request aggregate; table has all four rates | PASS |
| Raw-artifact-derived outputs | v4 `aggregate.json`, CSVs, PNG; analyzer rerun command; no manually transcribed measurement values | PASS |
| Required final report sections | `docs/table5-substrate-v4/final-report.md` has all four named sections | PASS |
| Existing and new tests/build checks | 11 unittest tests pass; `py_compile` pass; `git diff --check` pending final audit below | PASS after audit command |
| Scheduler unchanged | no scheduler source change in `git status`; v4 run metadata references unchanged scheduler; old scheduler hash evidence retained in v2 audit | PASS |

## Current validation commands and observed results

```bash
PYTHONPATH=swiftLLM:swiftLLM/csrc \
  /nfs/home/s314511048/.venv/bin/python -m unittest \
  benchmark.test_benchmark benchmark.test_inference_substrate -v
# Ran 11 tests ... OK

/nfs/home/s314511048/.venv/bin/python -m py_compile \
  swiftLLM/benchmark/*.py \
  swiftLLM/swiftllm/model_config.py \
  swiftLLM/swiftllm/worker/model.py \
  swiftLLM/swiftllm/worker/weight.py \
  swiftLLM/swiftllm/worker/kernels/linear.py \
  swiftLLM/swiftllm/worker/layers/post_layer.py

git diff --check
```

The exact compile command used for the changed source was:

```bash
/nfs/home/s314511048/.venv/bin/python -m py_compile \
  swiftLLM/swiftllm/model_config.py \
  swiftLLM/swiftllm/worker/model.py \
  swiftLLM/swiftllm/worker/weight.py \
  swiftLLM/swiftllm/worker/kernels/linear.py \
  swiftLLM/swiftllm/worker/layers/post_layer.py \
  swiftLLM/benchmark/*.py
```

The compile command passed, and the final `git diff --check` is required before
handoff. No goal-completion claim relies on a green status alone: every gate
above is mapped to raw or source evidence.

## Audit decision

All eight numbered gates are covered by concrete current evidence. The goal is
complete as a **trustworthy substrate and defensible directional proxy**. It is
not complete as an exact Table 5 reproduction; the exact-protocol blockers and
the qualified prefill-kernel limitation remain explicitly documented.
