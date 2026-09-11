# Static MorphServe Table 5 reproduction report

Status: **complete as a proxy reproduction; exact Table 5 protocol is not available locally**.

The experiment is restricted to the four static conditions in Table 5. It does not run MorphServe dynamic adaptation, KV resizing, or other paper figures.

## Protocol and provenance

The paper targets are reference values only:

| setting | paper F1 (%) | paper P95 TTFT (s) | paper SLO violation |
|---|---:|---:|---:|
| 0 layers / FP16 | 25.19 | 5.65 | 12.7% |
| 8 layers / selective | 24.16 | 2.61 | 4.2% |
| 16 layers / selective | 23.93 | 1.74 | 0% |
| 32 layers / full quant. | 23.66 | 1.62 | 0% |

Source evidence and the frozen pre-run specification are in [`experiment-spec.md`](experiment-spec.md). The machine-readable manifest is [`benchmark-results/table5-reproduction/protocol_manifest.json`](../../benchmark-results/table5-reproduction/protocol_manifest.json).

The final v2 workload uses the first request-timestamp-started 72-second half-open BurstGPT window with at least 100 rows: raw `[744286, 744358)`, 106 requests, and 1.75x time scaling. Scaled offsets span 0–124.25 seconds. Every request uses a fixed 1024-token prompt and 512 requested output tokens. The exact trace rows, DuReader records, prompts, and hashes are under `benchmark-results/table5-reproduction/input/`.

All final v2 conditions use the same local Llama-3.1-8B snapshot/tokenizer, fixed workload, greedy decoding, seed 2025, SwiftLLM engine/scheduler, RTX 3090 device 3, software environment, and measurement code. The paper's exact Llama 3 checkpoint, L4 hardware, English-translated DuReader subset, raw Table 5 interval, and LIS order were unavailable. Layers are therefore selected once in fixed front-to-back order `[0, ..., N-1]`.

AWQ was attempted through public AutoAWQ source (`0.2.9`); selective AWQ fused execution was unavailable in this SwiftLLM path. All W4 conditions use the same `bitsandbytes` NF4 weight-only implementation (`blocksize=64`, no nested statistics), explicitly labeled a **uniform W4 proxy**, not AWQ.

## Final measured table

Values below are regenerated from raw request files by `benchmark.table5_analyze`. Each condition has two independent runs; the table reports the run mean and min–max where useful.

| setting | quantized layers | F1 (%) mean | P95 TTFT (s) mean [min–max] | SLO violation mean [min–max] | requests/run | completed |
|---|---:|---:|---:|---:|---:|---:|
| FP16 | 0 | 0.3208 | 92.55 [92.24–92.85] | 84.43% [83.96–84.91%] | 106 | 106 |
| W4 selective | 8 | 0.2955 | 113.02 [112.49–113.55] | 87.26% [86.79–87.74%] | 106 | 106 |
| W4 selective | 16 | 0.2901 | 72.86 [72.75–72.97] | 91.51% [91.51–91.51%] | 106 | 106 |
| W4 full | 32 | 0.3489 | 84.87 [84.66–85.08] | 91.51% [91.51–91.51%] | 106 | 106 |

The F1 values are percentage points from macro per-request character-overlap F1. They are not numerically comparable to the paper's values because this run uses the Chinese public demo data, a base Llama-3.1 checkpoint, and a deliberately labeled W4 proxy. The low absolute F1 is retained rather than hidden.

The corresponding raw-derived files are:

- [`table5_results.csv`](../../benchmark-results/table5-reproduction/table5_results.csv)
- [`configuration_summary.csv`](../../benchmark-results/table5-reproduction/configuration_summary.csv)
- [`aggregate.json`](../../benchmark-results/table5-reproduction/aggregate.json)
- [`execution_commands.json`](../../benchmark-results/table5-reproduction/execution_commands.json)
- [`quality_vs_p95_ttft.png`](../../benchmark-results/table5-reproduction/quality_vs_p95_ttft.png)
- per-run `metadata.json`, `requests.jsonl`, `telemetry.jsonl`, `summary.json`, and `quality_metrics.jsonl` under `benchmark-results/table5-reproduction/runs/v2-*`.

## Claim assessment

1. **FP16 gives highest/near-highest quality but worst latency/SLO:** **not confirmed.** FP16 quality exceeds W4-8 and W4-16, but W4-32 is higher in this proxy. FP16 has lower P95 TTFT than W4-8 but higher P95 TTFT than W4-16/W4-32. FP16 has lower SLO violation than every W4 condition in this saturated workload.
2. **More W4 layers usually improve P95 TTFT and/or SLO compliance:** **partially supported for P95 TTFT only.** W4-16 and W4-32 are faster than FP16, while W4-8 is slower. SLO compliance does not improve: all conditions violate the 2-second SLO for most requests, and W4-16/W4-32 have the highest measured violation rate.
3. **More quantization measurably degrades quality:** **partially supported but non-monotonic.** W4-8 and W4-16 are below FP16 by 0.0253 and 0.0306 F1 points respectively; W4-32 is above FP16 by 0.0281 points. The monotonic claim is not supported.
4. **The four conditions expose a quality–latency Pareto trade-off:** **supported as a proxy-level finding, not an exact Table 5 reproduction.** FP16 versus W4-16 shows a quality decrease with lower P95 TTFT, and W4-8/W4-32 show distinct quality/latency points. However, W4-32's higher quality and the poor SLO behavior mean the expected monotonic frontier is not reproduced.

## CONFIRMED FINDINGS

- All four static configurations have auditable raw measurements: 8 runs total, 106/106 completed requests per run, no fabricated or manually entered request metrics.
- The exact same v2 workload and measurement definitions were used across conditions; only the quantized-layer count changed.
- The 2-second SLO is violated by most requests in the selected high-burst workload for every condition.
- The fixed front-to-back layer order and one shared NF4 W4 implementation are recorded in each run manifest/metadata.
- The aggregate table and plot regenerate from raw per-request files.

## SUPPORTED BUT UNCERTAIN FINDINGS

- W4-16 and W4-32 reduce P95 TTFT relative to FP16 on this RTX 3090 proxy, while W4-8 does not.
- Selective W4-8/W4-16 reduce measured F1 relative to FP16, but the absolute scores are extremely low and the full W4 condition reverses the direction.
- The experiment demonstrates a workload-dependent quality/latency trade-off, but not the paper's claimed monotonic static frontier.

## BLOCKED QUESTIONS

- The exact Table 5 LIS layer sequence is not published in the supplied paper. Recovering it would require the full MorphServe profiling implementation, outside this scope.
- Selective per-layer AWQ could not be executed reliably in the available SwiftLLM engine. AWQ results are therefore blocked; the reported W4 results are explicitly a proxy.
- The paper does not publish the raw BurstGPT 72-second bounds. The deterministic first-eligible high-density window is a documented substitution.
- The paper's English-translated DuReader subset and the exact Llama 3 (rather than local Llama 3.1) checkpoint were unavailable.

## REMAINING UNCERTAINTY

Absolute differences from Table 5 are dominated by checkpoint identity (Llama 3.1 base vs paper Llama 3), Chinese demo vs English-translated DuReader, the substituted BurstGPT interval, RTX 3090 vs L4 hardware, SwiftLLM plus NF4 proxy vs MorphServe/AWQ kernels, unavailable LIS order, and finite two-run variability. Quantization also changes SwiftLLM's profiled KV capacity non-monotonically (`1768`, `1646`, `2876`, `3928` GPU blocks for 0/8/16/32), reflecting proxy/kernel memory overhead rather than a paper-faithful resource profile. The v1 workload/prompt-construction runs are retained under `input-v1`, `runs-v1`, and `logs-v1`; v2 is final because it preserves the answer cue after discovering that prefix-only truncation could remove it.

## Reproduction and regeneration

Prepare the frozen v2 workload:

```bash
cat /tmp/DuReader/DuReader-2.0/data/demo/devset/search.dev.json \
    /tmp/DuReader/DuReader-2.0/data/demo/trainset/search.train.json \
    > /tmp/DuReader/demo_dev_train.jsonl
PYTHONPATH=swiftLLM /nfs/home/s314511048/.venv/bin/python -m benchmark.table5_prepare \
  --trace /tmp/BurstGPT/data/BurstGPT_1.csv \
  --dureader /tmp/DuReader/demo_dev_train.jsonl \
  --tokenizer /nfs/home/s314511048/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b \
  --output-dir benchmark-results/table5-reproduction/input
```

Run one condition with the command template in `protocol_manifest.json` (change only the condition, layer count, and run ID). Regenerate the final aggregate and plot:

```bash
PYTHONPATH=swiftLLM /nfs/home/s314511048/.venv/bin/python -m benchmark.table5_analyze \
  --manifest benchmark-results/table5-reproduction/protocol_manifest.json \
  --runs \
    benchmark-results/table5-reproduction/runs/v2-fp16_0-rep0 \
    benchmark-results/table5-reproduction/runs/v2-fp16_0-rep1 \
    benchmark-results/table5-reproduction/runs/v2-w4_8-rep0 \
    benchmark-results/table5-reproduction/runs/v2-w4_8-rep1 \
    benchmark-results/table5-reproduction/runs/v2-w4_16-rep0 \
    benchmark-results/table5-reproduction/runs/v2-w4_16-rep1 \
    benchmark-results/table5-reproduction/runs/v2-w4_32-rep0 \
    benchmark-results/table5-reproduction/runs/v2-w4_32-rep1 \
  --output-dir benchmark-results/table5-reproduction
```

No Table 5 value is used as a measured input; paper values appear only as reference markers and comparison targets.
