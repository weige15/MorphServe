# Common-engine end-to-end benchmark report

**Classification:** modified-condition measurement; not exact MorphServe paper reproduction.
**Source:** conference-final PDF adopted as primary in `docs/paper-version-delta.md`; the paper's exact serving inputs/configurations remain unavailable.

## Frozen protocol

- Model: local Llama 3.1 8B base snapshot `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`.
- GPU: RTX 3090, `CUDA_VISIBLE_DEVICES=1`; before/after GPU and process snapshots are saved per run.
- Prompt: deterministic 1,024-token payload, SHA-256 `30c15554a1cb39be1006f6b422e8e628da1e6058e33c91f5169eb16173f18b49`.
- Generation: exactly 512 tokens, EOS-independent, for every completed request.
- Conditions: FP16, static real AutoAWQ W4 G128 zero-point, and reconstructed MorphServe-default using the same SwiftLLM model path.
- Arrivals: frozen figure-inferred, systematically thinned manifests; 123 BurstGPT requests over 72 s and 94 Azure Code requests over 72 s. Context contents are not recovered paper contexts.
- Serving: independent arrival tasks, one GPU worker, continuous-batch prefill/decode, maximum five active requests, base KV capacity 512 blocks for FP16/MorphServe and 1,024 for static W4. MorphServe expands physical capacity only through observed controller actions.
- Warmup/model load/JIT are outside the timed origin. Raw request, token, system, controller, command, source-revision, GPU, process, and manifest sidecars are retained.

## Results

All six selected runs passed the runner gate: every request ID appears exactly once; all 123/94 requests completed without error or timeout; every output has exactly 512 token IDs; telemetry is nonempty; and final layer-transfer/executor state is clean.

| Workload | Condition | TTFT p50 / p95 / p99 (s) | TPOT p50 / p95 / p99 (s) | Output tok/s | Duration (s) | Peak KV capacity / used |
|---|---|---:|---:|---:|---:|---:|
| BurstGPT | FP16 | 149.386 / 286.232 / 298.326 | 0.02625 / 0.03778 / 0.05374 | 164.584 | 382.638 | 512 / 474 |
| BurstGPT | static real W4 | 238.018 / 438.980 / 458.019 | 0.03740 / 0.05442 / 0.08523 | 114.647 | 549.302 | 1,024 / 477 |
| BurstGPT | MorphServe-default | 368.262 / 750.289 / 782.281 | 0.06549 / 0.08083 / 0.09121 | 72.243 | 871.730 | 1,736 / 473 |
| Azure Code | FP16 | 106.205 / 194.879 / 207.382 | 0.02611 / 0.03596 / 0.05897 | 165.527 | 290.756 | 512 / 457 |
| Azure Code | static real W4 | 165.798 / 310.954 / 329.866 | 0.03767 / 0.05653 / 0.11029 | 114.727 | 419.502 | 1,024 / 470 |
| Azure Code | MorphServe-default | 281.815 / 579.144 / 614.722 | 0.06811 / 0.11112 / 0.19904 | 68.115 | 706.571 | 1,736 / 454 |

The 2-second TTFT SLO was exceeded by 118/123 BurstGPT requests and 89/94 Azure requests in each condition. This is an observed property of this saturated modified workload, not a paper claim or a claim of SLO reduction. Queue delay dominates TTFT in these runs.

The MorphServe controller recorded four pressure morphs and four recoveries per workload, reached eight simultaneous real W4 layers, and observed W4 module class `WQLinear_GEMM`. Peak physical KV capacity reached 1,736 blocks and final capacity returned to the 512-block base with no pending transfer events. Static W4 used 32 real quantized layers throughout its serving interval; the integrated static-W4 preflight records `AWQTransformerLayer` plus `WQLinear_GEMM` in its setup metadata.

## Artifacts and regeneration

- BurstGPT raw runs: `experiments/end-to-end-burstgpt/{fp16,static-w4,morphserve-default}/...`.
- Azure raw runs: `experiments/end-to-end-azure/{fp16,static-w4,morphserve-default}/...`.
- Pre-Azure audit: `results/raw/burstgpt-end-to-end-audit.json`; final six-run audit: `results/raw/end-to-end-audit.json`.
- Consolidated metrics: `figures/end-to-end/comparison.{json,csv}`.
- Regenerated plots: `figures/end-to-end/{ttft-percentiles,throughput,kv-capacity-occupancy,morphserve-controller}.{png,pdf}`.
- Plot file hashes and generator/input provenance: `results/raw/end-to-end-plot-provenance.json`.
- Plot regeneration: `/tmp/morphserve-analysis-venv/bin/python reproduction/scripts/plot_end_to_end_benchmark.py --root reproduction --output-dir reproduction/figures/end-to-end`.
- Failed/aborted runner attempts are preserved under `experiments/end-to-end-attempts/runner-path-bug/`; they are not selected result rows.

## Interpretation limits

These measurements do not reproduce the paper's L4/A100 hardware, exact model revision, task contexts, prompts, controller settings, request mapping, or paper result tables. They provide the first auditable common-engine end-to-end comparison under the frozen modified condition. No F1, ROUGE-L, perplexity, quality-gap, LLM-PQ, PyramidKV, or exact paper headline result is inferred from these token-level serving measurements. The supplied conference-final PDF contains no LLM-PQ/PyramidKV result claimed by the user objective; that distinction remains recorded in the claim register.
