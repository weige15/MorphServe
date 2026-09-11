# Static quantization benchmark completion audit

Audit date: 2026-09-11 UTC. This audit uses the v2 final artifacts, not the earlier v1 pilot artifacts.

## Concrete deliverables

The objective requires either all four static configurations with auditable measurements and claim assessment, or a concrete blocker report preserving setup/proxy evidence. The current state has all four measured conditions plus explicitly labeled blockers.

| Requirement | Evidence | Status |
|---|---|---|
| Evidence-backed specification constructed before benchmarking | `docs/static-quantization-quality-latency/static-quantization-benchmark-protocol.md`; paper SHA and Table 5 targets in `protocol_manifest.json` | PASS |
| Scope restricted to 0/8/16/32 quantized decoder layers | `protocol_manifest.json` conditions; eight `v2-*` raw runs; no dynamic condition | PASS |
| Paper values treated as reference targets | `static-quantization-benchmark-report.md` and `aggregate.json.paper_targets_are_reference_only=true` | PASS |
| F1 quality, P95 TTFT, and TTFT >2 s SLO defined | manifest protocol; `analyze_static_quantization_quality_latency.py`; per-request `quality_metrics.jsonl` | PASS |
| Same model/tokenizer/workload/engine/scheduler/hardware/software/generation settings | final metadata comparison in `completion_audit.txt`: fixed fields match and only condition/layer count differ | PASS |
| 1024-token prompts and 512-token requests | every final raw request row validated; 106/106 per run | PASS |
| BurstGPT public source/revision and selected 72 s interval recorded | manifest: HPMLL/BurstGPT revision, file hash, raw `[744286,744358)`, 106 rows, 1.75 scale | PASS (interval is an explicit documented substitution) |
| DuReader public source/revision, mapping, references, and hashes recorded | manifest and `input/dureader_selected.jsonl`; 106 answer-bearing records | PASS (Chinese demo substitution) |
| AWQ attempted | AutoAWQ 0.2.9 public source import and blocker recorded in manifest/report | PASS as an attempt; exact AWQ execution BLOCKED |
| One consistent W4 implementation across all quantized conditions | all W4 metadata: bitsandbytes NF4, blocksize 64, same proxy label | PASS |
| LIS ordering attempted/recovered without full MorphServe | paper formula inspected; exact sequence unavailable; fixed front-to-back fallback recorded | PASS (exact sequence BLOCKED) |
| Machine-readable manifest recorded before final runs | `benchmark-results/static-quantization-quality-latency/protocol_manifest.json`; final metadata hash matches it | PASS |
| GPU/software/checkpoint/library identity recorded | manifest hardware/software/model hashes and public revisions | PASS |
| Raw per-request TTFT/SLO/config/output/reference artifacts saved | 8 final run directories, each `metadata.json`, `requests.jsonl`, `telemetry.jsonl`, `summary.json`, `quality_metrics.jsonl`; all 106 complete | PASS |
| Exact final command lines and logs saved | `benchmark-results/static-quantization-quality-latency/execution_commands.json` and 8 v2 log files | PASS |
| Raw telemetry and failure information saved | per-run telemetry, logs, summaries; no final run failures; v1 failed/setup evidence retained separately | PASS |
| Enough repetitions to inspect run noise | two runs for every final configuration; summary min/max and repeat IDs preserved | PASS |
| Four-row-equivalent result and quality-vs-P95 plot | `static_quantization_quality_latency_results.csv`, `configuration_summary.csv`, `quality_vs_p95_ttft.png` | PASS |
| Derived values regenerated from raw artifacts | exact analyzer command in `static-quantization-benchmark-report.md`; analyzer rerun during this audit | PASS |
| Claim (1) tested without forcing the trend | report says NOT CONFIRMED; FP16 is not highest quality and does not have worst result on every metric | PASS |
| Claim (2) tested without forcing the trend | report says only P95 partial support; SLO worsens/non-monotonic | PASS |
| Claim (3) tested without forcing the trend | report says selective degradation partial, full W4 reverses it | PASS |
| Claim (4) tested | report documents proxy-level Pareto evidence and non-monotonicity | PASS |
| Iterative uncertainty reduction performed | v1 prompt-construction issue was identified before final v2; v2 fixed prompt construction to preserve the answer cue and reran all conditions | PASS |
| Final report has required separated sections | `static-quantization-benchmark-report.md`: CONFIRMED FINDINGS, SUPPORTED BUT UNCERTAIN FINDINGS, BLOCKED QUESTIONS, REMAINING UNCERTAINTY | PASS |
| Existing tests and code gates run | `completion_audit.txt`: 6 unittest tests OK, py_compile pass, git diff --check pass | PASS |
| Plot opens and scheduler remains unchanged | `completion_audit.txt`: PNG verify pass; scheduler current/HEAD SHA identical | PASS |

## Final result from raw artifacts

The final v2 configuration means are:

| condition | F1 (%) | P95 TTFT (s) | SLO violation |
|---|---:|---:|---:|
| FP16 / 0 | 0.3208 | 92.55 | 84.43% |
| W4 / 8 | 0.2955 | 113.02 | 87.26% |
| W4 / 16 | 0.2901 | 72.86 | 91.51% |
| W4 / 32 | 0.3489 | 84.87 | 91.51% |

These are generated values, not Table 5 transcriptions. The exact run-level values and ranges are in `aggregate.json` and `configuration_summary.csv`.

## Uncovered exact-protocol requirements and blockers

The following cannot be silently marked reproduced:

1. The supplied paper does not disclose its raw 72-second BurstGPT bounds.
2. The exact LIS ordering is not published and the full MorphServe profiling implementation is out of scope.
3. Selective AWQ could not be run in the available SwiftLLM engine; W4 results are a uniform NF4 proxy.
4. The local checkpoint is Llama 3.1 8B rather than the paper's Llama 3 8B, and the GPU is an RTX 3090 rather than an L4.
5. The paper's English-translated DuReader subset is unavailable; the official Chinese demo dev/train records are used.

These blockers are precisely stated in the final report. They affect exact numerical fidelity, but do not block the requested strongest reproducible proxy evidence because all four static conditions were run under one auditable protocol.

## Completion decision

**Complete under the objective's alternative completion clause:** all four static configurations have auditable measurements, raw-derived aggregates/plot, repeated runs, and a claim-by-claim assessment. Exact Table 5 reproduction is not claimed; all known protocol substitutions and blocked questions are preserved.
