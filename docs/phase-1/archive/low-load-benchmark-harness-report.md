# Phase-1 SwiftLLM benchmark-harness report

Status: **PASS for the deliberately low-load validation only.** No saturation
sweep or overload mitigation was attempted.

## Deliverables and locations

The open-loop runner is `swiftLLM/benchmark/run.py`; pure workload/metric logic
is in `workload.py` and `metrics.py`; summary regeneration is
`swiftLLM/benchmark/summarize.py`. The runner writes machine-readable
`metadata.json`, `requests.jsonl`, `telemetry.jsonl`, and `summary.json`.

Validated runs are retained at:

- `benchmark-results/phase-1/baseline-low-load/runs/fixed-arrivals/`
- `benchmark-results/phase-1/baseline-low-load/runs/poisson-arrivals/`

The metadata records upstream SwiftLLM commit
`682cf9a28f97f7490409981a2f181528f377eb5d`, current MorphServe commit
`1dda4d9cd2041db4b96247f65bf5d553b8e1edf5`, a working-tree status and diff
fingerprint, the baseline EngineConfig, safe model path/identifier, CUDA device, seed, arrival mode, lengths, warmup policy, and run timestamps. The
working tree was intentionally dirty because the harness changes were not
committed at run time.

## Exact low-load reproduction command

From the repository root (the generated fixed run was made with this command):

```bash
MODEL=/nfs/home/s314511048/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
VENV=/nfs/home/s314511048/.venv
PYTHONPATH="$PWD/swiftLLM:$PWD/swiftLLM/csrc" CUDA_VISIBLE_DEVICES=3 \
  "$VENV/bin/python" -m benchmark.run \
  --model-path "$MODEL" \
  --target-rps 0.5 --arrival-mode fixed --request-count 4 \
  --prompt-token-count 8 --output-token-count 4 --seed 2025 \
  --telemetry-interval-s 0.1 \
  --output-dir benchmark-results/phase-1/baseline-low-load/runs \
  --run-id fixed-arrivals --expected-num-gpu-blocks 3880
```

The Poisson path was also exercised with three real requests:

```bash
PYTHONPATH="$PWD/swiftLLM:$PWD/swiftLLM/csrc" CUDA_VISIBLE_DEVICES=3 \
  "$VENV/bin/python" -m benchmark.run \
  --model-path "$MODEL" --target-rps 0.5 --arrival-mode poisson \
  --request-count 3 --prompt-token-count 8 --output-token-count 2 \
  --seed 7 --telemetry-interval-s 0.1 \
  --output-dir benchmark-results/phase-1/baseline-low-load/runs \
  --run-id poisson-arrivals --expected-num-gpu-blocks 3880
```

## Evidence from real runs

Fixed run (`fixed-arrivals`):

- 4/4 requests completed; every prompt measured 8 tokens and every output had
  4 tokens.
- Planned offsets were exactly `0, 2, 4, 6` seconds; the launcher created
  asynchronous tasks without awaiting earlier completion.
- `num_gpu_blocks` was 3880 in initialization and every telemetry sample;
  `gpu_kv_token_slots` was 62,080.
- 80 telemetry samples populated queue/running/KV fields and physical memory
  fields. Peak running count was 4, peak logical KV utilization was
  `0.00103093`, and peak waiting depth was 0. No swap/preemption occurred.
- Summary: completed throughput `0.5046` requests/s, TTFT mean `4.8042` s
  (P50 `4.8042`, P95 `7.5039`, P99 `7.7438`), queueing mean `0.002153` s,
  TPOT mean `0.04004` s (P95 `0.04007`, P99 `0.04007`), and 16 generated
  tokens (`2.0185` tokens/s). The high first-token time is retained as observed
  baseline behavior; it was not optimized.

Poisson run (`poisson-arrivals`):

- 3/3 requests completed with fixed 8-token prompts and 2-token outputs.
- Seeded planned offsets were `0`, `0.782629688`, and `1.109666602` seconds;
  telemetry observed a waiting depth of 2 while the first request was running,
  demonstrating that arrival did not wait for completion.
- 105 telemetry samples populated the same fields. `num_gpu_blocks` stayed
  3880; peak running count was 3, peak logical KV utilization `0.00077320`,
  and swap/preemption counts were zero.

For both runs, raw request timestamps satisfy
`arrival <= scheduler eligibility <= first prefill <= first engine output <=
first streamed output <= engine completion <= streamed completion`. Prompt and
output counts, status, telemetry population, and the saved summaries were
checked from the raw files.

## Summary regeneration and tests

Summary regeneration was verified with:

```bash
PYTHONPATH="$PWD/swiftLLM" "$VENV/bin/python" -m benchmark.summarize \
  benchmark-results/phase-1/baseline-low-load/runs/fixed-arrivals
PYTHONPATH="$PWD/swiftLLM" "$VENV/bin/python" -m benchmark.summarize \
  benchmark-results/phase-1/baseline-low-load/runs/poisson-arrivals
```

The regenerated JSON was byte-for-byte equivalent to each saved `summary.json`.
The final telemetry timestamp was also verified to be no later than the saved
measurement end timestamp, and each raw request now includes actual arrival
offset and launch jitter.
CPU-only tests and syntax checks passed:

```bash
PYTHONPATH="$PWD/swiftLLM" "$VENV/bin/python" \
  -m unittest benchmark.test_benchmark -v       # 6 tests, OK
PYTHONPATH="$PWD/swiftLLM:$PWD/swiftLLM/csrc" "$VENV/bin/python" \
  -m py_compile swiftLLM/benchmark/*.py \
  swiftLLM/swiftllm/server/engine.py swiftLLM/swiftllm/server/structs.py
git diff --check
```

The unit tests cover fixed and seeded Poisson schedule generation, duration
bounds, TTFT/queueing/TPOT formulas, one-token TPOT, and summary regeneration
from saved raw files.

## Prompt-to-artifact completion checklist

| Objective requirement | Concrete evidence |
|---|---|
| Preserve the validated unquantized Llama 3.1 8B baseline on CUDA device 3 | Both metadata files record the safe model snapshot, `CUDA_VISIBLE_DEVICES=3`, baseline EngineConfig, `num_gpu_blocks=3880`, and 62,080 GPU KV token slots; both runs initialized and completed real inference. |
| Open-loop fixed and Poisson arrivals independent of completion | `benchmark/workload.py`, the non-awaiting launcher in `benchmark/run.py`, CPU schedule tests, and both real run request offsets; the Poisson run reached waiting depth 2 while the first request ran. |
| Fixed/configurable prompt and output lengths with streaming Engine path | CLI fields, all raw rows with equal observed lengths, and `Engine.add_request_and_stream` in the real runs. |
| Per-request identifiers, arrival/eligibility/prefill/output/completion timestamps and formulas | `requests.jsonl`, `metadata.timestamp_definition`, `metadata.metric_formula`, derived metrics, and raw timestamp-order checks. |
| Queue/KV/swap/throughput/memory telemetry without using HBM as the saturation signal | `telemetry.jsonl` contains all requested queue, block, logical-utilization, counter, throughput, and `torch.cuda.mem_get_info` fields; summaries report the logical KV peak separately. |
| Reproducible metadata and raw machine-readable outputs | Each run directory contains metadata, JSONL raw files, and summary; model path is safe/redacted, and commit/status/diff provenance is recorded. |
| Summary regeneration from saved raw data | `benchmark.summarize` was run for both directories and its output was byte-identical to `summary.json`. |
| Low-load validation and CPU-only tests | 4-request fixed and 3-request Poisson real runs were all successful; six unit tests, compilation, diff check, raw checks, and timestamp-order checks passed. |
| Scheduler behavior unchanged and audit documented | `git diff -- swiftLLM/swiftllm/server/scheduler.py` is empty; only timestamp/counter observations are present in Engine/Request, as detailed below. |
| Explicitly avoid MorphServe/quantization/KV resizing/new scheduling/mitigation | No such files or logic were added; baseline precision, capacity, and scheduling policy remain upstream. |

## Scheduler-semantics audit

`git diff -- swiftLLM/swiftllm/server/scheduler.py` is empty. The only
control-plane modifications are in `server/engine.py` and are observational:

1. `Request` receives optional benchmark metadata; it is never read by
   admission, ordering, preemption, swapping, or model execution.
2. Engine timestamps are assigned after tokenization before the existing
   `Scheduler.on_requests_arrival` call, immediately before the existing model
   forward, and immediately after it returns.
3. Swap counters observe the existing swap lists. Materializing the existing
   reversed swap-out iterator once preserves the upstream IDs and model call;
   it does not alter scheduler decisions.
4. `get_benchmark_snapshot()` only reads existing scheduler queue/block state.

`server/structs.py` only carries optional metadata and timestamps. No scheduler,
EngineConfig capacity, precision, KV policy, inference argument, FCFS order,
admission condition, preemption policy, or swap policy was changed. Physical
HBM telemetry is deliberately supplementary; logical KV block occupancy and
waiting/running/swapped queue state are the mandatory saturation signals.
