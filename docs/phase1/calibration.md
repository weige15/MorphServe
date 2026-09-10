# Phase-1 SwiftLLM KV-capacity calibration

Status: **PASS — a reproducible KV-pressure pilot was found.** The final publication-style RPS sweep was **not** run in this calibration.

## Objective and interpretation policy

The existing short-workload result remains a confirmed negative result for KV saturation: the approximately 35 RPS completed-throughput plateau occurred at only approximately **0.206%** peak logical KV utilization, with no swaps or preemptions. It is not reinterpreted as memory saturation.

This calibration sought one small, fixed workload that makes SwiftLLM's logical KV admission constraint observable before `max_batch_size` or another independent configured limit. The selected pilot reaches a strong KV-pressure region and has direct admission evidence; large TTFT is not used as the success criterion by itself.

## Inspected baseline EngineConfig

The values below were read from `swiftLLM/examples/online.py`, the benchmark defaults in `swiftLLM/benchmark/run.py`, and the prior baseline reports:

| field | validated Phase-1 baseline |
|---|---:|
| `block_size` | 16 |
| `gpu_mem_utilization` | 0.99 |
| `num_gpu_blocks` | 3880 (profiled) |
| `num_cpu_blocks` | 1024 |
| `max_seqs_in_block_table` | 128 |
| `max_blocks_per_seq` | 3072 |
| `max_batch_size` | 4 |
| `max_tokens_in_batch` | 1024 |

The baseline GPU KV capacity is therefore:

- `3880 * 16 = 62,080` logical GPU KV token slots.
- CPU swap capacity is `1024 * 16 = 16,384` token slots.
- For `C` resident sequences, the approximate total context target is `62,080 / C` tokens per sequence.

Useful baseline planning values were:

| concurrent sequences `C` | approximate total context per sequence |
|---:|---:|
| 4 | 15,520 tokens |
| 8 | 7,760 tokens |
| 20 | 3,104 tokens |

Thus a 15,500-token output is not generally the correct target: prompt tokens must be included, and the target depends on `C`.

With the baseline `max_batch_size=4` and `max_tokens_in_batch=1024`, an output-only route to KV pressure requires tens of thousands of generated tokens. Two such attempts exceeded the safe 30-minute pilot limit before producing a completed raw run (details below). No saturation claim is made from those aborted attempts.

## Candidate pilots

All successful pilots used the unchanged SwiftLLM scheduler and the same FP16 Llama-3.1-8B model on `CUDA_VISIBLE_DEVICES=3`.

| pilot | prompt / requested output | EngineConfig / offered pattern | observed result and bottleneck |
|---|---:|---|---|
| Existing baseline short workload | 8 / 4 tokens | baseline config; prior fixed/open-loop saturation sweep | Confirmed negative KV result: approximately 0.206% peak logical KV utilization, no swap/preemption; the throughput plateau is compute-bound, not memory-bound. |
| `phase1-kv-calibration-c20-short` | 50 / 128 tokens | `max_batch_size=32`, `max_tokens_in_batch=1024`, fixed 1000 RPS, 20 requests | 20/20 completed; peak 20 running, waiting 0, 220/3880 blocks = **5.67%**; no swaps. This is a fast low-pressure control, not KV saturation. TTFT mean 10.901 s, queueing mean 0.098 s, TPOT mean 0.02718 s. |
| Baseline long-output attempt | 128 / 7616 tokens | `max_batch_size=16`, `max_tokens_in_batch=1024`, fixed 1000 RPS, 8 requests | Process exceeded 1800 s before finalization; no raw telemetry/request artifact was written. It is treated only as a practical-runtime failure, not as evidence of achieved capacity. |
| Baseline long-output attempt | 50 / 3054 tokens | `max_batch_size=32`, `max_tokens_in_batch=1024`, fixed 1000 RPS, 20 requests | Process exceeded 1800 s before finalization; no raw telemetry/request artifact was written. It is treated only as a practical-runtime failure, not as evidence of achieved capacity. |
| **Selected: `phase1-kv-calibration-long-prompt-c13-mem`** | **2048 / 16 tokens** | **documented separate config below; fixed 100 RPS, 20 requests** | **13 resident requests, 7 waiting while pressure is high, 1677/1768 blocks = 94.85%; max batch is 32. No swap/preemption. Direct KV admission pressure.** |

The two baseline long-output attempts were not used to claim that KV capacity was or was not reached. Their logs contain initialization only, and their incomplete run directories contain no measurement files.

## Selected fixed configuration and admission calculation

The selected pilot deliberately uses a separate, documented `EngineConfig` because retaining baseline `max_tokens_in_batch=1024` made a practical high-context output pilot exceed the safe runtime limit. Scheduler code and semantics were not changed.

| field | selected value |
|---|---:|
| `block_size` | 16 |
| `gpu_mem_utilization` | 0.99 |
| `num_gpu_blocks` | 1768 (profiled and observed in all telemetry) |
| `num_cpu_blocks` | 4096 |
| `max_seqs_in_block_table` | 128 |
| `max_blocks_per_seq` | 3072 |
| `max_batch_size` | 32 |
| `max_tokens_in_batch` | 49152 |

The selected configuration has `1768 * 16 = 28,288` logical GPU KV token slots. The larger profiling batch/token budget changes the profiled GPU-block count; this is reported explicitly rather than being presented as the 3880-block baseline.

The fixed workload is:

- 20 requests, fixed open-loop arrival at 100 RPS (actual arrivals span approximately 0.19 s).
- Exact 2048-token prompt and requested output length 16.
- No warmup, seed 2025, telemetry interval 0.25 s.
- A 2048-token prompt requires `ceil(2048 / 16) = 128` blocks.
- The scheduler can admit 13 prompts: `13 * 128 = 1664`; the 14th would require `1792 > 1768` blocks.
- The first prefill batch uses `13 * 2048 = 26,624` prompt tokens, below `max_tokens_in_batch=49,152`.
- `13 < max_batch_size=32`, so `max_batch_size` is not the admission limit.
- After one generated token, each resident sequence needs 129 blocks and the resident set uses `13 * 129 = 1677` blocks. A new 2048-token request would require 128 more blocks, giving `1805 > 1768`.
- `129 / 3072` is far below `max_blocks_per_seq`; 13 resident IDs are far below `max_seqs_in_block_table=128`.

Therefore waiting at the pressure point is attributable to GPU KV block availability, not to batch size, prompt-token batch budget, per-sequence block limit, or sequence-table exhaustion.

## Exact reproduction command

Run from the repository root with an unused `--run-id`:

```bash
cd /nfs/home/s314511048/MorphServe
MODEL=/nfs/home/s314511048/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
VENV=/nfs/home/s314511048/.venv
PYTHONPATH="$PWD/swiftLLM:$PWD/swiftLLM/csrc" CUDA_VISIBLE_DEVICES=3 \
  "$VENV/bin/python" -u -m benchmark.run \
  --model-path "$MODEL" --target-rps 100 --arrival-mode fixed \
  --request-count 20 --prompt-token-count 2048 --output-token-count 16 \
  --seed 2025 --telemetry-interval-s 0.25 --output-dir benchmark-results \
  --run-id phase1-kv-calibration-long-prompt-c13-mem-repro \
  --expected-num-gpu-blocks 1768 \
  --max-batch-size 32 --max-tokens-in-batch 49152 --num-cpu-blocks 4096
```

The validated run IDs are:

- `benchmark-results/phase1-kv-calibration-long-prompt-c13-mem/`
- `benchmark-results/phase1-kv-calibration-long-prompt-c13-mem-repeat/`

Each contains `metadata.json`, `requests.jsonl`, `telemetry.jsonl`, and `summary.json`.

## Selected pilot evidence

### Primary run

| metric | value |
|---|---:|
| requested / observed prompt | 2048 / 2048 tokens |
| requested / observed output | 16 / 16 tokens |
| offered / actual arrival rate | 100.0 / 100.537 RPS |
| completed / unfinished | 20 / 0 |
| peak `running_q` | 13 |
| peak `waiting_q` | 19 overall; **7 while KV utilization was >=80%** |
| peak `swapped_q` | 0 |
| peak `num_decoding_gpu_blocks` | 1677 |
| `num_gpu_blocks` | 1768 throughout |
| peak logical KV utilization | **0.9485294118 (94.85%)** |
| TTFT mean / P95 | 18.9047 / 21.7864 s |
| queueing mean / P95 | 2.7251 / 6.9141 s |
| TPOT mean / P95 | 0.07074 / 0.07160 s |
| swap-in / swap-out / preemption | 0 / 0 / 0 |

There were 26 telemetry samples at utilization >=80%. In those samples:

- `running_q=13` for every sample;
- `waiting_q=7` for every sample;
- `num_decoding_gpu_blocks` ranged from 1664 to 1677;
- `num_gpu_blocks=1768` for every sample; and
- swap-in, swap-out, and preemption counters remained zero.

The `waiting_q=19` maximum occurs earlier while requests are still being tokenized/admitted. The pressure-region evidence is the conditional `waiting_q=7`, not that transient maximum.

### Repeat run

The same command/configuration was repeated as `phase1-kv-calibration-long-prompt-c13-mem-repeat`:

- 20/20 completed; no unfinished requests.
- Peak `running_q=13`, pressure-region `waiting_q=7`, `swapped_q=0`.
- Peak `num_decoding_gpu_blocks=1677`, `num_gpu_blocks=1768`, utilization **94.85%**.
- TTFT mean/P95 17.6316/20.3481 s.
- Queueing mean/P95 2.6514/6.6800 s.
- TPOT mean/P95 0.05291/0.05494 s.
- Swap-in, swap-out, and preemption all zero.

The repeated exact block/utilization/queue state demonstrates that the selected pilot is reproducible. The long-prompt prefill contributes materially to TTFT, so latency is classified as a compute component alongside the direct KV admission bottleneck; the pilot is not declared successful from latency alone.

## Outcome classification

1. **KV pressure:** selected pilot. Logical utilization reaches 94.85%, and a 14th 2048-token prefill cannot be admitted because `1677 + 128 > 1768` while only 13 of 32 batch slots are resident.
2. **Max-batch/configuration pressure:** not the selected pressure-region cause. `running_q=13 < max_batch_size=32`; `13 * 2048 < max_tokens_in_batch`; per-sequence and sequence-table limits are also slack.
3. **Compute pressure:** the prior short-output sweep and `c20-short` control show latency/throughput effects while logical KV remains low. They are not memory evidence.
4. **Mixed pressure:** the selected run has a compute cost from the 2048-token prefill and therefore high TTFT/TPOT, but the waiting queue coincides with the independently calculated GPU-block admission failure. The primary admission bottleneck is KV capacity.

No preemption or swap occurred in the selected pilot. That is expected: the workload demonstrates prefill admission blocking without requiring a decode overrun.

## Recommended next experiment

Use the selected configuration and fixed workload as the starting point for the final memory/KV saturation experiment:

```text
block_size=16, gpu_mem_utilization=0.99, num_gpu_blocks=1768 (profiled),
num_cpu_blocks=4096, max_seqs_in_block_table=128, max_blocks_per_seq=3072,
max_batch_size=32, max_tokens_in_batch=49152,
prompt=2048 tokens, requested output=16 tokens,
fixed arrival=100 RPS, request count=20, seed=2025, telemetry=0.25 s.
```

Hold this configuration fixed in that later experiment and vary only the intended offered-load parameter. Do not treat the selected 1768-block profile as the original 3880-block baseline; both capacities are recorded above.

## Validation and completion audit

| requirement | concrete evidence | status |
|---|---|---|
| Read required baseline, benchmark, saturation, audit, and reference documents | Files read before calibration: `baseline.md`, `archive/benchmark-harness-report.md`, `archive/short-workload-saturation-report.md`, `archive/completion-audit.md`, `../../references/MORPHSERVE_PHASE1_REFERENCE.md` | PASS |
| Preserve the prior short-workload result as negative KV evidence | Prior report value (~0.206%, no swaps/preemptions) retained verbatim in this report | PASS |
| Inspect/report all requested EngineConfig fields | Baseline and selected tables above include batch, token, block, CPU/GPU, sequence, and utilization fields | PASS |
| Use 3880-block/16-token baseline capacity explicitly | 62,080 slots and `62,080/C` planning table above | PASS |
| Try controlled candidate workloads rather than a final sweep | Short control, two documented baseline long-output attempts, and two selected fixed pilot runs | PASS |
| Capture prompt/output, arrivals, queues, blocks, utilization, latency, swaps, completion state | Raw JSONL and summary for `c20-short`, primary selected run, and repeat; tables above | PASS |
| Distinguish KV, max-batch/config, compute, and mixed outcomes | Outcome classification above | PASS |
| Achieve strong KV pressure | 94.85% logical utilization, above the 80% target | PASS |
| Show memory rather than max batch is limiting | Pressure-region `running_q=13<32`, 26,624<49,152 prefill tokens, and `1677+128>1768` | PASS |
| Do not declare success from TTFT/queueing alone | Direct block/admission and queue correlation is the stated basis | PASS |
| Do not perform final publication-style RPS sweep | No final sweep was launched in this goal | PASS |
| Keep `swiftLLM/swiftllm/server/scheduler.py` unchanged | `git diff --exit-code -- swiftLLM/swiftllm/server/scheduler.py` passed | PASS |
| Regenerate and validate artifacts | `benchmark.summarize` regenerated selected summaries byte-identically; raw files retained | PASS |
| Run code/test gates | Six benchmark unit tests passed; benchmark/engine/structs/scheduler `py_compile` passed; `git diff --check` passed | PASS |

This calibration therefore selects the fixed long-prompt configuration above for the later final KV saturation experiment, without claiming that the original short baseline was memory-saturated.
