# MorphServe Table 5 static quality–latency reproduction specification

Status: **protocol recorded before benchmarking** (2026-09-11 UTC)

## 1. Objective and scope

Reproduce the static quality–latency trade-off represented by MorphServe Table 5, using only the four static configurations:

| configuration ID | paper description | quantized decoder layers |
|---|---|---:|
| `fp16_0` | 0 quantized layers (FP16) | 0 |
| `w4_8` | 8 layers (Selective) | 8 |
| `w4_16` | 16 layers (Selective) | 16 |
| `w4_32` | 32 layers (Full quant.) | 32 |

The dynamic MorphServe condition, other paper figures/tables, scheduler changes, KV-cache resizing, and runtime swapping are out of scope. Table 5 numbers are reference targets, not pass/fail values.

Primary quality metric: answer F1. Primary latency metric: P95 time-to-first-token (TTFT). SLO criterion: TTFT > 2 seconds. The result must distinguish a directional trade-off from a numerical reproduction.

## 2. Evidence extracted from the supplied paper

Source: `references/MorphServe.pdf`, supplied MorphServe MLSys 2026 paper.

- Section 5, **Evaluation Setup**: the paper evaluates Llama 3 8B with DuReader and BurstGPT; for Llama 3 8B it sets prompts to 1024 tokens and responses to 512 tokens; it reports F1 and ROUGE-L. It reports 24-GB NVIDIA L4 hardware for the 7B/8B models.
- Section 5.2, **Runtime Swapping over Static Selective Quantization**: the Table 5 comparison uses the same Llama 3 8B model, DuReader dataset, and BurstGPT trace; it states that 16–32 swapped layers meet latency targets while 8 layers preserve quality better but miss the TTFT SLO under bursts.
- Table 5 (printed paper page 6):

  | setting | F1 (%) | TTFT P95 (s) | SLO violation |
  |---|---:|---:|---:|
  | 32 layers / full quant. | 23.66 | 1.62 | 0% |
  | 16 layers / selective | 23.93 | 1.74 | 0% |
  | 8 layers / selective | 24.16 | 2.61 | 4.2% |
  | 0 layers / FP16 | 25.19 | 5.65 | 12.7% |

- Section 4.2: the paper's Layer Importance Score (LIS) is `0.25 * LTS + 0.25 * LRS + 0.5 * MDS`; its greedy ordering is computed offline and reused at runtime. The supplied paper does not publish a concrete 32-layer ordering for the Table 5 run.
- Appendix B, **Serving Traces**: the paper extracts a representative 72-second BurstGPT segment and applies 1.75x downscaling to simulate saturation-level traffic. The paper does not identify the raw timestamp bounds of the Table 5 segment.
- Appendix B, **Evaluation Datasets**: the paper says DuReader is Chinese MRC and that it uses an English-translated version. The public `baidu/DuReader` repository exposes Chinese DuReader 2.0 data and a 100-example demo, but no English-translated Table 5 subset is present in the repository checkout used here.

## 3. Fixed protocol

Every condition must use the same:

- local checkpoint and tokenizer;
- prompt text, prompt token IDs, question/reference mapping, and request order;
- arrival offsets and trace scaling;
- generation parameters and requested output length;
- serving engine, scheduler, batch settings, and measurement code;
- GPU and software environment;
- quantization implementation for every W4 condition; and
- layer ordering (the only independent model change is the count of W4 decoder layers).

The planned protocol is:

| item | fixed value / rule |
|---|---|
| Model | local `meta-llama/Llama-3.1-8B` snapshot `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`; closest available local Llama-3 checkpoint |
| Tokenizer | tokenizer files from that same snapshot; tokenizer hash recorded in the run manifest |
| Serving engine | the repository's unchanged SwiftLLM engine plus observational benchmark runner; one fresh process per condition |
| Scheduler | upstream vendored `swiftLLM/swiftllm/server/scheduler.py`, byte/hash checked before and after; no semantic scheduler changes |
| GPU | one reserved local CUDA device, recorded with model, driver, CUDA, and free/total memory; planned device is `CUDA_VISIBLE_DEVICES=3` (RTX 3090, 24 GiB), not paper's L4 |
| Prompt | fixed content-rich DuReader prompt, tokenized to exactly 1024 tokens by the same tokenizer; prompt IDs and SHA-256 saved in workload manifest |
| Response | `max_new_tokens=512`, exactly the same stopping rule for all conditions; deterministic decoding (`do_sample=false`, temperature/top-p unset/disabled); EOS may end a request early and is recorded |
| Arrival | exact trace offsets from one saved BurstGPT segment, open-loop and independent of completion; the 1.75x scale is applied as `scaled_offset = (raw_timestamp - segment_start) * 1.75` |
| Dataset mapping | trace row `i` maps deterministically to DuReader demo test row `i`; dataset IDs, references, and prompt hashes saved; no content is changed between conditions |
| Seeds | Python/NumPy/PyTorch seed `2025`; deterministic decoding; seed recorded per run |
| Warm-up | no measured warm-up requests; model initialization and any cache/profile pass finish before measurement clock starts |
| Repetitions | at least one run per condition initially; repeat conditions where time/GPU capacity permits, especially `fp16_0` and `w4_32`; repeat results are not merged silently |
| TTFT | `(first_stream_token_received_time - actual_open_loop_arrival_time)` using monotonic nanosecond timestamps; P95 is the nearest-rank/explicit percentile method recorded by the aggregator |
| SLO | request is a violation iff TTFT is finite and strictly greater than `2.0` seconds; denominator is all launched requests, with failures/incomplete requests separately reported and never silently dropped |
| F1 | macro mean of per-request character-overlap F1 against the best available reference answer, using official DuReader-style whitespace removal/character tokenization; the exact evaluator implementation and per-request values are saved |

### Layer order

The exact LIS sequence needed to reproduce the paper is unavailable in the supplied artifact and recovering it would require implementing the full MorphServe profiling path. The controlled fallback is **front-to-back decoder order `[0, 1, ..., 31]`**: `w4_8` quantizes layers 0–7, `w4_16` layers 0–15, and `w4_32` layers 0–31. This order is selected once before measurement and is not tuned per condition. The manifest labels all results `lis_order_unavailable_front_to_back`.

### Quantization decision and proxy label

The paper identifies static quantization as AWQ INT4. AWQ support is attempted first using public AutoAWQ/vLLM capabilities. If selective per-layer AWQ cannot be loaded and executed reliably in the same engine, all three quantized conditions will use one consistent local W4 implementation (same group size, codebook, scale handling, and kernels) and will be labeled **uniform W4 proxy**, not AWQ reproduction. No condition may mix AWQ and a different W4 implementation. If no reliable W4 execution path exists, the experiment stops with the blocker and preserves the setup evidence rather than fabricating measurements.

## 4. Workload reconstruction rule

Public source checkout used for reconstruction: `HPMLL/BurstGPT`, revision to be recorded from `git rev-parse HEAD`, file `data/BurstGPT_1.csv`. The raw trace's `Timestamp` is seconds from the first trace day. Because the paper's Table 5 raw interval is not published, choose and record one deterministic interval before runs: the **first 72-second half-open window in `BurstGPT_1.csv` containing at least 100 rows**. This currently resolves to raw `[744286, 744358)` seconds with 106 rows; all rows in the interval are retained in the segment artifact, including rows whose trace response token count is zero, because the paper uses timestamps as workload arrivals and fixes Llama 3 prompt/response lengths. The scaled offsets span 0 to 124.25 seconds after multiplication by 1.75. This is an explicit interval substitution, not a claim that it is the paper's hidden interval.

The selected trace's `Model`, input/output token columns, and `Log Type` remain metadata only. Llama 3 requests use the fixed 1024/512 protocol above. The public trace CSV and its selected rows are hashed in the manifest.

The selected trace row `i` maps to the `i`-th answer-bearing DuReader demo record in the concatenated dev-then-train source; this filtered mapping is fixed before benchmarking and the selected records are copied into the input artifact.

Public dataset source: `baidu/DuReader`, revision to be recorded, the concatenation of official DuReader 2.0 `data/demo/devset/search.dev.json` and `data/demo/trainset/search.train.json`, filtered to its first 106 answer-bearing records. The paper's English-translated subset is not available locally; the benchmark therefore uses the Chinese official demo subset and labels this as a dataset-language/subset substitution. The exact selected records are copied into the workload artifact with source IDs and hashes, so the measured mapping remains auditable.

## 5. Required artifacts and gates

Before any model run, write one machine-readable `run_manifest.json` containing:

- protocol/spec revision and paper target values;
- checkpoint path, snapshot/revision, config/tokenizer hashes;
- GPU model/index/memory, CUDA/driver/PyTorch/Transformers/vLLM/bitsandbytes versions;
- serving/quantization library versions and git commits;
- BurstGPT repository URL/revision, file hash, raw interval, row count, scale and exact offsets;
- DuReader repository URL/revision, file/subset/hash, language substitution and request mapping;
- four configuration IDs, quantized layer count, fixed layer order, quantization label, and seeds;
- generation, warm-up, timing, F1, percentile, and SLO definitions; and
- exact command lines and environment variables.

For each condition, preserve:

- raw `metadata.json`, per-request `requests.jsonl`, and `telemetry.jsonl`;
- generated text, reference IDs, per-request F1 and SLO status;
- aggregate JSON derived from those raw rows;
- stdout/stderr log and failure information; and
- the configuration identity and manifest hash.

The final derived output must include:

1. a four-row table with configuration, quantized layers, F1, P95 TTFT, SLO violation rate, request counts, and uncertainty/repeat information;
2. a quality-vs-P95-TTFT plot generated from raw per-request results;
3. raw-data regeneration commands/scripts;
4. claim-by-claim assessment of the four requested claims; and
5. separate **CONFIRMED FINDINGS**, **SUPPORTED BUT UNCERTAIN FINDINGS**, **BLOCKED QUESTIONS**, and **REMAINING UNCERTAINTY** sections.

No conclusion may be based on manually transcribed Table 5 values or on a passing setup check alone. A monotonicity failure must be reported and investigated, not tuned away.
