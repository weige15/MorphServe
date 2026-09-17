# MorphServe reproduction evidence brief

## 1. Primary conference-final source attestation

The authoritative paper source for the current reproduction is the exact requested local file:

- `../references/morphserve-mlsys2026-conference-final.pdf`
- absolute path: `/nfs/home/s314511048/MorphServe/references/morphserve-mlsys2026-conference-final.pdf`
- SHA-256: `080ddcfe8c23e12bb421e4c1107c09246c82144345064018401b49e1a8dd2678`
- page count: **20**
- title: *MorphServe: Efficient and Workload-Aware LLM Serving via Runtime Quantized Layer Swapping and KV Cache Resizing*
- authors: Zhaoyuan Su, Zeyu Zhang, Tingfeng Lan, Zirui Wang, Haiying Shen, Juncheng Yang, Yue Cheng
- venue: Proceedings of the 9th MLSys Conference, Bellevue, WA, USA, 2026
- extracted text: `../sources/morphserve-mlsys2026-conference-final.txt`
- source audit: `../results/raw/conference-final-source-audit.json`
- version delta: `paper-version-delta.md`

The final contains Figures 1–7 and Tables 1–9. Table 9 is present on PDF page 19. MuPDF text extraction and visual inspection covered pages 8–11 and 17–20, including the main result figures/tables, §5.1, Appendix C, and final appendix tables. Full-document searches confirm Table 9, §5.1, Appendix C, LLM-PQ, PyramidKV, 41.3%, 82.3%, 1.73×, and 2.4×.

The conference-final paper reports, in §5.1, LLM-PQ accuracy-gap closure of **41.3% on average and up to 82.3%**, and accuracy-mode PyramidKV P95 TTFT improvement of **1.73× on average and up to 2.4×**. These are paper references only. They must not be mixed with modified-condition local measurements.

## 2. Historical arXiv-v2 source retained

The former primary target remains preserved as a historical secondary source:

- `../references/morphserve-2506.02006-v2.pdf`
- SHA-256: `e2c0f12fcbc5188a04078a31a732acaa95e93e9662aff4b766a0b9d6a73fbe30`
- 19 pages; Figures 1–7 and Tables 1–8 only; no Table 9
- extracted text: `../sources/morphserve-2506.02006-v2.txt`

Its audit, old table classifications, hashes, and prior measured evidence are not deleted or rewritten. See `paper-version-delta.md` for numbering, result, LLM-PQ/PyramidKV, and Appendix C differences.

## 3. Historical v2 page-by-page source-to-implementation requirements

The following detailed extraction is retained from the v2 audit and remains useful for mechanism comparison. Where its table numbers differ from the conference final, the final numbering in the primary source and `configs/paper-reference-values.json` takes precedence.

### PDF page 1 — abstract and problem contract

- Implement two asynchronous token-level adaptations:
  1. Selective replacement of less impactful decoder layers by quantized variants.
  2. Pressure-aware resizing of KV-cache capacity.  
  Transitions must preserve active inference state and avoid re-prefill/model flushing. (PDF p.1, Abstract; §1)
- Target dynamic and bursty traffic rather than a fixed workload/precision configuration. Relevant latency metrics are TTFT and time per output token, called TOPT on p.1 and TPOT elsewhere. (PDF p.1, §1)
- Reported headline targets are a 92.45% average reduction in SLO violations and 2.2–3.9× better P95 TTFT versus full precision. These are prose aggregates, not reconstructable from a result table. (PDF p.1, Abstract)
- Footnote: affiliations are University of Virginia and Harvard University; the artifact is labeled “Preprint.” This has no implementation consequence. (PDF p.1, footnote)

### PDF page 2 — top-level component behavior

- On load increase, reduce model memory by quantizing selected layers and attach more KVC blocks; on pressure decrease, restore full precision and reclaim temporary KVC memory. (PDF p.2, §1)
- Required properties:
  - Full-precision and quantized layers coexist.
  - Runtime reconfiguration must not change architecture or flush the model.
  - Controller exposes a tunable latency/accuracy policy.
  - LayerSwapper and KVResizer operate asynchronously, with KV resizing parallel to decode on a separate CUDA stream. (PDF p.2, §1)
- Compatibility claims include GPTQ/AWQ-style weight quantization and modern continuous-batching/PagedAttention serving. (PDF p.2, §2)
- Additional prose targets: up to 88.85% lower F1/ROUGE-L degradation than static AWQ and 29.29% higher memory utilization. No underlying per-case values for the 88.85% result are supplied. (PDF p.2, §1)

### PDF page 3 — workload and SLO semantics

- Use dynamic Azure LLM and BurstGPT arrival/token traces rather than a stationary synthetic arrival process. (PDF p.3, Fig.1a; §3)
- Define the TTFT SLO threshold as **2 seconds**. Define saturation as the point at which GPU memory cannot admit another prefill or continue the current decode batch. (PDF p.3, §3)
- Figure 1c’s accuracy test uses an AWQ INT4 model, F1, and GovReport from LongBench; the model and hardware are not identified on this page. (PDF p.3, §3)
- Implement dynamic mixed precision as a continuum rather than a binary static W4/FP16 choice. Figure 1d distinguishes `W4`, `MorphServe-slo-pref`, `MorphServe`, `MorphServe-acc-pref`, and `FP16`. (PDF p.3, Fig.1d)
- System objectives are dynamic adaptation, no degradation at light/moderate load, limited token-level degradation beyond saturation, and low overlap-related overhead. (PDF p.3, §4)

### PDF page 4 — architecture, policy inputs, and objective

- Required control path:
  1. Requests enter the Serving Monitor.
  2. Workers return telemetry.
  3. Monitor sends work to Request Dispatcher.
  4. Dispatcher routes requests.
  5. Metrics go to Morphing Controller.
  5′. Controller sends adaptation commands to worker executors.
  6. Responses return to users. (PDF p.4, Fig.2)
- Serving Monitor metrics: GPU memory utilization, queue depth, throughput, TTFT, and TPOT, smoothed over unspecified “short time windows.” (PDF p.4, §4.1)
- Morphing Controller is the global GPU memory manager. The only explicit trigger examples are **KVC usage >85%** and **queue delay >100 ms**; these are examples, not declared experimental policy values. (PDF p.4, §4.1)
- Each worker contains a model replica, cache engine, and Morphing Executor. The executor may change FP16→INT8→INT4 and resize block count. Buffers are preallocated and communication/computation overlap. (PDF p.4, Fig.2; §4.1)
- Token-level behavior may switch, for example, two layers to INT4 partway through a request and restore them later without changing retained attention state. (PDF p.4, §4.1)
- Optimization objective:

\[
\min_{\{n_k\}}\sum_{t=t_1}^{t_n}
\Delta\left(f(x_t), f^{(Q_t)}(x_t)\right)
\tag{1}
\]

  `n_k` and the degradation function \(\Delta\) are not defined. (PDF p.4, Eq.1; §4.2)

### PDF page 5 — LIS equations and model staging

For candidate layer \(p\), input \(x_p\), original layer output \(h_p(x)\), quantized output \(h_p^Q(x)\), and already-quantized set \(Q\):

\[
\mathrm{LTS}_p=\cos(h_p(x),x_p)
\tag{2}
\]

\[
\mathrm{LRS}_p=\cos(h_p(x),h_p^Q(x))
\tag{3}
\]

\[
\mathrm{MDS}^{(Q)}_p=
\cos\left(f^{(Q)}(x),f^{(Q\cup\{p\})}(x)\right)
\tag{4}
\]

\[
\mathrm{LIS}_p=
\alpha_1\mathrm{LTS}_p+
\alpha_2\mathrm{LRS}_p+
\beta\mathrm{MDS}^{(Q)}_p
\tag{5}
\]

(PDF p.5, §4.2)

- Use \(\alpha_1=\alpha_2=0.25,\ \beta=0.5\). High cosine similarity means the candidate is safer to swap; Algorithm 1 therefore takes the maximum LIS. (PDF pp.5, 15–16, Eq.5/6; Algorithm 1)
- Cosine similarity is selected over L2 to reduce scale sensitivity. (PDF p.5, §4.2)
- Preload decoder variants FP16, INT8, INT4, and INT3 into contiguous pinned CPU memory; place the initial FP16 model in contiguous preallocated GPU memory; record all addresses; precompile kernels and use fusion. (PDF p.5, §4.3)
- Footnote: “INT#” means weight-only quantization, W#. (PDF p.5, footnote 1)

### PDF page 6 — layer copy and KVC resizing mechanics

- Execute in-place `cudaMemcpyAsync`-style replacement on a separate CUDA stream while preceding layers continue. The illustrated example swaps layers 28 and 29 while layers 0–27 run. Quantized weights use the same base addresses to avoid pointer remapping. (PDF p.6, Fig.3; §4.3)
- Llama 2 7B examples:
  - FP16 decoder layer: 0.4 GB.
  - INT4 decoder layer: 0.1 GB.
  - PCIe Gen4 bandwidth: 26–28 GB/s.
  - INT4 transfer: about 4 ms.
  - FP16 transfer: about 16 ms.
  - Complete INT4 transfer plus reconstruction: about 6 ms.  
  These are prose measurements, not a table. (PDF p.6, §4.3)
- FP16→W4 is claimed to free up to 75% of layer memory. (PDF p.6, Fig.3; §4.4)
- KVResizer explicitly **does not quantize or compress existing KV entries**. It reallocates memory freed by weight quantization to new PagedAttention blocks. (PDF p.6, §4.4)
- Required PagedAttention extension: block-level on-demand allocation/deallocation and remapping without kernel recompilation, with resize operations on a separate CUDA stream. (PDF p.6, §4.4)

### PDF page 7 — queue/preemption behavior and evaluation start

- Attach blocks when queue length/wait exceeds a threshold to admit prefills; attach during decoding to avoid preemption, host swapping, or recomputation. Restore model precision and release temporary blocks after pressure subsides. (PDF p.7, §4.4)
- Figure 4 establishes the result matrix:
  - Columns: Vicuna 7B v1.5, Llama 2 7B, Llama 3 8B, CodeLlama 34B.
  - Azure rows: GovReport F1 and QMSum ROUGE-L.
  - BurstGPT rows: DuReader F1 and Multi-News ROUGE-L.
  - X-axis: P95 TTFT seconds.
  - Methods: static INT4, MorphServe performance/default/accuracy modes, and FP16. (PDF p.7, Fig.4)
- Evaluate representative **72-second** trace snippets. Downscale BurstGPT by **1.75×** and Azure by **4.75×**. The precise snippet offsets and the method of downscaling are not given. (PDF p.7, §5)

### PDF page 8 — principal test configuration

- Prompt/output lengths:
  - Vicuna 7B v1.5 and Llama 2 7B: 512/256 tokens.
  - Llama 3 8B and CodeLlama 34B: 1024/512 tokens. (PDF p.8, §5)
- Hardware:
  - Vicuna/Llama 2/Llama 3: one NVIDIA L4, 24 GB HBM, 256 GB CPU DRAM.
  - CodeLlama 34B: one A100, 80 GB HBM, 2 TB CPU DRAM. (PDF p.8, §5)
- Use the same SwiftLLM-based engine for all baselines; approximately 2,200 Python and 500 C++/CUDA lines were added. (PDF p.8, §5)
- Baselines are FP16 and static INT4 AWQ. Accuracy mode swaps conservatively; performance mode swaps aggressively. Exact trigger values for either mode are absent. (PDF p.8, §5)
- Figure 5 uses BurstGPT/DuReader for all four models over 0–72 seconds. Full-precision KVC capacity lines are 958, 958, 2599, and 3435 blocks for Vicuna, Llama 2, Llama 3, and CodeLlama respectively. (PDF p.8, Fig.5)

### PDF page 9 — throughput, TPOT, and aggregate improvements

- Figure 6 uses DuReader with increasing RPS; compare W4, default MorphServe, and FP16. Saturation is observed via sharply increasing TTFT. (PDF p.9, Fig.6; §5.1)
- Figure 7 shows Azure/GovReport TPOT CDFs for all four models and all three MorphServe modes plus baselines. (PDF p.9, Fig.7)
- Reported aggregate results:
  - Default MorphServe P95 TTFT: 2.9–15.7× improvement.
  - Accuracy mode: 2.2–3.9×.
  - Performance mode: 3.4–19.5×.
  - MorphServe quality degradation: 0.51–3.82%; accuracy mode 0.11–2.18%.
  - Static INT4 degradation: 2.34–9.47%.
  - KVC utilization improvement over static W4: 29.29%.
  - Output accuracy improvement over static W4: 3.58%.
  - Peak KVC expansion beyond FP16 capacity: 32.97%.
  - Queue-delay reduction: up to 3.8×.
  - Throughput: 1.6–1.83× FP16. (PDF pp.8–9, §5.1)
- These are prose aggregates without per-configuration raw data.

### PDF page 10 — ablations and deployment assumptions

- Figure 7 prose claims P95/P99 TPOT improvements of up to 1.06×/1.23×, performance-mode average TPOT improvement of 1.11–1.17×, and accuracy-mode overhead up to 1.06×. (PDF p.10, §5.1)
- Ablate AWQ against uniform INT4 while preserving the same engine, dataset, trace, and runtime logic. (PDF p.10, Table 2; §5.2)
- Validate LIS cross-dataset transfer by calibrating on WikiText-2 and evaluating C4 without re-profiling. (PDF p.10, Table 3; §5.2)
- Distributed design assumption: one independent Morphing Executor per GPU, compatible in prose with data/tensor/pipeline/sequence parallelism. No synchronization protocol for cross-GPU swaps is supplied. (PDF p.10, §6)
- Fairness relies on external continuous batching FIFO, chunked prefill, and fine-grained KV management. (PDF p.10, §6)

### PDF pages 11–14 — references

- These pages contain dependency provenance, not additional system parameters. They identify the intended conceptual dependencies: Sarathi chunked prefill, vLLM PagedAttention, Orca continuous batching, FlashAttention, AWQ/GPTQ, Azure/BurstGPT traces, LongBench-style datasets, SwiftLLM, and CUDA streams. (PDF pp.11–14, References)
- No package commits, framework versions, CUDA version, compiler flags, driver version, or model checkpoint hashes are given. A reproduction must record these independently. (PDF pp.11–14, References)
- Page 13 includes the GPUDirect Storage reference later proposed only as future work; GDS is not part of the implemented mechanism. (PDF pp.13, 19)

### PDF page 15 — appendix profiling semantics

- Profiling is optional and offline. It combines layer-local and model-output similarity in the duplicate LIS definition, Eq.6, identical to Eq.5. (PDF p.15, Appendix A.1)

\[
\mathrm{LIS}_p=
\alpha_1\mathrm{LTS}_p+
\alpha_2\mathrm{LRS}_p+
\beta\mathrm{MDS}^{(Q)}_p
\tag{6}
\]

- Cosine similarity is:

\[
\cos(a,b)=\frac{a\cdot b}{\lVert a\rVert\lVert b\rVert}
\tag{7}
\]

  Higher similarity is interpreted as lower risk from swapping. (PDF p.15, Eq.7; Appendix A.1)
- Calibration setting: a small WikiText-2 subset, sequence length **2,048**. Sample count, exact examples, tokenizer, and seed are absent. (PDF p.15, Appendix A.2)

### PDF page 16 — exact profiler and profiling costs

- Tables 4 and 5 define the selected LIS weights and cosine metric numerically; see §4 below. (PDF p.16, Tables 4–5)
- Execute Algorithm 1 exactly as transcribed in §3 below. (PDF p.16, Algorithm 1)
- If profiling is unavailable, use front-to-back ordering as the default. (PDF p.16, Appendix A.3)
- Claimed profiling cost: under 15 minutes on one unspecified GPU for a 32-layer model; run once and reuse. (PDF p.16, Appendix A.3)

### PDF page 17 — full layer-order evaluation and verbatim output

- Use Table 6 to validate the ordering at 0, 1, 2, 4, 8, 16, and fully quantized layer counts. Random is averaged over an unspecified number of runs. (PDF p.17, Table 6; Appendix A.3)
- Mixed-output demonstration uses Llama 3 8B Instruct and 128 generated tokens from the quoted zebra prompt. The displayed responses are truncated, so they are not sufficient as golden outputs. (PDF p.17, Appendix B.2)

### PDF page 18 — implementation and trace details

- Initialization:
  - Load FP16, W8, and W4 transformer weights into pinned CPU memory.
  - Precompile GEMM kernels with dummy inputs.
  - Preallocate each layer’s GPU region.
  - Use asynchronous in-place copies.
  - Extend PagedAttention with dynamic memory registration/remapping.
  - Use separate morph/decode CUDA streams. (PDF p.18, Appendix C)
- This appendix omits W3 even though p.5 lists INT3 as a possible preloaded variant. (PDF pp.5, 18)
- Azure sample: 72 seconds, downscale 4.75×. BurstGPT sample: 72 seconds, downscale 1.75×. Both source URLs are supplied, but offsets are not. (PDF p.18, Appendix C)
- All models are said to use pre-quantized AWQ INT4 weights; this statement does not specify checkpoint IDs, group size, zero-point configuration, calibration set, or Uniform INT4 implementation. (PDF p.18, Appendix C)

### PDF page 19 — dataset construction and limitations

- Pair arrival timestamps from the traces with sampled benchmark contexts because the trace and content datasets are separate. Sampling policy and seed are unspecified. (PDF p.19, Appendix C)
- GovReport is used for long-form summarization; QMSum for query-based meeting summarization; English-translated DuReader for factual QA; Multi-News for multi-document summarization. (PDF pp.18–19, Appendix C)
- Host-memory overhead is “typically under 2× model size” because both full and quantized weights are stored. (PDF p.19, Appendix D)
- Current granularity is a whole transformer layer; independent attention/MLP morphing and workload prediction are future work. (PDF p.19, Appendix D)

## 3. Exact Algorithm 1

The following preserves the PDF’s expressions and selection direction. The typeset PDF joins `end for` and `return` on the last line; semantically, return is an unnumbered operation after line 14. (PDF p.16, Algorithm 1)

```text
Algorithm 1 Swapping Sequence Profiling Based on Layer
Importance Scoring (LIS)

Require: Full-precision model M, quantized model M^Q,
         calibration dataset D, params (α1, α2, β)

1:  for each layer i do
2:      Compute LTS_i = CosSim(Input_i, Output_i)
3:      Compute LRS_i = CosSim(Output_i, Output_i^Q)
4:  end for
5:  Initialize set of quantized layers Q ← ∅
6:  for t = 1 to L do
7:      for each unquantized layer j ∉ Q do
8:          Temporarily quantize layer j and evaluate
            model outputs
9:          Compute MDS_j^(Q) =
            CosSim(f^(Q)(x), f^(Q∪{j})(x))
10:         Compute LIS_j = α1 · LTS_j + α2 · LRS_j
            + β · MDS_j^(Q)
11:     end for
12:     Select j* = arg max_j LIS_j
13:     Add j* to Q and replace corresponding layer in M
14: end for
    return Ordered layer swap sequence Q
```

Implementation-critical observations:

- `L` is implicitly the number of decoder layers but is absent from `Require`. (PDF p.16, Algorithm 1)
- `D` is supplied but individual calibration examples or aggregation over `D` are not shown in the equations. (PDF p.16, Algorithm 1)
- `M^Q` is supplied, yet line 8 says “temporarily quantize” rather than explicitly copying layer \(j\) from \(M^Q\). (PDF p.16, Algorithm 1)
- Because all three terms are cosine similarities, selecting `arg max` ranks the least disruptive candidate first. (PDF pp.15–16, Eq.7; Algorithm 1)

## 4. Tables 1–9: exact values and settings

### Table 1 — Llama 3 8B, BookSum Chapters, 6K input/2K output

(PDF p.8, Table 1)

| Generation strategy | F1 | ROUGE-L | Within accuracy bound |
|---|---:|---:|---|
| All tokens INT4 (W4) | 14.4716 | 12.1598 | Lower Bound |
| First 1K FP16, last 1K W4 | 16.6112 | 15.2346 | Yes |
| First 1K W4, last 1K FP16 | 14.7101 | 12.3213 | Yes |
| First 512 FP16, middle 1K W4, last 512 FP16 | 16.0802 | 14.4013 | Yes |
| First 512 W4, middle 1K FP16, last 512 W4 | 16.0413 | 13.1127 | Yes |
| Switch every 256 tokens, starting W4 | 15.2634 | 13.0353 | Yes |
| Switch every 256 tokens, starting FP16 | 16.0012 | 14.0146 | Yes |
| All tokens FP16 | 17.6458 | 15.4847 | Upper Bound |

### Table 2 — Llama 2 7B, DuReader, BurstGPT

(PDF p.10, Table 2)

| INT4 method | Metric | Static quantization | MorphServe | FP16 |
|---|---|---:|---:|---:|
| AWQ | TTFT P95 (s) | 1.1686 | 1.2420 | 5.5223 |
| AWQ | TPOT P99 (s) | 0.0735 | 0.1064 | 0.1241 |
| AWQ | F1 | 25.55 | 27.33 | 27.68 |
| Uniform | TTFT P95 (s) | 1.1321 | 1.2144 | 5.5223 |
| Uniform | TPOT P99 (s) | 0.0728 | 0.1053 | 0.1241 |
| Uniform | F1 | 24.87 | 26.94 | 27.68 |

Caption claims degradation reductions of 83.57% for AWQ and 74.67% for Uniform and >77% lower P95 TTFT than FP16. The AWQ percentage agrees with rounded cells; the Uniform percentage does not, as discussed below. (PDF p.10, Table 2 caption)

### Table 3 — C4 evaluation, LIS calibrated on WikiText-2

(PDF p.10, Table 3)

| Method | 1 | 2 | 4 | 8 | 16 | 32/full W4 |
|---|---:|---:|---:|---:|---:|---:|
| Front-to-Back | 7.0793 | 7.0822 | 7.0933 | 7.1119 | 7.1524 | 7.2463 |
| Back-to-Front | 7.0871 | 7.0938 | 7.1031 | 7.1224 | 7.1712 | 7.2463 |
| Random | 7.0871 | 7.0913 | 7.1012 | 7.1234 | 7.1676 | 7.2463 |
| LIS | 7.0801 | 7.0818 | 7.0912 | 7.1107 | 7.1497 | 7.2463 |

Values are perplexity; columns are number of quantized layers. (PDF p.10, Table 3)

### Table 4 — LIS weight ablation, Llama 2 7B/WikiText-2

(PDF p.16, Table 4)

| Weights \((\alpha_1,\alpha_2,\beta)\) | 1 | 2 | 4 | 8 | 16 | 32/full W4 |
|---|---:|---:|---:|---:|---:|---:|
| (0.33, 0.33, 0.33) | 5.4732 | 5.4748 | 5.4786 | 5.4905 | 5.5257 | 5.6002 |
| (0.25, 0.25, 0.5) | 5.4732 | 5.4743 | 5.4779 | 5.4875 | 5.5215 | 5.6002 |

### Table 5 — L2 versus cosine LIS, Llama 2 7B/WikiText-2

(PDF p.16, Table 5)

| Metric | 1 | 2 | 4 | 8 | 16 | 32/full W4 |
|---|---:|---:|---:|---:|---:|---:|
| LIS — L2 norm | 5.4733 | 5.4776 | 5.4848 | 5.5021 | 5.5321 | 5.6002 |
| LIS — cosine similarity | 5.4732 | 5.4743 | 5.4779 | 5.4875 | 5.5215 | 5.6002 |

### Table 6 — layer-order perplexity on WikiText-2

The FP16 baseline and fully quantized endpoint are merged cells shared by all strategies for a model. CodeLlama’s endpoint is 48 INT4 layers despite the printed header saying `32 (INT4)`. (PDF p.17, Table 6)

| Model | Method | 0/FP16 | 1 | 2 | 4 | 8 | 16 | Full INT4 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Vicuna 7B | Front-to-Back | 6.78 | 6.79 | 6.78 | 6.79 | 6.80 | 6.84 | 6.98 (32) |
| Vicuna 7B | Back-to-Front | 6.78 | 6.82 | 6.83 | 6.84 | 6.86 | 6.91 | 6.98 (32) |
| Vicuna 7B | Random | 6.78 | 6.78 | 6.79 | 6.80 | 6.82 | 6.87 | 6.98 (32) |
| Vicuna 7B | LIS | 6.78 | 6.78 | 6.78 | 6.79 | 6.79 | 6.84 | 6.98 (32) |
| Llama 2 7B | Front-to-Back | 5.47 | 5.47 | 5.47 | 5.48 | 5.50 | 5.54 | 5.60 (32) |
| Llama 2 7B | Back-to-Front | 5.47 | 5.48 | 5.48 | 5.49 | 5.50 | 5.53 | 5.60 (32) |
| Llama 2 7B | Random | 5.47 | 5.47 | 5.48 | 5.48 | 5.50 | 5.53 | 5.60 (32) |
| Llama 2 7B | LIS | 5.47 | 5.47 | 5.47 | 5.48 | 5.49 | 5.52 | 5.60 (32) |
| Llama 3 8B | Front-to-Back | 6.14 | 6.15 | 6.16 | 6.19 | 6.23 | 6.34 | 6.53 (32) |
| Llama 3 8B | Back-to-Front | 6.14 | 6.17 | 6.18 | 6.20 | 6.24 | 6.33 | 6.53 (32) |
| Llama 3 8B | Random | 6.14 | 6.15 | 6.16 | 6.18 | 6.24 | 6.34 | 6.53 (32) |
| Llama 3 8B | LIS | 6.14 | 6.15 | 6.15 | 6.18 | 6.22 | 6.32 | 6.53 (32) |
| CodeLlama 34B | Front-to-Back | 5.47 | 5.47 | 5.47 | 5.48 | 5.48 | 5.49 | 5.53 (48) |
| CodeLlama 34B | Back-to-Front | 5.47 | 5.47 | 5.48 | 5.48 | 5.48 | 5.49 | 5.53 (48) |
| CodeLlama 34B | Random | 5.47 | 5.47 | 5.47 | 5.48 | 5.48 | 5.49 | 5.53 (48) |
| CodeLlama 34B | LIS | 5.47 | 5.47 | 5.47 | 5.47 | 5.48 | 5.49 | 5.53 (48) |

### Table 7 — independence of layer quantization effects

Llama 2 7B, WikiText dataset; progressively quantize the first \(N\) layers, then additionally quantize layer 19 or 24. (PDF pp.17–18, Appendix B.1; Table 7)

| Quantized prefix | None | First 1 | First 2 | First 4 | First 8 | First 16 |
|---|---:|---:|---:|---:|---:|---:|
| PPL | 5.472089 | 5.473293 | 5.474974 | 5.484205 | 5.503622 | 5.539723 |
| PPL + layer 19 | 5.475612 | 5.476802 | 5.478512 | 5.487739 | 5.507146 | 5.543575 |
| Effect, layer 19 | 0.003523 | 0.003509 | 0.003538 | 0.003533 | 0.003524 | 0.003852 |
| PPL + layer 24 | 5.473969 | 5.475141 | 5.476794 | 5.486121 | 5.505570 | 5.541744 |
| Effect, layer 24 | 0.001880 | 0.001848 | 0.001820 | 0.001916 | 0.001948 | 0.002021 |

### Table 8 — Vicuna 7B v1.5, QMSum, Azure trace

(PDF p.18, Table 8)

| INT4 method | Metric | Static quantization | MorphServe | FP16 |
|---|---|---:|---:|---:|
| AWQ | TTFT P95 (s) | 0.4086 | 0.7281 | 6.2376 |
| AWQ | TPOT P99 (s) | 0.0846 | 0.1123 | 0.1384 |
| AWQ | F1 | 11.77 | 12.85 | 13.01 |
| Uniform | TTFT P95 (s) | 0.3958 | 0.7034 | 6.2376 |
| Uniform | TPOT P99 (s) | 0.0820 | 0.1095 | 0.1384 |
| Uniform | F1 | 11.32 | 12.57 | 13.01 |

### Table 9

**Absent.** The 19-page v2 PDF ends after Appendix D on p.19. It contains no Table 9 caption, values, or reference. Any reproduction request expecting Table 9 requires another paper revision or supplementary artifact.

## 5. Figures 1–7: claims and settings

### Figure 1 — motivation

- (a) Azure and BurstGPT request/token volume are bursty. Azure is shown over roughly 60 minutes and BurstGPT over 720 minutes. (PDF p.3, Fig.1a)
- (b) 72-second snippets compare full-precision and MorphServe TTFT against the 2-second SLO; FP16 spikes under load. (PDF p.3, Fig.1b; §3)
- (c) AWQ W4 exhibits continuous GovReport F1 degradation while MorphServe limits degradation to pressured periods. (PDF p.3, Fig.1c; §3)
- (d) Pareto plot axes are SLO violation rate and overall accuracy drop. It compares W4, default MorphServe, SLO-preferred, accuracy-preferred, and FP16 for both traces. Exact point values are not published. (PDF p.3, Fig.1d)

### Figure 2 — control workflow

- CPU DRAM holds pre-quantized attention layers and an offline sequence such as illustrative `[27,18,...,2]`.
- Monitor, dispatcher, global controller, per-worker executor/cache engine/model replica form the feedback loop.
- Green output regions indicate only a small portion of tokens use mixed precision.  
  (PDF p.4, Fig.2 and caption)

### Figure 3 — model/KVC memory synergy

- Demonstrates full-precision → layers 28/29 swapped → additional KVC blocks attached → mixed-precision serving.
- Decoder submodules shown include `self_attn.q_proj` and `mlp.gate_proj`.
- The diagram labels KVC blocks 0, 1, …, 960 and newly attached blocks 961/962; these are illustrative rather than declared general block counts.
- Caption claims lower preemption for decoding and lower prefill queue time. (PDF p.6, Fig.3 and caption)

### Figure 4 — latency/accuracy frontier

- Four models × four dataset/trace rows, settings listed under p.7 above.
- Accuracy mode is claimed to reduce P95 TTFT by 2.2–3.9× versus FP16 with comparable generation quality.
- Performance mode is claimed to beat W4 output quality without additional latency overhead.
- No underlying exact point table is supplied. (PDF p.7, Fig.4 and caption)

### Figure 5 — dynamic KVC capacity

- BurstGPT/DuReader, 72 seconds, four models.
- FP16 capacity boundaries: 958, 958, 2599, 3435 blocks.
- Green MorphServe usage rises above FP16 capacity under peak load; orange W4 uses fewer blocks despite memory headroom; blue FP16 saturates at the red boundary.
- Claimed effects are fewer preemptions/KVC swaps and dynamic release after bursts. (PDF p.8, Fig.5 and caption; p.9, §5.1)

### Figure 6 — throughput saturation

- DuReader, four models, increasing RPS, TTFT shown up to 3 seconds.
- W4, default MorphServe, and FP16 are compared.
- Caption reports up to 1.83× FP16 throughput; text reports 1.6–1.83×. (PDF p.9, Fig.6 and caption; §5.1)

### Figure 7 — TPOT distributions

- Visible setting is Azure LLM/GovReport across all four models; x-axis TPOT, y-axis CDF.
- Compares W4, performance/default/accuracy MorphServe, and FP16.
- Caption: default MorphServe has comparable average TPOT, up to 1.23× lower P99, and performance mode up to 1.17× better average TPOT. (PDF p.9, Fig.7 and caption)
- The following prose says the figure covers “two datasets,” but the rendered figure contains only one GovReport row. (PDF pp.9–10, §5.1)

## 6. Consolidated experimental conditions

| Dimension | Required setting |
|---|---|
| Engine | SwiftLLM-derived common engine for all methods; PagedAttention and FlashAttention compatibility (PDF pp.8, 18, §5; Appendix C) |
| Models | Vicuna 7B v1.5, Llama 2 7B, Llama 3 8B, CodeLlama 34B (PDF p.7, §5) |
| Traces | Azure LLM Inference Dataset 2023 and BurstGPT (PDF pp.7, 18, §5; Appendix C) |
| Trace window | 72 seconds for each; exact offsets absent (PDF pp.7, 18) |
| Trace scaling | Azure 4.75× downscale; BurstGPT 1.75× downscale (PDF pp.7, 18) |
| Datasets | GovReport, QMSum, DuReader English translation, Multi-News (PDF pp.7, 18–19) |
| Quality metrics | F1 and ROUGE-L; perplexity for profiling ablations (PDF pp.8, 10, 16–18) |
| Context lengths | 512/256 for 7B MHA setting; 1024/512 for GQA setting (PDF pp.8, 18) |
| Main hardware | L4 24 GB + 256 GB host for 7B/8B; A100 80 GB + 2 TB host for CodeLlama 34B (PDF pp.8, 18) |
| Quantization | AWQ INT4 primary; Uniform INT4 ablation; W8 support; W3 mentioned only in main design (PDF pp.5, 8, 10, 18) |
| Baselines | Static FP16 and static INT4 in same engine (PDF p.8, §5) |
| Runtime modes | Accuracy, default, performance; thresholds absent (PDF pp.7–10) |
| SLO | TTFT 2 seconds; no exact TPOT SLO (PDF p.3, §3) |
| LIS calibration | WikiText-2 subset, sequence length 2,048; weights 0.25/0.25/0.5 (PDF pp.15–16) |
| BookSum test | Llama 3 8B, 6K prompt, 2K output (PDF p.8, Table 1) |
| Verbatim test | Llama 3 8B Instruct, 128 output tokens, zebra prompt (PDF p.17, Appendix B.2) |

## 7. Claim register

### A. Directly specified or tabulated numeric references

These may be copied exactly into a reproduction specification:

- Every cell in Tables 1–8, transcribed above. (PDF pp.8, 10, 16–18)
- SLO threshold: TTFT 2 s. (PDF p.3, §3)
- Example controller thresholds: KVC >85%, queue wait >100 ms; examples only. (PDF p.4, §4.1)
- LIS: 0.25/0.25/0.5; calibration length 2,048. (PDF pp.5, 15–16)
- Prompt/output lengths: 512/256 and 1024/512. (PDF pp.8, 18)
- Hardware capacities: 24 GB/256 GB and 80 GB/2 TB. (PDF pp.8, 18)
- Trace length/downscaling: 72 s, 1.75× and 4.75×. (PDF pp.7, 18)
- Figure 5 FP16 capacities: 958, 958, 2599, 3435 KVC blocks. (PDF p.8, Fig.5)
- Layer-size/transfer examples: 0.4/0.1 GB, 16/4 ms, complete W4 6 ms, 26–28 GB/s. (PDF p.6, §4.3)
- Code size: approximately 2,200 Python and 500 C++/CUDA lines. (PDF pp.8, 17–18)
- Profiling: under 15 minutes for a 32-layer model on one unspecified GPU. (PDF p.16, Appendix A.3)

### B. Prose-only aggregate claims

These are not backed by sufficient per-run raw data in the PDF:

- Average SLO violations reduced 92.45%. (PDF pp.1–2)
- P95 TTFT improvements: default 2.9–15.7×, accuracy 2.2–3.9×, performance 3.4–19.5×. (PDF pp.7–8)
- MorphServe quality degradation 0.51–3.82%; accuracy mode 0.11–2.18%; static W4 2.34–9.47%. (PDF p.8)
- F1/ROUGE-L degradation reduced up to 88.85%. (PDF p.2)
- KVC utilization +29.29%, output accuracy +3.58%, KVC expansion +32.97%, queue delay up to 3.8×. (PDF p.9)
- Throughput 1.6–1.83×. (PDF p.9)
- P95/P99 TPOT up to 1.06×/1.23× better; performance average TPOT 1.11–1.17× better; accuracy mode overhead up to 1.06×. (PDF pp.9–10)
- Host storage typically under 2× model size. (PDF p.19)

### C. Qualitative claims requiring independent validation

- “No” or “negligible” TPOT overhead from overlapping layer copies. (PDF pp.6, 9)
- State-preserving mid-request switching without re-prefill. (PDF p.4)
- Compatibility with all modern scheduling, GQA, MLA, and distributed parallelism. (PDF pp.1, 10)
- Generalization without re-profiling and user-tolerable perceptual quality. (PDF p.10; Appendix A)

## 8. Unresolved inconsistencies and reproduction risks

1. **Blocker — online policy is not reproducible.** The paper does not define smoothing windows, exact accuracy/default/performance thresholds, hysteresis, number of layers swapped per action, restoration timing, control interval, or tie-breaking. The 85%/100 ms values are only examples. (`references/morphserve-2506.02006-v2.pdf`, pp.4, 7–10)

2. **Blocker — workload reconstruction is under-specified.** Exact 72-second offsets, context-sampling sequence, random seeds, and downscaling algorithm are absent. (`references/morphserve-2506.02006-v2.pdf`, pp.7, 18–19)

3. **High — dynamic memory ownership is incomplete.** The paper simultaneously says each layer’s FP16-sized GPU region is preallocated and reused in place, and that FP16→W4 “frees 75%” for KVC. It does not explain how the unused tail of a layer allocation is deregistered, exposed to the KVC allocator, synchronized, and later reclaimed safely. (`references/morphserve-2506.02006-v2.pdf`, pp.5–6, 18)

4. **High — stale table cross-references.**
   - Independence results are referred to as Appendix Table 4 but are Table 7.
   - LIS weight ablation is called Appendix Table 1 but is Table 4.
   - L2/cosine is called Appendix Table 2 but is Table 5.
   - Quantization-method appendix results are called Table 5 but are Table 8.  
   (`references/morphserve-2506.02006-v2.pdf`, pp.5, 10, 16, 18)

5. **High — LIS semantics conflict.** Equations define \(\alpha_1\) on LTS, an input/output cosine, while prose calls \(\alpha_1\) “weight sensitivity” or norm change after quantization. The appendix also calls LTS/LRS input-independent despite computing them from calibration activations. (`references/morphserve-2506.02006-v2.pdf`, pp.5, 15–16)

6. **Medium — Table 2 percentage mismatch.** Uniform INT4 degradation is \(27.68-24.87=2.81\); MorphServe degradation is \(27.68-26.94=0.74\). Reduction is about **73.67%**, not captioned **74.67%**. (`references/morphserve-2506.02006-v2.pdf`, p.10, Table 2)

7. **Medium — Figure 7 scope mismatch.** Prose says “two datasets,” but the visible figure has only Azure/GovReport. (`references/morphserve-2506.02006-v2.pdf`, pp.9–10)

8. **Medium — mode naming is inconsistent.** Figure 1 uses `slo-pref`/`acc-pref`, while Figures 4 and 7 use performance/accuracy modes; equivalence is likely but never defined. (`references/morphserve-2506.02006-v2.pdf`, pp.3, 7, 9)

9. **Medium — missing Table 9.** Only Tables 1–8 exist in this v2 PDF. (`references/morphserve-2506.02006-v2.pdf`, complete document)

10. **Medium — Figure/raw-result reproducibility.** Figures 1 and 4–7 supply no data files, exact point values, error bars, run counts, warmup policy, or variance. Random Table 6 results are said to be averaged over “multiple runs” without a count. (`references/morphserve-2506.02006-v2.pdf`, pp.3, 7–10, 16–17)

11. **Medium — quantization configuration is incomplete.** Checkpoint IDs, AWQ group size/calibration, Uniform INT4 definition, tokenizer revisions, W8/W3 kernel details, and seeds are absent. (`references/morphserve-2506.02006-v2.pdf`, pp.5, 8, 18)

12. **Low — terminology/typesetting issues.** TOPT on p.1 becomes TPOT elsewhere; Eq.1 contains undefined `n_k` and \(\Delta\); Algorithm 1’s final `end for` and `return` collide; Table 6’s header says 32 INT4 although CodeLlama ends at 48. (`references/morphserve-2506.02006-v2.pdf`, pp.1, 4, 16–17)

13. **Objective/PDF mismatch — unsourced baselines.** The user objective adds LLM-PQ gap-closure values (41.3%/82.3%) and PyramidKV TTFT values (1.73×/2.4×). Neither result occurs in the target PDF or LaTeX source: LLM-PQ is absent and PyramidKV appears only as a related-work citation. They must not be treated as paper reference values.

## 9. Minimum faithful reproduction checklist

A defensible reproduction should therefore:

- Pin exact SwiftLLM/vLLM/AWQ/FlashAttention/CUDA commits and model/tokenizer checkpoints.
- Implement the monitor/dispatcher/controller/executor topology and separate morph/decode CUDA streams. (PDF pp.4, 18)
- Preserve per-request KVC across precision changes and validate synchronization at layer boundaries. (PDF pp.4, 6)
- Implement Algorithm 1 with WikiText-2 length 2,048 and the 0.25/0.25/0.5 weights, while also testing front-to-back as the documented fallback. (PDF pp.15–17)
- Reproduce Tables 1–8 before attempting aggregate Figure 4–7 claims.
- Explicitly choose and publish the missing runtime-policy values, trace offsets, sampling seed, block size, allocation/remapping API, and mode thresholds rather than presenting them as paper-specified.
- Distinguish exact table matches from approximate plot/aggregate-claim matches.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Reviewed all 19 pages of references/morphserve-2506.02006-v2.pdf; supplied page-by-page implementation requirements, Equations 1-7, exact Algorithm 1, exact Tables 1-8, explicit confirmation that Table 9 is absent, Figures 1-7 settings/claims, experimental conditions, severity-tagged review findings, and residual risks."
    }
  ],
  "changedFiles": [],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "mutool info references/morphserve-2506.02006-v2.pdf",
      "result": "passed",
      "summary": "Confirmed a 19-page PDF and inspected metadata/fonts."
    },
    {
      "command": "mutool draw -F txt -o /tmp/morphserve_pdf/page-%02d.txt references/morphserve-2506.02006-v2.pdf 1-19",
      "result": "passed",
      "summary": "Extracted text from every page for full-document review."
    },
    {
      "command": "mutool draw -r 150 -o /tmp/morphserve_pdf/page-%02d.png references/morphserve-2506.02006-v2.pdf 1-19",
      "result": "passed",
      "summary": "Rendered and visually inspected all pages, especially equations, figures, Algorithm 1, and Tables 1-8."
    },
    {
      "command": "sha256sum references/morphserve-2506.02006-v2.pdf",
      "result": "passed",
      "summary": "PDF SHA-256 is e2c0f12fcbc5188a04078a31a732acaa95e93e9662aff4b766a0b9d6a73fbe30."
    },
    {
      "command": "grep -h '^Table [0-9]' /tmp/morphserve_pdf/page-*.txt; grep -h '^Figure [0-9]' /tmp/morphserve_pdf/page-*.txt",
      "result": "passed",
      "summary": "Confirmed captions for Figures 1-7 and Tables 1-8; no Table 9 exists."
    },
    {
      "command": "git status --short; git diff --stat; git diff --cached --stat",
      "result": "passed",
      "summary": "No staged files or tracked diff from this review; an untracked reproduction/ directory was observed and was not created or modified by this task."
    }
  ],
  "validationOutput": [
    "All 19 PDF pages were text-extracted and rendered.",
    "All table cells were checked against page renders.",
    "Algorithm 1 and Equations 1-7 were visually checked.",
    "Table 9 is absent from the supplied v2 PDF."
  ],
  "residualRisks": [
    "Paper omits exact runtime thresholds, hysteresis, smoothing windows, block-allocation details, trace offsets, context sampling seeds, and full quantization configurations.",
    "Figures do not provide raw data; plotted point values cannot be treated as exact.",
    "Several stale table references, one Table 2 arithmetic discrepancy, and Figure 7 scope inconsistency remain unresolved.",
    "An untracked reproduction/ directory exists in the worktree but was not touched by this review."
  ],
  "noStagedFiles": true,
  "diffSummary": "No repository files were edited.",
  "reviewFindings": [
    "blocker: references/morphserve-2506.02006-v2.pdf:4,7-10 - exact online adaptation policy and mode thresholds are not specified.",
    "blocker: references/morphserve-2506.02006-v2.pdf:7,18-19 - exact trace windows, scaling method, request-to-dataset sampling, and seeds are missing.",
    "high: references/morphserve-2506.02006-v2.pdf:5-6,18 - preallocated in-place layer storage is not reconciled with reclaiming 75% of its memory for KVC.",
    "high: references/morphserve-2506.02006-v2.pdf:5,10,16,18 - multiple appendix table references are stale or incorrect.",
    "high: references/morphserve-2506.02006-v2.pdf:5,15-16 - LIS prose definitions conflict with the equations and the claim that activation-derived metrics are input-independent.",
    "medium: references/morphserve-2506.02006-v2.pdf:10 - Uniform INT4 degradation reduction computes to about 73.67%, not the captioned 74.67%.",
    "medium: references/morphserve-2506.02006-v2.pdf:9-10 - Figure 7 prose says two datasets while the figure shows only GovReport.",
    "medium: references/morphserve-2506.02006-v2.pdf - the supplied document contains no Table 9."
  ],
  "manualNotes": "Extraction uncertainty is limited mainly to unreported plot coordinates. Tables, equations, captions, and Algorithm 1 were cross-checked visually."
}
```
