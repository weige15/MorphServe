# Experiment and claim register

This is a living register. `reported` values come from the **primary supplied MLSys 2026 conference-final PDF** and `observed` values may only come from saved raw outputs. No paper value below is a measurement from this workspace. The former arXiv-v2 classifications and measured evidence remain preserved in linked historical artifacts; see `docs/paper-version-delta.md`.

Current paper source: `../references/morphserve-mlsys2026-conference-final.pdf`, SHA-256 `080ddcfe8c23e12bb421e4c1107c09246c82144345064018401b49e1a8dd2678`, 20 pages. The conference final contains Tables 1–9; its table-number mapping is authoritative in `configs/paper-reference-values.json`.

Status vocabulary: **reproduced**, **approximate/modified-condition**, **blocked exact**, **tested-not-reproduced**, **unverified**, **source-verified only**.

| ID | Paper location / scope | Reported reference | Required comparison/evidence | Current status |
|---|---|---|---|---|
| H1 | Abstract; §5.1 aggregate | Average SLO-violation reduction 92.45% | Exact matrix requests, 2 s TTFT SLO, per-cell violations and denominator, aggregation script | **Blocked exact**: Figure 1 yields strong evaluation-window inferences, but sub-second boundaries, scaling/thinning, context mapping/config and raw logs remain unavailable |
| H2 | §5.1, accuracy mode | P95 TTFT 2.2×–3.9× better than FP16 | Per-cell P95 from same hardware/workload and quality evidence | **Unverified** |
| H3 | §5.1, default mode | P95 TTFT 2.9×–15.7× better than FP16 | Same as H2, default-mode config fixed before run | **Unverified**; mode config unavailable |
| H4 | §5.1, performance mode | P95 TTFT 3.4×–19.5× better than FP16 | Same as H2, performance-mode config fixed before run | **Unverified**; mode config unavailable |
| H5 | §5.1 quality scope | F1/ROUGE-L degradation 0.51%–3.82%; accuracy mode 0.11%–2.18% | Generated texts, references, exact metric implementation and absolute/relative calculations | **Unverified** |
| H6 | Conference-final §5.1 | LLM-PQ accuracy-gap closure 41.3% average, 82.3% max | Final PDF text/visual audit; any local comparison must use a separately frozen LLM-PQ implementation | **Source-verified paper reference; local comparison blocked**: absent from historical v2, now present in the primary conference final |
| H7 | Conference-final §5.1 | PyramidKV P95 TTFT improvement 1.73× average, 2.4× max in accuracy mode | Final PDF text/visual audit; any local comparison must use a separately frozen PyramidKV implementation | **Source-verified paper reference; local comparison blocked**: absent as an experiment from historical v2, now present in the primary conference final |
| H8 | Fig. 5 | Dynamic KVC capacity follows load, prevents preemption/swap seen in FP16 | Physical capacity/occupancy time series, preemptions, block addresses | **Approximate/modified-condition**: six common-engine runs save physical capacity/occupancy; MorphServe expands 512→1,736 blocks and recovers; paper preemption claim remains unverified |
| H9 | Fig. 6 | Saturation delayed; throughput up to 1.83× FP16 | Arrival-rate sweep, all arrivals accounted through completion/timeout, ≥3 repeats if feasible | **Approximate/modified-condition measurement**: first common-engine 123/94-request runs complete all requests, but no rate sweep and the paper multiplier is unverified |
| H10 | Fig. 7 | P99 TPOT up to 1.23× lower; performance-mode mean TPOT up to 1.17× better | Token timing logs and exact CDF/percentile regeneration | **Approximate/modified-condition measurement**: all six runs save per-token timestamps and type-7 TPOT summaries; exact paper comparison and performance mode remain unverified |
| H11 | §4.3 | Llama 2 7B transfer ≈4 ms W4, ≈16 ms FP16 on PCIe Gen4 26–28 GB/s | CUDA events for H2D bytes; distinguish transfer/reconstruction/stall | **Modified-condition tested-not-reproduced**: isolated 3-repeat RTX 3090/Llama 3.1 copies are W4 15.174–15.475 ms and FP16 57.769–58.293 ms (~7.3–7.6 GB/s); model/hardware differ |
| H12 | §4.3 | Complete W4 layer swap ≈6 ms and fully hidden by decoding | CUDA timeline with distinct morph/decode streams and exposed stall | **Modified-condition partial, numerical claim unverified**: candidate remains blocking; independent persistent-stream copier queues 4 MiB in 0.206 ms while prior use is unfinished and event/transaction tests pass, but full-layer/decode timeline is pending GPU headroom |
| H13 | Appendix A | Full 32-layer LIS sequence under 15 minutes on one GPU | Timed conditioned-MDS profiler excluding model init, exact model/calibration identified | **Unverified full claim**: real 8-layer/36-set inner work took 58.14 s (92.78 s with load); paper-style all-variant pinning exceeds local memlock by 741 MB; no extrapolation/full run |
| H14 | Table 1 | Eight Llama 3 8B BookSum 6K/2K schedules; exact F1/ROUGE-L in reference JSON | Same inputs/settings and real W4/FP16 precision history | **Blocked exact**: BookSum sample/preprocess/prompt/checkpoint unspecified and local context differs |
| H15 | Conference-final Table 4 (historical v2 Table 2; objective label is now correct) | Llama 2 7B + English DuReader + BurstGPT: FP16 `[5.5223,0.1241,27.68]`; static AWQ `[1.1686,0.0735,25.55]`; MorphServe `[1.2420,0.1064,27.33]` | Exact trace/window, translated data, model, AWQ, engine, settings, raw request log | **Blocked exact**: model/data/trace window/config absent |
| H16 | PDF Table 2 | Uniform INT4 row in `paper-reference-values.json` | Same as H15 with sourced uniform-INT4 implementation | **Blocked exact** |
| H17 | Conference-final Table 9 / Appendix B.2 (historical v2 Table 8) | Vicuna/QMSum/Azure AWQ and uniform INT4 rows | Exact trace/data/model and metric (paper table uses F1) | **Blocked exact** |
| H18 | C4 transfer table | WikiText-2 LIS transfers to C4 and mostly beats heuristics | Recovered calibration choices and exact PPL evaluation | **Unverified** |
| H19 | Conference-final Table 6 / Appendix A.1 | 0.25/0.25/0.5 rows in reference JSON | Exact Llama 2 7B AWQ, WikiText-2 inputs, implementation | **Unverified** |
| H20 | Conference-final Table 7 / Appendix A.1 | Cosine rows in reference JSON beat L2 before full W4 endpoint | Exact implementation and inputs | **Unverified** |
| H21 | Conference-final Table 2 / Appendix B.1 | Layer 19 effect ≈0.0035, layer 24 ≈0.0018–0.0020 | Exact incremental quantization PPL runs | **Unverified** |
| H22 | Conference-final Table 8 / Appendix A.3 | Vicuna/Llama 2/Llama 3/CodeLlama values, CodeLlama endpoint is 48 W4 layers | All four exact checkpoints and quantized variants; random repetitions | **Unverified**; only local Llama 3.1 8B assets available |
| H23 | §4.2; Algorithm 1 | Conditioned greedy MDS, LIS weights 0.25/0.25/0.5, `argmax` | Unit tests that fail for static/unconditioned MDS; saved calibration and profile | **Approximate/modified-condition expanded reproduction**: 8-layer real WikiText-2/2048/W4 run executes all 36 conditioned calls and saves `[25,24,26,27,28,29,30,31]`; full 32-layer paper order remains unverified |
| H24 | §4.3; Appendix C | Real in-place W4/FP16 decoder-layer swapping, pinned variants, precompiled GEMM | Packed-weight storage including metadata, stable GPU base address, numerical execution, event/lifetime tests | **Approximate/modified-condition**: real AutoAWQ layer runs from same FP16 base with zero allocator delta and exact restore; six common-engine runs execute real FP16/static W4/MorphServe paths; persistent-stream copier and full-model overlap gates also pass; paper timing remains unverified |
| H25 | §4.4; Appendix C | Reclaimed weight bytes become physically usable, non-contiguous KV blocks | Address-bound checks, mapping oracle, capacity expansion/shrink/content preservation | **Approximate/modified-condition repaired**: explicit fallback fixes vendor misrouting; current MorphServe runs physically expand/recover KV capacity while all requests complete; two-request ownership and dense-oracle gates pass; paper scheduler behavior remains unverified |
| H26 | §4.1 controller | Persistent pressure causes coordinated morph/resize and recovers without oscillation | Frozen mode configs and synthetic pressure tests | **Approximate/modified-condition**: frozen reconstructed default controller drives four morphs/four recoveries per workload, reaches eight real W4 layers and 1,736 KV blocks, and leaves clean final state; exact author settings and paper modes remain unverified |
| H27 | Appendix C implementation size | ≈2,200 Python + 500 C++/CUDA LOC added to SwiftLLM | Known base commit and diff accounting method | **Blocked exact**: candidate source base commit absent; candidate code is circumstantially aligned |
| H28 | State preservation | Morphing needs no flush, re-prefill, restart, KV quantization, or KV eviction | Repeated prefill/decode adaptations against same-precision-history oracle | **Approximate/modified-condition**: active same-history and two-request ownership tests pass; six common-engine runs complete 1,024/512-token requests without re-prefill; ordinary scheduler preemptions and exact paper trace remain unverified |
| H29 | Baseline integrity | No-morph FP16 preserves SwiftLLM behavior | Token/logit parity and scheduler trace | **Approximate/modified-condition (numerical partial)**: normalized candidate Llama 3.1 8B matches Transformers top-1/top-5, relative L2 `0.00204`, 291/291 exact tensors; KV/scheduler trace remains unverified |
| H30 | Supporting fixed-W4 gate | Real packed static AWQ INT4 behavior and storage | Packed module/dtype/storage audit and repeated logits | **Mixed result**: 224 real `WQLinear_GEMM` modules and 3.626 GB decoder storage including metadata verified; top-k stable, but bit-exact repeat gate fails due split-K atomic variance (relative L2 up to 0.00493) |

The preserved historical arXiv-v2 PDF contains Tables 1–8 only. The primary 20-page conference-final PDF contains Tables 1–9. The conference-final mapping and values are in `configs/paper-reference-values.json`; the v2 numbering mismatch is retained explicitly in `docs/paper-version-delta.md`.

## Input and configuration register

| Item | Paper fact | Artifact-recovered fact | Classification |
|---|---|---|---|
| Calibration sequence length | 2,048 | None beyond paper | explicit |
| Calibration dataset | WikiText-2 | Local cached WikiText-2 exists | explicit / available |
| LIS weights | 0.25, 0.25, 0.5 | No candidate-code profiler | explicit / missing implementation |
| Layer order fallback | Front-to-Back | Candidate runtime currently starts from last layer (`num_layers-1`) and decrements | explicit paper / candidate conflict |
| Quantization | Weight-only AWQ INT4 for main setup | Local Llama 3.1 8B AutoAWQ W4 G128 zero-point asset exists; candidate expects a different checkpoint loader shape | explicit / modified asset |
| Azure trace | 72 seconds, 4.75× downscaling | No exact file/window | partially specified |
| BurstGPT trace | 72 seconds, 1.75× downscaling | No exact file/window | partially specified |
| TTFT SLO | 2 seconds | No replay config | explicit |
| L4 host | 24 GB HBM, 256 GB DRAM | Local: RTX 3090 24 GB, 125 GiB DRAM | hardware deviation |
| A100 host | 80 GB HBM, 2 TB DRAM | unavailable | blocked |
| Controller examples | KV usage >85%, queue delay >100 ms | Candidate scheduler is capacity-triggered and has no mode configs | examples only / unresolved |

## Next evidence gates

1. Establish whether the candidate artifact imports/builds unmodified and save every failure.
2. Create a separate minimal runtime normalization (package name/config plumbing only), then run candidate no-morph FP16 on a tiny/local model before trusting morph paths.
3. Unit-test Algorithm 1 independently with conditioned MDS.
4. Exercise reclaimed-KV mapping and layer swap on synthetic tensors before full-model serving.
5. Completed: the first common-engine modified-condition workload comparison is in `experiments/end-to-end-report.md`; preserve its raw telemetry and do not relabel it as exact L4/Llama 3 8B or paper-context reproduction.
