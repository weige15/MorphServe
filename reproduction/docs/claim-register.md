# Experiment and claim register

This is a living register. `reported` values come from the supplied paper and `observed` values may only come from saved raw outputs. No paper value below is a measurement from this workspace.

Status vocabulary: **reproduced**, **approximate/modified-condition**, **blocked exact**, **tested-not-reproduced**, **unverified**, **source-verified only**.

| ID | Paper location / scope | Reported reference | Required comparison/evidence | Current status |
|---|---|---|---|---|
| H1 | Abstract; §5.1 aggregate | Average SLO-violation reduction 92.45% | Exact matrix requests, 2 s TTFT SLO, per-cell violations and denominator, aggregation script | **Blocked exact**: trace windows, mapping, configs, and raw logs unavailable |
| H2 | §5.1, accuracy mode | P95 TTFT 2.2×–3.9× better than FP16 | Per-cell P95 from same hardware/workload and quality evidence | **Unverified** |
| H3 | §5.1, default mode | P95 TTFT 2.9×–15.7× better than FP16 | Same as H2, default-mode config fixed before run | **Unverified**; mode config unavailable |
| H4 | §5.1, performance mode | P95 TTFT 3.4×–19.5× better than FP16 | Same as H2, performance-mode config fixed before run | **Unverified**; mode config unavailable |
| H5 | §5.1 quality scope | F1/ROUGE-L degradation 0.51%–3.82%; accuracy mode 0.11%–2.18% | Generated texts, references, exact metric implementation and absolute/relative calculations | **Unverified** |
| H6 | §5.1 vs LLM-PQ | Accuracy mode closes quality gap 41.3% average, up to 82.3% | Recovered LLM-PQ plan and gap-closure denominator per cell | **Blocked exact**: layer plan/config absent |
| H7 | §5.1 vs PyramidKV | Accuracy-mode P95 TTFT 1.73× average, up to 2.4× | Exact PyramidKV policy with FP16 weights on common engine/workload | **Blocked exact**: policy/config absent |
| H8 | Fig. 5 | Dynamic KVC capacity follows load, prevents preemption/swap seen in FP16 | Physical capacity/occupancy time series, preemptions, block addresses | **Unverified**; candidate mechanism found |
| H9 | Fig. 6 | Saturation delayed; throughput up to 1.83× FP16 | Arrival-rate sweep, all arrivals accounted through completion/timeout, ≥3 repeats if feasible | **Unverified** |
| H10 | Fig. 7 | P99 TPOT up to 1.23× lower; performance-mode mean TPOT up to 1.17× better | Token timing logs and exact CDF/percentile regeneration | **Unverified** |
| H11 | §4.3 | Llama 2 7B transfer ≈4 ms W4, ≈16 ms FP16 on PCIe Gen4 26–28 GB/s | CUDA events for H2D bytes; distinguish transfer/reconstruction/stall | **Modified-condition tested-not-reproduced**: RTX 3090/Llama 3.1 layer blocking operation walls were 15.67 ms W4 and 58.55 ms FP16; transfer not isolated from stream/reconstruction overhead |
| H12 | §4.3 | Complete W4 layer swap ≈6 ms and fully hidden by decoding | CUDA timeline with distinct morph/decode streams and exposed stall | **Modified-condition tested-not-reproduced**: one real W4 switch completes correctly but takes 15.67 ms blocking; candidate synchronizes and no overlap timeline exists |
| H13 | Appendix A | Full 32-layer LIS sequence under 15 minutes on one GPU | Timed conditioned-MDS profiler excluding model init, exact model/calibration identified | **Unverified full claim**: real 8-layer/36-set inner work took 58.14 s (92.78 s with load); paper-style all-variant pinning exceeds local memlock by 741 MB; no extrapolation/full run |
| H14 | Table 1 | Eight Llama 3 8B BookSum 6K/2K schedules; exact F1/ROUGE-L in reference JSON | Same inputs/settings and real W4/FP16 precision history | **Blocked exact**: BookSum sample/preprocess/prompt/checkpoint unspecified and local context differs |
| H15 | PDF Table 2 (the objective calls this “Table 4”) | Llama 2 7B + English DuReader + BurstGPT: FP16 `[5.5223,0.1241,27.68]`; static AWQ `[1.1686,0.0735,25.55]`; MorphServe `[1.2420,0.1064,27.33]` | Exact trace/window, translated data, model, AWQ, engine, settings, raw request log | **Blocked exact**: model/data/trace window/config absent |
| H16 | PDF Table 2 | Uniform INT4 row in `paper-reference-values.json` | Same as H15 with sourced uniform-INT4 implementation | **Blocked exact** |
| H17 | PDF Table 8 / Appendix B.3 (the objective mentions nonexistent “Table 9”) | Vicuna/QMSum/Azure AWQ and uniform INT4 rows | Exact trace/data/model and metric (paper table uses F1) | **Blocked exact** |
| H18 | C4 transfer table | WikiText-2 LIS transfers to C4 and mostly beats heuristics | Recovered calibration choices and exact PPL evaluation | **Unverified** |
| H19 | LIS weight table | 0.25/0.25/0.5 rows in reference JSON | Exact Llama 2 7B AWQ, WikiText-2 inputs, implementation | **Unverified** |
| H20 | cosine-vs-L2 table | Cosine rows in reference JSON beat L2 before full W4 endpoint | Exact implementation and inputs | **Unverified** |
| H21 | layer-effect-independence table | Layer 19 effect ≈0.0035, layer 24 ≈0.0018–0.0020 | Exact incremental quantization PPL runs | **Unverified** |
| H22 | ordering table | Vicuna/Llama 2/Llama 3/CodeLlama values, CodeLlama endpoint is 48 W4 layers | All four exact checkpoints and quantized variants; random repetitions | **Unverified**; only local Llama 3.1 8B assets available |
| H23 | §4.2; Algorithm 1 | Conditioned greedy MDS, LIS weights 0.25/0.25/0.5, `argmax` | Unit tests that fail for static/unconditioned MDS; saved calibration and profile | **Approximate/modified-condition expanded reproduction**: 8-layer real WikiText-2/2048/W4 run executes all 36 conditioned calls and saves `[25,24,26,27,28,29,30,31]`; full 32-layer paper order remains unverified |
| H24 | §4.3; Appendix C | Real in-place W4/FP16 decoder-layer swapping, pinned variants, precompiled GEMM | Packed-weight storage including metadata, stable GPU base address, numerical execution, event/lifetime tests | **Approximate/modified-condition (partial)**: one real AutoAWQ layer runs from same FP16 base with zero allocator delta; 23/23 packed and 8/8 restored tensors exact; restored logits bit-exact; copies still synchronize and overlap/precompile remain unverified |
| H25 | §4.4; Appendix C | Reclaimed weight bytes become physically usable, non-contiguous KV blocks | Address-bound checks, mapping oracle, capacity expansion/shrink/content preservation | **Approximate/modified-condition (strong partial)**: real layer frees 322,895,872 bytes; active 8B request allocates reclaimed block, K/V matches same-history reference, byte-exact migration preserves block table; fixed-stride/one-request limits remain |
| H26 | §4.1 controller | Persistent pressure causes coordinated morph/resize and recovers without oscillation | Frozen mode configs and synthetic pressure tests | **Approximate/modified-condition integration partial**: 6 policy + 5 fake-executor tests cover signals, modes, persistence, FCFS preservation, expand/shrink ordering and rollback; settings not author-recovered, real GPU integration pending |
| H27 | Appendix C implementation size | ≈2,200 Python + 500 C++/CUDA LOC added to SwiftLLM | Known base commit and diff accounting method | **Blocked exact**: candidate source base commit absent; candidate code is circumstantially aligned |
| H28 | State preservation | Morphing needs no flush, re-prefill, restart, KV quantization, or KV eviction | Repeated prefill/decode adaptations against same-precision-history oracle | **Approximate/modified-condition reproduced for one pilot**: active request continues FP16 prefill→W4×4→FP16 with same-history top-1 agreement, reclaimed K/V, byte-exact migration, no flush/re-prefill/eviction; broader serving unverified |
| H29 | Baseline integrity | No-morph FP16 preserves SwiftLLM behavior | Token/logit parity and scheduler trace | **Approximate/modified-condition (numerical partial)**: normalized candidate Llama 3.1 8B matches Transformers top-1/top-5, relative L2 `0.00204`, 291/291 exact tensors; KV/scheduler trace remains unverified |
| H30 | Supporting fixed-W4 gate | Real packed static AWQ INT4 behavior and storage | Packed module/dtype/storage audit and repeated logits | **Mixed result**: 224 real `WQLinear_GEMM` modules and 3.626 GB decoder storage including metadata verified; top-k stable, but bit-exact repeat gate fails due split-K atomic variance (relative L2 up to 0.00493) |

The supplied 19-page PDF contains Tables 1–8 only. There is no Table 9. Several in-paper cross-references and the objective's Table 4/Table 9 labels are stale; this register follows the actual rendered PDF while retaining the mismatch explicitly.

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
5. Only after correctness gates, select a feasible modified-condition numerical target; never compare a local RTX 3090/Llama 3.1 run as an exact L4/Llama 3 8B reproduction.
