# Paper-version delta audit

**Primary target after this audit:** `references/morphserve-mlsys2026-conference-final.pdf`

This artifact compares the supplied local MLSys conference-final PDF with the preserved older arXiv-v2 audit. It does not alter any measured local result.

## Immutable source identity

| Source | Absolute path | SHA-256 | Pages | Role |
|---|---|---|---:|---|
| MLSys 2026 conference final | `/nfs/home/s314511048/MorphServe/references/morphserve-mlsys2026-conference-final.pdf` | `080ddcfe8c23e12bb421e4c1107c09246c82144345064018401b49e1a8dd2678` | 20 | **Primary target** |
| arXiv v2 | `/nfs/home/s314511048/MorphServe/references/morphserve-2506.02006-v2.pdf` | `e2c0f12fcbc5188a04078a31a732acaa95e93e9662aff4b766a0b9d6a73fbe30` | 19 | Historical secondary source; retained |

The conference PDF metadata identifies the title as **“MorphServe: Efficient and Workload-Aware LLM Serving via Runtime Quantized Layer Swapping and KV Cache Resizing”**, authors **Zhaoyuan Su, Zeyu Zhang, Tingfeng Lan, Zirui Wang, Haiying Shen, Juncheng Yang, and Yue Cheng**, with University of Virginia and Harvard affiliations. It states “Proceedings of the 9th MLSys Conference, Bellevue, WA, USA, 2026.”

The final PDF was extracted with MuPDF to `sources/morphserve-mlsys2026-conference-final.txt` (SHA-256 `4045eb047ae9545aa4788028e3904c088a50b5dcc1928bb79c30c2bcdb37984a`). Extraction was run over pages 1–20. Pages 8–11 and 17–20 were rendered with MuPDF and visually inspected, covering the main result figures/tables, §5.1, final appendix tables, and Appendix C. Full-document text checks confirm Figures 1–7, Tables 1–9, Table 9, §5.1, Appendix C, LLM-PQ, PyramidKV, 41.3%, 82.3%, 1.73×, 2.4×, and the final implementation details are present.

## Verification result

All expected conference-final properties are present:

- exactly 20 pages;
- Figures 1–7;
- Tables 1–9, including Table 9 on PDF page 19;
- final §5.1 LLM-PQ comparison: 41.3% average and up to 82.3% accuracy-gap closure;
- final §5.1 PyramidKV comparison: 1.73× average and up to 2.4× P95 TTFT improvement in accuracy mode;
- Appendix C implementation, experiment-setting, trace, and dataset details;
- final title, author list, affiliations, and MLSys proceedings notice.

## Numbering and structural changes

The conference final promotes and renumbers material that was in the v2 appendix:

| Topic | arXiv-v2 audit | Conference final |
|---|---|---|
| Layer-wise independence | Table 7 / Appendix B.1 | Table 2 / Appendix B.1 |
| C4 cross-dataset LIS | Table 3 / §5.2 | Table 3 / §5.2 |
| AWQ/uniform Llama 2 + DuReader/BurstGPT | Table 2 / §5.2 | Table 4 / §5.2 |
| Runtime vs static selective Llama 3 + DuReader/BurstGPT | absent | **New Table 5** |
| LIS weight ablation | Table 4 / Appendix A | Table 6 / Appendix A.1 |
| cosine vs L2 | Table 5 / Appendix A | Table 7 / Appendix A.1 |
| layer-order perplexity | Table 6 / Appendix A | Table 8 / Appendix A.3 |
| Vicuna/QMSum/Azure quantization comparison | Table 8 / Appendix B.3 | **Table 9 / Appendix B.2** |

The figure numbering remains Figures 1–7. The final corrects the v2's missing Table 9 and changes appendix subsection labels and several stale cross-references. Algorithm 1 gains an explicit separate line 15 for `return Ordered layer swap sequence Q`; the v2 extracted text had `end forreturn` due a typesetting collision.

## Consequential result and claim changes

1. **New baselines in the final.** The final explicitly benchmarks LLM-PQ (planning-based mixed precision) and PyramidKV (KV-cache compression) in the comparison scope and Figure 4. LLM-PQ was absent from the v2 experiment scope. PyramidKV was only a related-work citation in v2.
2. **New §5.1 values.** The final reports that accuracy mode closes **41.3% on average and up to 82.3%** of LLM-PQ's accuracy gap at comparable or lower P95 TTFT. It reports accuracy-mode PyramidKV P95 TTFT reduction of **1.73× on average and up to 2.4×**, and performance-mode speedup of **4.25× on average and up to 8.68×**. These are paper-reported values, not local measurements.
3. **Existing aggregate ranges retained.** The final retains 92.45% average SLO-violation reduction, 2.2×–3.9× accuracy-mode P95 TTFT improvement, 2.9×–15.7× default, 3.4×–19.5× performance, 0.51%–3.82% quality degradation, 29.29% KVC utilization improvement, 3.58% output-accuracy improvement, 32.97% expansion beyond FP16, 1.6×–1.83× throughput, and P95/P99 TPOT factors up to 1.06×/1.23×.
4. **Corrected wording/numbers.** Final Table 4 captions the Uniform accuracy-degradation reduction as **73.66%**, correcting v2's 74.67% wording. Final Figure 6 says “up to 1.83× higher throughput.” Final prose consistently uses TPOT rather than the v2 abstract/introduction's TOPT variant. Final Table 5 adds: full W4 F1 23.66 / P95 TTFT 1.62 / SLO 0%; 16-layer selective 23.93 / 1.74 / 0%; 8-layer selective 24.16 / 2.61 / 4.2%; FP16 25.19 / 5.65 / 12.7%; MorphServe dynamic 24.63 / 1.77 / 0%.
5. **Expanded implementation claims.** Final Appendix C explicitly says all GEMM kernels are precompiled with dummy data, the critical swap path is a single `cudaMemcpyAsync`, Morphing and decoding use separate CUDA streams, and custom Triton kernels handle dynamic non-contiguous KV registration/remapping. It also names the source URL, L4/A100 settings, model context lengths, AWQ INT4 loading, trace URLs, 72-second sampling, 4.75× Azure and 1.75× BurstGPT downscaling, and dataset URLs. These details were either absent or less explicit in v2.
6. **New broader-impact/limitations wording.** The final adds workload forecasting, native FP8/FP4 implications, broader impacts, and explicit hysteresis language. It says host overhead is typically under 2× model size and notes finer-grained layer adaptation as future work.

## Reproduction consequence

The primary paper reference and claim register now use the conference final. The v2 PDF, v2 extracted text, v2 hashes, old report classifications, and historical measurements remain retained as secondary evidence. Local benchmark results are never renumbered or changed merely because the paper moved tables or added baselines. Exact LLM-PQ/PyramidKV paper values are now reference claims; they remain distinct from the modified-condition RTX 3090/Llama 3.1 8B measurements and do not make those measurements exact paper reproductions.

## Reproduction commands used

```bash
sha256sum references/morphserve-mlsys2026-conference-final.pdf references/morphserve-2506.02006-v2.pdf
mutool info references/morphserve-mlsys2026-conference-final.pdf
mutool draw -F txt -o sources/morphserve-mlsys2026-conference-final.txt references/morphserve-mlsys2026-conference-final.pdf
mutool draw -r 120 -o /tmp/morphserve-conference-final/png/page-%02d.png references/morphserve-mlsys2026-conference-final.pdf 1-20
```
