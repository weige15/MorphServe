# MorphServe Reproduction Report

**Investigation status:** current-revision GPU implementation verification complete / partial modified-condition reproduction / exact paper conditions blocked
**Paper target:** supplied 20-page MLSys 2026 conference-final PDF
**Overall result:** all four frozen current-revision full-model GPU verification surfaces pass on one RTX 3090 under the labeled reconstruction. No headline paper latency/quality aggregate or exact-condition table is reproduced. The official lab repository remains README-only. Several candidate-code correctness defects were found and repaired only in the labeled reconstruction.

## 1. Scope and source truth

The supplied 20-page conference-final PDF was read and visually checked page by page. It contains Figures 1–7 and Tables 1–9. The objective's AWQ `[5.5223,0.1241,27.68]` target is the conference-final Table 4 row; the historical 19-page arXiv-v2 PDF called this Table 2. Exact reported values are isolated in `configs/paper-reference-values.json`; measured values never populate that file. The conference final also contains the LLM-PQ/PyramidKV comparison references, but exact task/config reproduction remains blocked.

Primary evidence:

- Paper audit: `docs/paper-evidence-brief.md`
- Source/implementation map: `docs/source-map.md`
- Claim register: `docs/claim-register.md`
- Public code audit: `docs/author-artifact-audit.md`
- Trace audit: `docs/trace-audit.md`
- Environment/model hashes: `results/raw/environment.json`

## 2. Source and implementation fidelity

| Component | Status | Evidence and limitation |
|---|---|---|
| Official author/lab implementation | **Blocked** | `ds2-lab/MorphServe@1c42999...` contains only LICENSE/README and promises future code. |
| Candidate project-account source | **Partial, authorship unresolved** | `MorphServe/MorphServe@85c4fbf...`, preserved under `vendor/author-morphserve`; not linked by paper/lab identity. |
| Unmodified candidate startup | **Tested-not-reproduced** | Package/config/import failures in `experiments/candidate-artifact-smoke/`. |
| FP16 engine behavior | **Modified-condition reproduced** | Local Llama 3.1 8B top-1/top-5 match Transformers; relative logit L2 0.00204; 291/291 tensors exact. |
| Real packed W4 | **Modified-condition reproduced** | AutoAWQ INT32 qweight/qzeros + FP16 scales; not fake quantization. Candidate's public llm-awq call/checkpoint format is incompatible. |
| In-place one-layer W4 switch | **Modified-condition reproduced** | Same GPU base, zero allocator delta, 23/23 W4 and 8/8 restored tensors exact; restored logits exact. |
| Physical KV reclaim | **Modified-condition reproduced** | One real layer frees 322,895,872 bytes after metadata; synthetic and active-request accesses use real storage. |
| Arbitrary non-contiguous/profile order | **Vendor negative; reconstruction repaired** | Vendor `[25,24,26]` writes group 2 into layer 23. Explicit-region fallback writes layer 26 and matches dense attention error 0. |
| CUDA lifetime safety | **Vendor negative; reconstruction repaired** | Uncoordinated restore corrupts 1,530 bytes. Recorded per-region events reduce corruption to 0. |
| State preservation | **Modified-condition strong partial** | Active FP16→W4×4→FP16 request continues without re-prefill/eviction; same-history top-1 agrees; K/V migrated exactly. |
| Controller | **Reconstructed, not author-recovered** | Frozen machine-readable EMA/persistence/hysteresis/mode choices in `configs/reconstructed-controller-modes.json`; current real-executor and ownership GPU gates pass on the independent atomic reconstruction. Settings remain not author-recovered. |
| Full 32-layer preloading | **Local resource blocked** | Local FP16+W4 decoder variants require 17,585,668,096 pinned bytes, 741,253,120 bytes above memlock before overhead. |

## 3. Reproduction environment

Paper hardware: L4 24 GB + 256 GB RAM for 7B/8B and A100 80 GB + 2 TB for 34B. Local hardware is 7× RTX 3090 24 GB with 125 GiB RAM and 16,844,414,976-byte memlock. Driver/CUDA/PyTorch/Triton/checkpoint hashes and PCIe link evidence are in `results/raw/environment.json`. Saved GPU-runner intervals total 818.287 process-wall seconds across 26 successful/failed records; this includes model load/JIT/test work and is not kernel or exclusive GPU time (`results/raw/resource-usage-summary.json`).

Main local model is **Llama 3.1 8B base**, snapshot `d04e592...`, not a silently substituted exact “Llama 3 8B” paper revision. W4 is a local AutoAWQ G128 zero-point GEMM derivative. All results using these are labeled modified condition.

## 4. Key measured results

### 4.1 FP16 numerical baseline

| Metric | Observed |
|---|---:|
| Relative logit L2 vs Transformers | 0.0020401 |
| Max absolute logit error | 0.0234375 |
| Top-1/top-5 | exact match |
| Loaded tensor audit | 291/291 exact |

Evidence: `experiments/candidate-fp16-baseline/results-attempt-3/metrics.json`.

### 4.2 Real W4 storage and switching

| Metric | Observed |
|---|---:|
| FP16 layer region | 436,224,000 bytes |
| W4 layer including norms/qweight/qzeros/scales | 113,328,128 bytes |
| Reclaimable tail | 322,895,872 bytes (74.02%) |
| GPU allocator delta | 0 bytes |
| Blocking W4 operation wall | 15.67 ms |
| Blocking FP16 restore wall | 58.55 ms |
| Async W4 H2D CUDA copy, 3 repeats | 15.174–15.475 ms; median 15.214 ms (7.32–7.47 GB/s) |
| Async FP16 H2D CUDA copy, 3 repeats | 57.769–58.293 ms; median 57.894 ms (7.48–7.55 GB/s) |
| Async host enqueue | 0.276–0.427 ms; transfer incomplete at return in 6/6 cases |
| Paper example (different Llama 2/L4-like PCIe Gen4 condition) | ≈4/16 ms transfer; ≈6 ms complete W4 |

The isolated local transfer timings are stable but do not reproduce the paper examples, under different model/layer, RTX 3090/host and runtime conditions. The strengthened current-revision full-model run passes three unprofiled W4/FP16 timing repeats plus a separate raw CUDA-activity cycle. It establishes actual size-matched H2D/kernel intersection on distinct streams, but remains a modified-condition measurement rather than a paper-latency reproduction.

Evidence: `experiments/autoawq-layer-switch/results/metrics.json` and `experiments/async-layer-transfer/full-model-results-attempt-1/transfer-summary.json`.

### 4.3 Active state preservation

Schedule: FP16 prefill → layer-31 W4 for four decode steps → FP16 decode. Forced same token history: `[505,279,3363,11,304]`.

- W4-step reference/in-place relative logit L2: 0.00058–0.00088; all top-1 match.
- Reclaimed/reference K and V relative L2: 0.000095 / 0.000207.
- Occupied reclaimed block migration: byte exact.
- Final FP16/reference relative L2: 0.000354; top-1 match.
- No re-prefill, model reload, scheduler restart, KV quantization, or eviction.

Evidence: `experiments/active-kv-switch/results-attempt-2/metrics.json`.

### 4.4 Offline LIS profiling

Frozen reconstruction semantics: first 2,048 WikiText-2 train tokens; flattened float32 full-layer residual streams for LTS/LRS; flattened last-token logits for MDS; zero-based indices; lowest-index ties.

Eight-layer order: **`[25,24,26,27,28,29,30,31]`**.

- 36/36 conditioned candidate sets evaluated.
- Candidate counts 8→1.
- 336 restored tensors and final FP16 logits exact.
- Inner profile work 58.14 s; runner wall 92.78 s.

This does not reproduce a full 32-layer paper order or the under-15-minute claim. Profile: `profiles/llama31-8b-wikitext2-layers24-31.json`.

### 4.5 Transactional controller/executor

The following historical results remain preserved for comparison. The current-revision rerun below executes the atomic implementation and is the evidence used for this goal.

Accuracy-mode modified-condition pilot:

- Injected first KV expansion failure rolls back to exact FP16/4-block state.
- Successful profile order `[25,24,26]` grows 4→1,849 physical blocks (615/group).
- Non-descending third layer selects explicit mapping.
- Recovery shrinks/restores `26→24→25`.
- Final FP16 logits/capacity exact; FCFS queue order unchanged.

Two-request ownership pilot:

- Request block rows `[0,1,2,3,4]` and `[5]` occupy reclaimed group 0.
- Recovery removes free groups 26/24, refuses occupied group 25 while preserving rows/counts/sentinel/FCFS, then succeeds after real frees.
- Swapped queue unchanged. Ordinary scheduler preemptions were not measured because this bounded static scheduler has no cumulative preemption counter.

Historical evidence: `experiments/real-executor/results-before-atomic-repair/` and `experiments/multirequest-ownership/results-before-atomic-repair/`. Current-revision evidence: `experiments/real-executor/results/metrics.json` and `experiments/multirequest-ownership/results/metrics.json`, both passing with source-revision and GPU before/after sidecars. The historical directories remain explicitly classified and are not overwritten.

### 4.6 Asynchronous copy seam

The immutable candidate remains blocking (`cudaMemcpyAsync` followed by immediate stream synchronization). The independent reconstruction now exposes the registered GPU layer region, prebuilds FP16/W4 wrapper variants, and queues pinned copies on one persistent morphing stream. A model-wide last-forward event protects old use; the decode stream waits only immediately before the replaced layer.

A current-revision 4 MiB correctness pilot enqueued in 0.244 ms while its injected prior-use event was unfinished; the CUDA-event copy interval was 0.562 ms, address was unchanged, and all bytes matched. The device was externally occupied at 100%, so those durations remain supporting diagnostics. Model-use, partial-expansion barrier, no-redundant-event, transaction, alignment and invalid-source checks pass. The rejected full-model attempt 1 is preserved separately for its JIT and insufficient-interval defects. After the focused atomic repairs, the current full-model run passed three W4/FP16 timing repeats and the separate raw CUDA-activity gate. W4 copies were 14.8536–14.8766 ms and FP16 copies 53.1627–53.2377 ms; host enqueue was 0.213–0.397 ms and incomplete at return for all six rows. The raw trace found size-matched H2D copies on morph stream 17 intersecting kernels on decode stream 7 by 288.290 μs (W4) and 290.210 μs (FP16). Same-history top-1 and tolerance gates passed, and final FP16 bytes were exact. These are modified-condition measurements, not the paper's ≈4/16/6 ms examples.

Evidence: `experiments/async-layer-transfer/`. The separately valid transfer subset regenerates byte-identically as `figures/transfer-diagnostics.{csv,pdf,png}`; its caption explicitly excludes overlap and paper-agreement claims.

## 4.7 Current-revision GPU closure

**Exact execution source revision:** `67bbdc2d7094c2b6dcaab6e58c745d1a1a0a815b`, recorded in each current runner's `source-revision.txt`; subsequent audit commits are direct descendants and preserve the run sidecars.

| Surface | Command | GPU/provenance | Status |
|---|---|---|---|
| Transactional executor | `CUDA_VISIBLE_DEVICES=0 reproduction/scripts/run_real_executor_pilot.sh` | RTX 3090 GPU 0; 24,124 MiB free before/pre-run/after; manifests pass | **PASS** |
| Multi-request ownership | `CUDA_VISIBLE_DEVICES=0 reproduction/scripts/run_multirequest_ownership.sh` | RTX 3090 GPU 0; 24,124 MiB free before/pre-run/after; manifests pass | **PASS** |
| Full-layer async overlap | `CUDA_VISIBLE_DEVICES=0 reproduction/scripts/run_async_full_model_overlap.sh` | RTX 3090 GPU 0; 24,124 MiB free before/pre-run/after; manifests pass | **PASS** |
| Corrected synthetic replay | `CUDA_VISIBLE_DEVICES=0 reproduction/scripts/run_synthetic_gpu_replay.sh` | RTX 3090 GPU 0; 24,124 MiB free before/pre-run/after; manifests pass | **PASS** |

Main correctness evidence is exact rollback/final state for the executor, two-request occupied-region refusal plus post-release recovery for ownership, and exact final FP16 restoration for async swapping. Raw activity analysis proves size-matched H2D/kernel intersections on distinct streams 17/7 of 288.290 μs (W4) and 290.210 μs (FP16); transfer/decode/event summaries are in `full-model-results/{metrics,transfer-summary,activity-analysis}.json`. Replay accounts for all three independent arrivals exactly once, 3 tokens each (9 total), zero errors/timeouts, explicit KV occupancy/capacity and `preemptions: null`, with warmup/init excluded.

Current claim classifications strengthened only for bounded implementation evidence: **H8, H10, H12, H24, H25, H26 and H28**. No paper headline claim changed to reproduced. Remaining blockers before paper-facing experiments are exact author code/configuration, exact model/task/data revisions and prompts, trace scaling/context mapping, controller settings, and paper-equivalent hardware.

This closes current-revision implementation verification, not the paper's headline latency, quality, throughput or exact-table reproduction.

## 4.8 First common-engine end-to-end workload comparison

The first auditable modified-condition comparison of FP16, static real W4, and reconstructed MorphServe-default is complete for both frozen figure-inferred workloads. It uses 1,024-token prompts and exactly 512 generated tokens, with raw per-request/system/controller telemetry and clean final-state gates. Results and limitations are in `experiments/end-to-end-report.md`; raw artifacts are under `experiments/end-to-end-{burstgpt,azure}/`; regenerated plots and consolidated metrics are under `figures/end-to-end/`. The pre-Azure BurstGPT three-condition audit is `results/raw/burstgpt-end-to-end-audit.json`.

| Workload | FP16 output tok/s | static W4 output tok/s | MorphServe-default output tok/s | MorphServe peak KV blocks | All requests exact 512 |
|---|---:|---:|---:|---:|---|
| BurstGPT (123) | 164.584 | 114.647 | 72.243 | 1,736 | yes |
| Azure Code (94) | 165.527 | 114.727 | 68.115 | 1,736 | yes |

These are not paper-condition latency/quality results: hardware, exact model revision, task contexts, trace operation, context mapping, and controller settings differ or are unavailable. All six runs are valid modified-condition measurements, not confirmation of the paper's headline claims. No quality metric, LLM-PQ value, PyramidKV comparison, or exact paper-table agreement is inferred.

## 5. Claim-by-claim status

`configs/claim-evidence-map.json` provides a machine-readable H1–H30 index of each claim's paper-reference, executed command, frozen config/protocol, raw artifact, comparison/analysis, and limitation paths. `results/raw/claim-evidence-provenance.json` hashes and Git-audits the current linked artifacts. Empty command/raw lists explicitly mean that no measurement is claimed. Historical runners predate exact source-revision capture, so their artifact commit is preserved but is not presented as proof of the executed source tree; the four current runners write `source-revision.txt` and source-status sidecars.

| ID | Paper claim/location | Reported | Observed/agreement | Classification and evidence |
|---|---|---|---|---|
| H1 | Abstract/§5.1 average SLO reduction | 92.45% | No exact replay/result | **Blocked exact**: strong figure-inferred evaluation windows exist, but sub-second/scaling/context/mode/raw logs are missing. `docs/trace-audit.md` |
| H2 | §5.1 accuracy P95 TTFT | 2.2×–3.9× FP16 | Not measured under paper conditions | **Blocked exact** |
| H3 | §5.1 default P95 TTFT | 2.9×–15.7× | Not measured | **Blocked exact** |
| H4 | §5.1 performance P95 TTFT | 3.4×–19.5× | Not measured | **Blocked exact** |
| H5 | §5.1 quality degradation | 0.51%–3.82%; accuracy 0.11%–2.18% | No exact task corpus/generated outputs | **Blocked exact** |
| H6 | User objective only; absent from PDF | 41.3% average, 82.3% max LLM-PQ gap closure | No occurrence in target PDF/LaTeX | **Not a target-paper claim; not reproduced** |
| H7 | User objective only; absent from PDF | 1.73× average, 2.4× max vs PyramidKV | PyramidKV appears only as related-work citation | **Not a target-paper result; not reproduced** |
| H8 | Fig. 5 dynamic capacity | load-following KV expansion | Six common-engine workload runs save physical capacity/occupancy; MorphServe expands 512→1,736 blocks and recovers; ownership gates preserve occupied regions | **Modified-condition partial; paper preemption result unverified** |
| H9 | Fig. 6 throughput | up to 1.83× FP16 | First common-engine BurstGPT (123) and Azure Code (94) runs complete all requests and save output-token throughput; no rate sweep | **Modified-condition measurement; paper multiplier unverified** |
| H10 | Fig. 7 TPOT | P99 up to 1.23×; average up to 1.17× | Six common-engine runs save every output-token timestamp and regenerated type-7 TTFT/TPOT, in addition to the corrected replay seam | **Modified-condition measurement; exact paper comparison and performance mode unverified** |
| H11 | §4.3 transfer | ≈4 ms W4, ≈16 ms FP16 | 15.67/58.55 ms blocking modified operation | **Tested-not-reproduced under modified conditions** |
| H12 | §4.3 complete swap/overlap | ≈6 ms, hidden | Current full-layer W4/FP16 run passes raw activity overlap: 288.290/290.210 μs distinct-stream intersections; exposed delay remains 0.614–1.138 ms W4 and 37.781–37.895 ms FP16 | **Modified-condition overlap verified; paper numerical claim unverified** |
| H13 | Appendix A profile time | <15 min for 32 layers | 8-layer inner 58.14 s; full simultaneous pin blocked | **Full claim unverified** |
| H14 | Table 1 BookSum schedules | exact 8-row F1/ROUGE-L | Exact sample/prompts/checkpoint unavailable | **Blocked exact** |
| H15 | Conference-final Table 4 AWQ/DuReader/Burst (historical v2 Table 2) | FP16 `[5.5223,.1241,27.68]`; AWQ `[1.1686,.0735,25.55]`; MorphServe `[1.2420,.1064,27.33]` | No exact Llama 2/translated DuReader/window/controller | **Blocked exact** |
| H16 | PDF Table 2 Uniform INT4 | values in reference JSON | Uniform implementation/config unavailable | **Blocked exact** |
| H17 | Conference-final Table 9 Vicuna/QMSum/Azure (historical v2 Table 8) | values in reference JSON | Exact model/window/prompt unavailable | **Blocked exact** |
| H18 | Table 3 C4 transfer | LIS rows in reference JSON | No full 32-layer profile/perplexity run | **Unverified** |
| H19 | Table 4 LIS weights | reference perplexities | Formula used; table values not run | **Algorithm reproduced, numerical table unverified** |
| H20 | Table 5 cosine vs L2 | reference perplexities | Cosine implemented; L2 table not run | **Partial** |
| H21 | Table 7 layer independence | layer-19/24 PPL effects | Exact Llama 2 assets unavailable | **Blocked exact** |
| H22 | Table 6 ordering | four models, CodeLlama 48 endpoint | 8-layer local order only | **Modified-condition partial** |
| H23 | §4.2/Algorithm 1 | conditioned argmax LIS | 8 layers, 36 sets, saved real order | **Modified-condition expanded reproduction** |
| H24 | §4.3 in-place W4/FP16 | real packed same-address swapping | Real W4 same base/exact restore/zero allocator delta; current full-layer async W4/FP16 run passes three repeats, same-history gates, raw activity overlap and exact FP16 restoration | **Modified-condition implementation verified; paper timing unverified** |
| H25 | §4.4 non-contiguous KV | physical arbitrary-region capacity | Vendor corruption found; explicit fallback CUDA test passed; common-engine MorphServe runs physically expand/recover capacity while ownership gates preserve rows/counts/sentinels | **Modified-condition repaired and current integrated implementation verified; paper scheduler unverified** |
| H26 | §4.1 controller | persistent coordinated adaptation, 3 modes | Frozen reconstructed default controller drives four real pressure morphs/four recoveries per workload, reaches eight W4 layers and 1,736 blocks, and ends clean | **Modified-condition implementation/workload measurement; author settings and other modes unverified** |
| H27 | Appendix C added LOC | ≈2200 Python +500 C++/CUDA | Base commit unavailable | **Blocked exact** |
| H28 | state preservation | no flush/re-prefill/eviction | Active same-history/ownership gates pass, and all six 1,024/512-token common-engine runs complete without re-prefill or eviction | **Modified-condition bounded serving/state evidence; exact paper behavior unverified** |
| H29 | no-morph FP16 | baseline integrity | top-k match, rel L2 0.00204, exact weights | **Modified-condition reproduced numerically** |
| H30 | supporting fixed W4 | real packed/deterministic behavior | Packed execution/storage verified; bit-exact repeats fail from atomic split-K, top-k stable | **Mixed result** |

## 6. Important negative findings

1. Candidate package is not installable/runnable unmodified.
2. Candidate C++ static `py::object` metadata segfaults at interpreter shutdown.
3. Candidate restore has no in-flight lifetime barrier; delayed writes corrupt restored weights.
4. Candidate fixed-stride KV mapping silently corrupts unrelated layers for real LIS order `[25,24,26]`.
5. Candidate public llm-awq API/checkpoint assumptions match neither public llm-awq nor local AutoAWQ.
6. AutoAWQ split-K fused logits are not bit-exact across repeats.
7. Candidate/local blocking-path swaps are much slower than paper examples; the independent async path now has full-layer overlap evidence, but its RTX 3090 timings are not paper-condition results.
8. Full local all-variant pinning exceeds memlock.
9. Exact trace/task/controller identity is absent.
10. The paper-linked `baidu/DuReader@c625076...` tree contains no English/translation artifact or release despite Appendix C stating that the English-translated version is hosted there.

Repairs are isolated under `runtime/` and never attributed to author code.

## 7. Trace status

Recovered primary files:

- Azure Code SHA `54e9a6...`, 8,819 requests.
- Azure Conversation SHA `2f1e5b...`, 19,366 requests.
- BurstGPT v1.1 SHA `4bb378...`, 1,429,737 requests.

Figure 1a matches Azure Code, not Conversation. Approximate Figure 1b shape matching ranks Azure relative second 1073 (443 requests/72 s; Pearson 0.8793) and BurstGPT timestamp 1,781,278 (214 requests/72 s; Pearson 0.7356), both with the same top candidate under request-count and token-volume rankings. Section 5 says evaluation uses the representative snippets in Figure 1, so these are frozen before serving outcomes as **figure-inferred evaluation-window candidates**, though sub-second boundaries remain approximate rather than explicitly published. A separately labeled deterministic systematic-thinning reconstruction produces 94 Azure and 123 Burst arrivals in `traces/figure1b-inferred/`; contexts remain unmapped. The unpublished author operation/seed and request-to-context map still block exact replay. The corrected synthetic GPU replay now passes as a short modified-condition instrumentation baseline; its raw records, warmup metadata, 3/3 completion and 9/9 token accounting are under `experiments/synthetic-gpu-replay/results/`. The first common-engine end-to-end runs now also exist for 123 BurstGPT and 94 Azure Code requests at 1,024/512 tokens; see §4.8 and `experiments/end-to-end-report.md`. They are modified-condition measurements, not exact Azure/BurstGPT context replay or paper-latency evidence.

## 8. Reproduction commands

See `doc/onboarding.md`. Core commands:

```bash
PYTHONPATH="$PWD/reproduction/runtime:$PWD/reproduction/runtime/candidate-python" \
  python3 -m unittest reproduction.tests.test_profiling \
  reproduction.tests.test_candidate_runtime reproduction.tests.test_controller \
  reproduction.tests.test_controller_integration reproduction.tests.test_replay -v

CUDA_VISIBLE_DEVICES=1 reproduction/scripts/run_autoawq_layer_switch.sh
CUDA_VISIBLE_DEVICES=1 reproduction/scripts/run_active_kv_switch.sh
CUDA_VISIBLE_DEVICES=1 reproduction/scripts/run_lis_real_pilot.sh
CUDA_VISIBLE_DEVICES=1 reproduction/scripts/run_real_executor_pilot.sh
CUDA_VISIBLE_DEVICES=1 reproduction/scripts/run_multirequest_ownership.sh

rm -rf /tmp/morphserve-analysis-venv
uv venv --python python3.12 /tmp/morphserve-analysis-venv
uv pip install --python /tmp/morphserve-analysis-venv/bin/python \
  -r reproduction/configs/analysis-requirements.txt
/tmp/morphserve-analysis-venv/bin/python \
  reproduction/figures/gen_fig_transfer_diagnostics.py \
  --input reproduction/experiments/async-layer-transfer/full-model-results-attempt-1/metrics.json \
  --output-prefix reproduction/figures/transfer-diagnostics
```

Every experiment directory includes its locked protocol, commands, raw logs, machine-readable metrics, and verification output. Regenerate the claim provenance with `python3 reproduction/scripts/build_claim_evidence_provenance.py --map reproduction/configs/claim-evidence-map.json --output reproduction/results/raw/claim-evidence-provenance.json`, then run `python3 reproduction/scripts/verify_report_artifacts.py`. `figures/README.md` records expected hashes and the successful byte-for-byte regeneration check for the available local diagnostic plot; no unavailable numbered paper figure is presented as regenerated.

## 9. Smallest missing inputs/resources that unlock exact work

1. Official implementation/config/raw-result release or author verification of the candidate repository.
2. Exact Azure/BurstGPT file, 72-second offsets, scaling algorithm and request-to-context mapping.
3. English-translated DuReader artifact, sample IDs, prompts, splits, decoding/EOS and metric versions.
4. Exact default/accuracy/performance controller configs. LLM-PQ/PyramidKV plans would be needed only for the objective-added comparisons, which are absent from target v2.
5. Exact model/tokenizer/AWQ revisions for all four models.
6. Paper-equivalent L4/A100 hosts and sufficient pinned-memory allowance for all variants.

## 10. Current conclusion

The investigation supports MorphServe's **mechanical feasibility** under a labeled independent reconstruction: real same-address W4 replacement, physical KV reclamation, active state preservation, event-safe recovery, arbitrary-region mapping fallback, conditioned LIS, and coordinated controller actions all have executed evidence.

It does **not** support the paper's headline performance/quality claims or exact result tables. Overall reproduction status remains **partial / exact reproduction blocked**, with several candidate-code negative findings and no manufactured agreement. This goal closes current-revision implementation verification and the first common-engine modified-condition workload comparison: real transactional execution, two-request ownership/recovery, full-layer asynchronous W4/FP16 activity overlap, corrected synthetic replay, and six 1,024/512-token BurstGPT/Azure runs pass their declared gates. The remaining blockers are paper-facing: exact author implementation/configuration, models, task artifacts, trace boundaries/mapping, controller settings, and paper-equivalent hardware. This is not reproduction of the paper's headline results.
