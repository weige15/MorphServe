# Prompt-to-artifact completion audit

**Audit state:** current-revision GPU closure audit
**Audit date:** 2026-09-17  
**Decision:** all four requested current-revision full-model GPU verification surfaces pass their frozen modified-condition gates. The package remains a **partial / exact-condition-blocked paper reproduction**, not confirmation of the paper's headline results.

## Objective restated as checkable deliverables

The terminal package must contain: (1) a complete paper/source/claim map; (2) immutable, correctly attributed upstream artifacts; (3) a recorded environment/resource audit; (4) a runnable SwiftLLM-based implementation of real packed AWQ swapping, physical non-contiguous KV resizing, controller coordination, and exact conditioned-MDS LIS; (5) executed correctness, lifetime, failure, active-state, and overlap tests; (6) exact or honestly modified-condition experiments for the prioritized paper claims using scheduled arrivals and complete accounting; (7) raw inputs/logs/configs/metrics plus regenerating analysis; and (8) a final claim-by-claim `REPORT.md` that separates implementation fidelity, numerical agreement, negative findings, blockers, and uncertainty. Completion requires either all feasible priority work or substantiated blocked-stop evidence—not merely passing unit tests.

For this continuation goal, the concrete deliverables are: (1) real transactional executor GPU artifacts; (2) current atomic multi-request reclaimed-KV ownership/recovery artifacts; (3) current full-layer W4/FP16 asynchronous timing plus raw CUDA activity overlap artifacts; and (4) corrected synthetic GPU replay artifacts with independent scheduled arrivals and complete accounting. Each must pass its protocol gates, preserve prior attempts, record command/source/GPU provenance, and be reflected in the report and claim map.

| Goal surface | Required command | Current evidence | Gate result |
|---|---|---|---|
| Real executor | `CUDA_VISIBLE_DEVICES=0 reproduction/scripts/run_real_executor_pilot.sh` | `experiments/real-executor/results/metrics.json`, logs, source/GPU/manifest sidecars | PASS |
| Multi-request ownership | `CUDA_VISIBLE_DEVICES=0 reproduction/scripts/run_multirequest_ownership.sh` | `experiments/multirequest-ownership/results/metrics.json`, logs, source/GPU/manifest sidecars | PASS |
| Full async overlap | `CUDA_VISIBLE_DEVICES=0 reproduction/scripts/run_async_full_model_overlap.sh` | `experiments/async-layer-transfer/full-model-results/metrics.json`, `cuda-activity-trace.json`, logs/sidecars | PASS |
| Synthetic replay | `CUDA_VISIBLE_DEVICES=0 reproduction/scripts/run_synthetic_gpu_replay.sh` | `experiments/synthetic-gpu-replay/results/{raw.jsonl,summary.json,run-metadata.json}`, logs/sidecars | PASS |

Legend: **PASS**, **PARTIAL**, **BLOCKED**, **MISSING**, **NEGATIVE**.

## A. Source truth and deliverables

| ID | Explicit requirement | Status | Inspected evidence / gap |
|---|---|---|---|
| A1 | Read complete supplied conference PDF, Sections 4–6, Algorithm 1, figures/tables/appendices | PASS | `docs/paper-evidence-brief.md`; page-render audit. PDF has Figures 1–7 and Tables 1–8. |
| A2 | Resolve objective references to Tables 1–9 | NEGATIVE | No Table 9 exists in the supplied 19-page PDF; objective's stated Table-4 target is PDF Table 2. Recorded in `REPORT.md` and claim register. |
| A3 | Source-to-implementation map | PASS | `docs/source-map.md` includes paper location, source mapping and unresolved semantics. |
| A4 | Experiment/claim register | PASS | `docs/claim-register.md`, H1–H30. |
| A5 | Visually verify ambiguous equations/tables/footnotes | PASS | `docs/paper-evidence-brief.md`; explicit cosine equations take precedence over contradictory prose. |
| A6 | Keep paper values separate from measurements | PASS | `configs/paper-reference-values.json`; measured metrics remain under experiment directories. |
| A7 | Runnable reproduction workspace and commands | PARTIAL | `README.md`, `doc/onboarding.md`, per-experiment `commands.txt`; exact headline workspace cannot be runnable without missing assets. |
| A8 | Raw artifacts, configs, profiles, tests, plots | PARTIAL | Raw logs/configs/profiles/tests exist. A byte-identically regenerating local transfer diagnostic exists under `figures/`; no headline matrix plots exist because no valid headline runs. |
| A9 | Final claim-by-claim report | PASS for bounded investigation | `REPORT.md` covers H1–H30, current and historical evidence, negative findings, blockers, restart commands and limitations. Missing full-model runs are classified as blocked rather than silently omitted. |

## B. Provenance, authorization and environment

| ID | Requirement | Status | Inspected evidence / gap |
|---|---|---|---|
| B1 | Recheck official repo branches/tags/releases/artifacts | PASS/NEGATIVE | 2026-09-17 recheck at `results/raw/official-repo-recheck-2026-09-17.txt`: one README-only branch at `1c42999...`, no tags/releases. |
| B2 | Prefer verifiable author implementation; do not call README code | PASS | `docs/author-artifact-audit.md`; official repo never presented as implementation. |
| B3 | Attribute candidate source cautiously | PASS | `MorphServe/MorphServe@85c4fbf...` labeled candidate project-account artifact, authorship unresolved. |
| B4 | Pin SwiftLLM and preserve original artifacts | PASS | `interestingLSY/swiftLLM@682cf9a...`; immutable vendor manifests pass. Repairs isolated under `runtime/`. |
| B5 | Record GPU/RAM/memlock/PCIe/software/model/data/disk | PASS | `results/raw/environment.json`, trace hashes and memory feasibility JSON. |
| B6 | Isolated environment and lock | PASS | `reproduction/.venv`, `configs/fp16-requirements-lock.txt`, per-run install logs. |
| B7 | No purchase/quota/access bypass/credential exposure/publication | PASS | No such operation in logs; Git work remains local/ahead of origin. |
| B8 | Track agent and GPU budgets separately | PASS with caveat | `results/raw/resource-usage-summary.json`: 818.287 s saved GPU-runner process wall across 26 records; explicitly not kernel/exclusive GPU time. Harness agent time is separate in state. |
| B9 | Full all-variant pinning feasibility | BLOCKED | Requires 17,585,668,096 pinned bytes vs 16,844,414,976 memlock; 741,253,120-byte shortfall before overhead. |

## C. Implementation constraints

| ID | Requirement | Status | Inspected evidence / gap |
|---|---|---|---|
| C1 | Extend SwiftLLM with monitor/controller/executor while preserving scheduling | PARTIAL | `runtime/morphserve/{controller,integration,real_executor}.py`; FCFS tests pass. Independent reconstruction, not author code. |
| C2 | Real weight-only AWQ INT4, not fake quantization | PASS (modified) | 224 AutoAWQ GEMM modules and packed qweight/qzeros/scales; `experiments/static-autoawq/`. |
| C3 | Account for quantization metadata | PASS | One layer uses 113,328,128 bytes including norms/qweight/qzeros/scales. |
| C4 | Contiguous pinned FP16/W4 variants | PARTIAL | Tested per-layer pinned buffers; full 32-layer simultaneous setup blocked by memlock. |
| C5 | Preallocate GPU regions and preserve same address | PASS (modified) | One-layer exact address/storage/allocator tests; `experiments/autoawq-layer-switch/`. |
| C6 | Warm/precompile applicable GEMM before timing | PASS for requested pilots | Full async warms FP16/W4/prefill/cached-decode before three timed repeats; synthetic replay records separate initialization and warmup exclusion. |
| C7 | Real asynchronous copy, separate morph/decode streams | PASS for modified-condition full-model gate | Independent persistent-stream copier passes seam tests; full-layer run records nonblocking enqueue, distinct streams 17/7 and raw H2D/kernel intersections for W4 and FP16. |
| C8 | Explicit lifetime/event synchronization | PASS at seams | Uncoordinated race corrupts 1,530 bytes; C++ region barrier and Python last-forward/layer-ready barriers pass. |
| C9 | No whole-model reload/per-request routing/fake W4 | PASS for tested paths | Same base region and packed kernels used. |
| C10 | Physical KV blocks in reclaimed weight bytes | PASS for requested current integration | Physical reclaimed storage and zero allocator delta pass; current real executor and ownership runs use three 615-block groups, explicit regions and ownership-aware recovery; invalid batches reject before acquisition. |
| C11 | Non-contiguous mapping and custom Triton lookup | PASS after repair | Vendor fixed-stride corruption found; explicit-region store/attention matches dense oracle error 0. |
| C12 | Shrink safely before FP16 restore | PASS in current CPU transactions / historical GPU evidence | Current transaction tests cover all-before-mutation shrink and compensation; occupied-group refusal/LIFO GPU evidence predates atomic repair. |
| C13 | Preserve KV, request progress, no re-prefill/eviction/restart | PASS for bounded pilot | Active FP16→W4×4→FP16 same-history test and exact occupied-block migration. |
| C14 | Measure ordinary scheduler preemption separately | PARTIAL | Historical ownership pilot shows unchanged FCFS/swapped queues but has no cumulative preemption counter or decode scheduler; preemptions are recorded as not measured rather than zero. Candidate source accounting is itself disconnected. |
| C15 | Controller signals: memory/queue/throughput/TTFT/TPOT | PASS in reconstructed monitor | CPU tests cover all signals. |
| C16 | Smoothing, persistence, coordinated action/recovery | PASS as reconstruction | Frozen choices and tests; settings not author-recovered. |
| C17 | Default/performance/accuracy modes | PASS as reconstruction | Exposed and tested action sizes/maxima; not exact author modes. |
| C18 | Recover exact controller configs or label/freeze choices | PARTIAL/BLOCKED | Choices frozen/labeled; exact settings absent from all sources. |
| C19 | Exact LIS equations and conditioned MDS argmax | PASS | `runtime/morphserve/profiling.py`; unit tests and real profiles. |
| C20 | WikiText-2 length 2048 and retain local metrics | PASS for tested subset | Real 8-layer profile saves LTS/LRS/all 36 conditioned MDS values. |
| C21 | Full 32-layer final order and reuse | BLOCKED locally | Full simultaneous pinned variants exceed memlock; 8-layer order only. |
| C22 | Record sample/seed/reduction/tie/indexing | PASS | Profile JSON/protocol records first 2,048 train tokens, flattened float32, low-index ties, zero-based indices. |
| C23 | Front-to-Back default for unprofiled models | PASS in documented controller fallback | Source map/controller tests; real pilot uses profiled order. |
| C24 | Record paper inconsistencies; don't assume FP16 erases history | PASS | Source map and active same-history protocol explicitly cover both issues. |

## D. Correctness and verification surface

| ID | Requirement | Status | Inspected evidence / gap |
|---|---|---|---|
| D1 | No-morph FP16 test | PASS (modified) | Top-1/top-5 match Transformers, rel-L2 0.0020401, 291/291 exact tensors. |
| D2 | Fixed-W4 test | MIXED | Real packed path works/top-k stable; repeat logits not bit-exact because split-K atomic accumulation (rel-L2 ≤0.00493). |
| D3 | LIS argmax/conditioned-MDS test | PASS | Four unit tests plus 36 real conditioned evaluations. |
| D4 | Mixed-precision numerical test | PASS (bounded) | Active W4 steps same-history rel-L2 0.00058–0.00088, top-1 match. |
| D5 | KV address/content preservation | PASS for requested current integration | Synthetic/dense-oracle/active migration plus current two-request GPU rows, counts, sentinels and final recovery all pass. |
| D6 | Expansion/shrink/restoration | PASS for requested current integration | Current executor passes injected partial-expansion rollback and 4→1,849→4 LIFO recovery; ownership passes occupied refusal then post-release recovery; async restores exact FP16 bytes. |
| D7 | Repeated adaptation during prefill and decode | PASS for requested async surface | Full async runs three W4/FP16 timed repeats after warmup; this remains one-layer modified-condition verification, not serving-scale evidence. |
| D8 | Allocation failure and rollback | PASS for requested current integration | Current real executor records injected expansion failure with partial acquisition and exact rollback; current CPU tests additionally cover poisoning and no-mutation preflight. |
| D9 | Race/lifetime hazard | PASS/NEGATIVE+REPAIR | Corruption reproduced; event-safe repair reaches zero corruption. |
| D10 | Oscillating pressure | PASS at controller seam | CPU controller persistence test; real multi-request oscillating serving not yet timed. |
| D11 | Same-precision-history reference | PASS for requested async surface | All six full-model timed rows use same-history comparisons and pass top-1/relative-L2 gates. |
| D12 | Inspect actual CUDA overlap | PASS for requested full-model gate | Separate raw Chrome trace proves size-matched H2D/kernel intersections on different streams: W4 288.290 μs and FP16 290.210 μs. The parser repair ignores mirrored GPU annotations without weakening size/stream/intersection checks. |
| D13 | CPU/tiny/simulation only supporting | PASS in classification | Report does not promote them to headline evidence. |

## E. Exact experimental setup and data

| ID | Requirement | Status | Inspected evidence / gap |
|---|---|---|---|
| E1 | Exact Vicuna 7B v1.5 | BLOCKED | Checkpoint unavailable locally; no authorized cached revision. |
| E2 | Exact Llama 2 7B | BLOCKED | Gated checkpoint unavailable locally. |
| E3 | Exact Llama 3 8B revision | BLOCKED | Paper revision absent; local Llama 3.1 8B is labeled deviation. |
| E4 | Exact CodeLlama 34B | BLOCKED | Checkpoint/hardware unavailable. |
| E5 | Paper L4/A100 hardware/RAM | BLOCKED | Local 7× RTX 3090/125 GiB differs materially. |
| E6 | Exact 512/256 and 1024/512 lengths | MISSING headline runs | Short bounded correctness inputs only. |
| E7 | Primary Azure/BurstGPT files | PARTIAL | Azure Code/Conversation and pre-paper BurstGPT v1.1 recovered; Figure 1a strongly resolves Azure Code. |
| E8 | Exact 72-s offsets | PARTIAL/BLOCKED | Section 5 ties evaluation to Figure 1; shape matching uniquely ranks starts 1073 and 1,781,278, frozen before outcomes. Sub-second boundaries/aggregation remain approximate. |
| E9 | Exact 4.75×/1.75× operation | BLOCKED exact / PASS reconstructed | Systematic index thinning is frozen and deterministic (94/123 requests), but author operation/seed remain unknown. |
| E10 | Context sampling/mapping/seed | BLOCKED | No artifact. |
| E11 | Scheduled arrivals independent of completion | PASS for requested synthetic GPU replay | Actual submit times are 0.000083, 0.010933 and 0.021131 s; span 21.049 ms, with independent IDs and no wait-for-completion behavior. |
| E12 | Account all requests through completion/timeout | PASS at replay seam | Success/error/timeout/partial-token records and summary regeneration. |
| E13 | 2-second TTFT SLO | MISSING actual trace run | Recorded target only. |
| E14 | GovReport preprocessing/prompts/metric | BLOCKED | Public HF/LongBench revisions pinned; exact corpus variant/sample IDs/templates/versions absent. `docs/task-artifact-audit.md` |
| E15 | QMSum preprocessing/prompts/F1/ROUGE-L scopes | BLOCKED | Public repo pinned; exact split/rows/query prompt/settings absent. |
| E16 | English-translated DuReader | BLOCKED/NEGATIVE | Linked pinned tree has 303 paths, no English/translation artifact and no releases despite paper saying it is hosted there. |
| E17 | Multi-News preprocessing/prompts/metric | BLOCKED | Public repo pinned; exact split/rows/prompt/reference formatting absent. |
| E18 | Figure 4 model×trace×task matrix | MISSING/BLOCKED | No exact input/config set. |
| E19 | Same-engine target-PDF baselines | PARTIAL/BLOCKED | PDF scope is FP16/static AWQ/MorphServe modes; static paths exist, author modes missing. Objective-added LLM-PQ/PyramidKV comparisons are absent from target v2. |

## F. Prioritized numerical claims and ablations

| ID | Requirement | Status | Evidence / gap |
|---|---|---|---|
| F1 | Prioritize feasible published Table 4/5 comparison | BLOCKED exact | Objective numbering conflicts with PDF; target PDF Table 2 needs Llama 2 + translated DuReader + Burst window, all unavailable. |
| F2 | Table-target references kept as references | PASS | `[5.5223,.1241,27.68]`, `[1.1686,.0735,25.55]`, `[1.2420,.1064,27.33]` only in reference/report fields. |
| F3 | Headline 92.45% SLO reduction | BLOCKED | No exact matrix/raw denominator. |
| F4 | Mode-specific TTFT improvements | BLOCKED | No exact modes/workloads/hardware. |
| F5 | Objective-added LLM-PQ quality-gap closure | NOT A TARGET-PDF CLAIM | LLM-PQ and 41.3%/82.3% do not occur in supplied PDF/LaTeX. |
| F6 | Objective-added PyramidKV TTFT comparison | NOT A TARGET-PDF RESULT | PyramidKV is related-work citation only; 1.73×/2.4× do not occur. |
| F7 | Figures 5–7 | PARTIAL/MISSING | Physical capacity behavior shown; no exact 72-s capacity plot, saturation RPS or valid P99 TPOT. |
| F8 | Tables 2–8 ablations | PARTIAL/BLOCKED | LIS algorithm/order subset measured; exact model/task/perplexity tables not reproduced. |
| F9 | Table 1 BookSum 6K/2K every schedule | BLOCKED | Exact BookSum sample/prompt/decoding and memory resources absent. |
| F10 | 4/16/6 ms transfer and hidden stall | NEGATIVE/PARTIAL | Three isolated async copies: W4 median 15.214 ms, FP16 57.894 ms; no valid hidden-stall timeline yet. |
| F11 | Sub-15-minute 32-layer profiling | PARTIAL/BLOCKED | Eight-layer inner 58.14 s; full setup memlock blocked. |
| F12 | CodeLlama 48-layer endpoint | BLOCKED | Model unavailable. |

## G. Measurement and artifact integrity

| ID | Requirement | Status | Evidence / gap |
|---|---|---|---|
| G1 | Request IDs, inputs/references, generated text/counts | PASS for requested synthetic GPU replay | `results/raw.jsonl` contains all three IDs, prompts, generated text/token IDs and exact 3-token counts. |
| G2 | Scheduled/actual/first/completion/token timestamps | PASS at replay seam | Same. |
| G3 | Queue/errors/timeouts/preemptions/precision/KV occupancy | PARTIAL | Queue/errors/timeouts/precision/KV occupancy are saved. Bounded adapters lack cumulative scheduler preemption counters and now record that field as not measured; no headline workload. |
| G4 | Separate init/JIT/warmup | PASS for requested pilots | Async raw run follows warm FP16/W4/prefill/cached-decode setup; replay metadata records 13.303-s init and 3.622-s warmup excluded from timed origin. |
| G5 | Verify percentile definitions/units/denominators | PASS at replay seam | Hyndman-Fan type 7 explicitly implemented/tested. Headline denominators absent. |
| G6 | Absolute/relative loss and gap closure distinct | PASS in report definitions; no exact quality run | No manufactured aggregate. |
| G7 | Repeat timings ≥3 when budget permits | PASS for requested async surface | Current async output has three W4 and three FP16 timed rows; replay is accounting, not a repeat-timing claim. |
| G8 | Predeclare tolerances before target inspection | PARTIAL | W4 repeat envelope derived independently; many exact experiments never reached. |
| G9 | Preserve failures/negative results | PASS | Multiple numbered attempts, SIGSEGV/race/mapping/JIT/OOM evidence retained. |
| G10 | Regenerate plots/tables from raw | PARTIAL | Summaries and verifier check saved JSON. `figures/gen_fig_transfer_diagnostics.py` regenerates CSV/PDF/PNG byte-identically from raw attempt-1 metrics and fails closed if the source run is not rejected; unavailable numbered-paper plots remain absent. |
| G11 | Every claimed result links command/config/raw/comparison | PASS after current closure update | `configs/claim-evidence-map.json` maps H1–H30 to current and historical evidence; regenerated provenance hashes/Git-audits every linked artifact. Current runners write source-revision/status and GPU/manifest sidecars; historical caveats remain explicit. |
| G12 | Tests execute real work, not canned success | PASS for inspected tests | CUDA tests mutate/compare actual storage; CPU tests compute policy/profile/replay behavior. |

## H. Iteration/checkpoint and finalization policy

| ID | Requirement | Status | Evidence / gap |
|---|---|---|---|
| H1 | Persisted checkpoints | PASS | `research-state.yaml`: materials, reconstruction, correctness, current-revision GPU closure and final audit are recorded as complete under modified conditions. |
| H2 | Record hypothesis/change/command/outcome/cost/next step | PASS for meaningful attempts | `research-log.md`, protocols, attempt notes, command/wall-time files. |
| H3 | Fix correctness before performance | PASS | Mapping/race/rollback fixes preceded timing work. |
| H4 | Retry only with changed hypothesis/transient reason | PASS | JIT warmup fix and recorded external OOM; no blind loop. |
| H5 | Continue other work when exact claims blocked | PASS | Independent profiling, KV, controller, replay and async work completed. |
| H6 | Stop only when no meaningful permitted verification remains | PASS for this goal | The one remaining implementation-level GPU goal is complete: all four full-model runners pass. Remaining work is paper-facing exact reproduction blocked by missing author/config/task/trace/model/hardware inputs, not by unattempted current-revision GPU verification. |
| H7 | Final report classifications and implementation/result separation | PASS | `REPORT.md` is final for this bounded resource state and clearly marks optional future reruns as missing evidence, not completed results. |
| H8 | Full prompt-to-artifact completion audit | PASS as an audit artifact, outcome NOT COMPLETE | This document maps explicit requirements and rejects proxy completion. |
| H9 | Call completion mechanism only after full audit | PASS | This audit checks all four requested commands, gates, raw artifacts, source/GPU/manifest provenance and report/map updates. It supports current-revision implementation closure only, not full paper reproduction. |

## Verifier coverage audit

`python3 reproduction/scripts/verify_report_artifacts.py` verifies file presence/JSON parseability, all 30 claim-evidence-map entries and their paths, the four current GPU gates, two separately labeled historical pre-atomic gates and their classification files, the expected static-W4 nondeterminism, memlock blocker, small async-copy gate, raw CUDA overlap details, replay accounting, inferred-window identities, deterministic trace-manifest hashes, task-source metadata hashes, and key report phrases. It **does not** establish author confirmation, exact scaling/context mapping, exact models, datasets, baselines, 32-layer profiling, main workload matrix, headline aggregates, or every prose claim. It is therefore a consistency check, not paper-completion proof.

The vendor manifest proves only that the candidate snapshot was not modified. Unit-test green status proves only the named seams. Neither is accepted as completion evidence for the paper-level objective.

## Current audit conclusion and optional restart gates

The current-revision implementation-verification deliverable is achieved: all four required runners pass, prior failed attempts remain preserved, and the report/claim map/provenance are updated from the raw artifacts. The result is still **partial / exact reproduction blocked** for the paper. No current GPU restart gate remains for this goal.

Exact headline work remains blocked pending exact sub-second boundaries plus scaling/context map, task artifacts/prompts/metrics, model revisions, target-PDF controller configs, and paper-equivalent hardware. The passing commands and outputs must not be relabeled as paper latency, quality, throughput or exact-table reproduction.
