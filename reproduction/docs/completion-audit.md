# Prompt-to-artifact completion audit

**Audit state:** evolving; **goal not complete**  
**Audit date:** 2026-09-17  
**Decision:** substantial partial reconstruction, but prioritized feasible GPU gates and many exact-condition claims remain incomplete or blocked.

## Objective restated as checkable deliverables

The terminal package must contain: (1) a complete paper/source/claim map; (2) immutable, correctly attributed upstream artifacts; (3) a recorded environment/resource audit; (4) a runnable SwiftLLM-based implementation of real packed AWQ swapping, physical non-contiguous KV resizing, controller coordination, and exact conditioned-MDS LIS; (5) executed correctness, lifetime, failure, active-state, and overlap tests; (6) exact or honestly modified-condition experiments for the prioritized paper claims using scheduled arrivals and complete accounting; (7) raw inputs/logs/configs/metrics plus regenerating analysis; and (8) a final claim-by-claim `REPORT.md` that separates implementation fidelity, numerical agreement, negative findings, blockers, and uncertainty. Completion requires either all feasible priority work or substantiated blocked-stop evidence—not merely passing unit tests.

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
| A9 | Final claim-by-claim report | PARTIAL | `REPORT.md` covers H1–H30 and current evidence, but remains explicitly active; full async/replay reruns are pending. |

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
| B8 | Track agent and GPU budgets separately | PASS with caveat | `results/raw/resource-usage-summary.json`: 807.058 s saved GPU-runner process wall across 25 records; explicitly not kernel/exclusive GPU time. Harness agent time is separate in state. |
| B9 | Full all-variant pinning feasibility | BLOCKED | Requires 17,585,668,096 pinned bytes vs 16,844,414,976 memlock; 741,253,120-byte shortfall before overhead. |

## C. Implementation constraints

| ID | Requirement | Status | Inspected evidence / gap |
|---|---|---|---|
| C1 | Extend SwiftLLM with monitor/controller/executor while preserving scheduling | PARTIAL | `runtime/morphserve/{controller,integration,real_executor}.py`; FCFS tests pass. Independent reconstruction, not author code. |
| C2 | Real weight-only AWQ INT4, not fake quantization | PASS (modified) | 224 AutoAWQ GEMM modules and packed qweight/qzeros/scales; `experiments/static-autoawq/`. |
| C3 | Account for quantization metadata | PASS | One layer uses 113,328,128 bytes including norms/qweight/qzeros/scales. |
| C4 | Contiguous pinned FP16/W4 variants | PARTIAL | Tested per-layer pinned buffers; full 32-layer simultaneous setup blocked by memlock. |
| C5 | Preallocate GPU regions and preserve same address | PASS (modified) | One-layer exact address/storage/allocator tests; `experiments/autoawq-layer-switch/`. |
| C6 | Warm/precompile applicable GEMM before timing | PARTIAL | Static/switch pilots warm paths; full-async and synthetic retry code now warms real cached decode, pending rerun. |
| C7 | Real asynchronous copy, separate morph/decode streams | PARTIAL | Candidate is blocking. Independent persistent-stream copier passes small CUDA seam; full-layer run pending. |
| C8 | Explicit lifetime/event synchronization | PASS at seams | Uncoordinated race corrupts 1,530 bytes; C++ region barrier and Python last-forward/layer-ready barriers pass. |
| C9 | No whole-model reload/per-request routing/fake W4 | PASS for tested paths | Same base region and packed kernels used. |
| C10 | Physical KV blocks in reclaimed weight bytes | PASS (modified) | 615 real blocks/layer in real-executor pilot; zero allocator delta. |
| C11 | Non-contiguous mapping and custom Triton lookup | PASS after repair | Vendor fixed-stride corruption found; explicit-region store/attention matches dense oracle error 0. |
| C12 | Shrink safely before FP16 restore | PASS (modified) | Occupied-group refusal and LIFO recovery in two-request pilot. |
| C13 | Preserve KV, request progress, no re-prefill/eviction/restart | PASS for bounded pilot | Active FP16→W4×4→FP16 same-history test and exact occupied-block migration. |
| C14 | Measure ordinary scheduler preemption separately | PARTIAL | Ownership pilot proves unchanged FCFS/swapped queues but has no cumulative preemption counter and no decode scheduler; it now records preemptions as not measured rather than zero. |
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
| D5 | KV address/content preservation | PASS | Synthetic, dense-oracle, active migration, two-request sentinels. |
| D6 | Expansion/shrink/restoration | PASS at tested seams; async real-GPU rerun pending | Historical 4→1,849→4 evidence plus all-before-mutation shrink, partial morph/restore rollback, atomic post-shrink compensation, fail-closed poisoning and second-expansion barrier tests. |
| D7 | Repeated adaptation during prefill and decode | PARTIAL | Repeated synthetic cycles and four active decode steps; full async alternating 3× run pending. |
| D8 | Allocation failure and rollback | PASS | Injected expansion failure restores exact state/queues. |
| D9 | Race/lifetime hazard | PASS/NEGATIVE+REPAIR | Corruption reproduced; event-safe repair reaches zero corruption. |
| D10 | Oscillating pressure | PASS at controller seam | CPU controller persistence test; real multi-request oscillating serving not yet timed. |
| D11 | Same-precision-history reference | PASS where used | Active-KV protocol; full async pilot also designed this way but pending. |
| D12 | Inspect actual CUDA overlap | MISSING valid full-model | Attempt 1 is rejected. Retry demotes event intervals to supporting bounds, exports a raw CUDA activity trace, and requires size-matched H2D/kernel activity intersection on different streams; three fail-closed parser tests pass, real trace pending. |
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
| E11 | Scheduled arrivals independent of completion | PASS at replay seam | Four tests; real corrected GPU replay pending. |
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
| G1 | Request IDs, inputs/references, generated text/counts | PASS at replay seam; pending real rerun | `runtime/morphserve/replay.py`, tests, synthetic attempt raw JSONL. |
| G2 | Scheduled/actual/first/completion/token timestamps | PASS at replay seam | Same. |
| G3 | Queue/errors/timeouts/preemptions/precision/KV occupancy | PARTIAL | Queue/errors/timeouts/precision/KV occupancy are saved. Bounded adapters lack cumulative scheduler preemption counters and now record that field as not measured; no headline workload. |
| G4 | Separate init/JIT/warmup | PASS in diagnosis, pending corrected result | Attempt 1 preserved and rejected for JIT contamination; corrected run OOM-blocked. |
| G5 | Verify percentile definitions/units/denominators | PASS at replay seam | Hyndman-Fan type 7 explicitly implemented/tested. Headline denominators absent. |
| G6 | Absolute/relative loss and gap closure distinct | PASS in report definitions; no exact quality run | No manufactured aggregate. |
| G7 | Repeat timings ≥3 when budget permits | PARTIAL | Frozen async pilot requests 3; historical timings mostly one repeat. Current GPU resource prevents repeat. |
| G8 | Predeclare tolerances before target inspection | PARTIAL | W4 repeat envelope derived independently; many exact experiments never reached. |
| G9 | Preserve failures/negative results | PASS | Multiple numbered attempts, SIGSEGV/race/mapping/JIT/OOM evidence retained. |
| G10 | Regenerate plots/tables from raw | PARTIAL | Summaries and verifier check saved JSON. `figures/gen_fig_transfer_diagnostics.py` regenerates CSV/PDF/PNG byte-identically from raw attempt-1 metrics and fails closed if the source run is not rejected; unavailable numbered-paper plots remain absent. |
| G11 | Every claimed result links command/config/raw/comparison | PASS for current report; historical source-revision caveat | `configs/claim-evidence-map.json` maps H1–H30 to paper references, commands, frozen configs/protocols, raw artifacts, comparisons and limitations; verifier checks all paths. Older runners did not save the exact source revision, while pending runners now do. |
| G12 | Tests execute real work, not canned success | PASS for inspected tests | CUDA tests mutate/compare actual storage; CPU tests compute policy/profile/replay behavior. |

## H. Iteration/checkpoint and finalization policy

| ID | Requirement | Status | Evidence / gap |
|---|---|---|---|
| H1 | Five persisted checkpoints | PARTIAL | `research-state.yaml`: checkpoint 1 complete, checkpoint 2 active, 3–5 pending. |
| H2 | Record hypothesis/change/command/outcome/cost/next step | PASS for meaningful attempts | `research-log.md`, protocols, attempt notes, command/wall-time files. |
| H3 | Fix correctness before performance | PASS | Mapping/race/rollback fixes preceded timing work. |
| H4 | Retry only with changed hypothesis/transient reason | PASS | JIT warmup fix and recorded external OOM; no blind loop. |
| H5 | Continue other work when exact claims blocked | PASS | Independent profiling, KV, controller, replay and async work completed. |
| H6 | Stop only when no meaningful permitted verification remains | NOT MET | Full async/replay/executor reruns remain feasible when a GPU becomes free. |
| H7 | Final report classifications and implementation/result separation | PARTIAL | Current `REPORT.md` does this but remains active and must absorb pending reruns. |
| H8 | Full prompt-to-artifact completion audit | PASS as an audit artifact, outcome NOT COMPLETE | This document maps explicit requirements and rejects proxy completion. |
| H9 | Call completion mechanism only after full audit | PASS so far | Goal remains active; no completion call. |

## Verifier coverage audit

`python3 reproduction/scripts/verify_report_artifacts.py` verifies file presence/JSON parseability, all 30 claim-evidence-map entries and their paths, six positive experiment gates, the expected static-W4 nondeterminism, memlock blocker, small async-copy gate, inferred-window identities, deterministic trace-manifest hashes, task-source metadata hashes, and key report phrases. It **does not** establish author confirmation, exact scaling/context mapping, exact models, datasets, baselines, 32-layer profiling, full-layer overlap, main workload matrix, headline aggregates, or every prose claim. It is therefore a consistency check, not completion proof.

The vendor manifest proves only that the candidate snapshot was not modified. Unit-test green status proves only the named seams. Neither is accepted as completion evidence for the paper-level objective.

## Current audit conclusion and next gates

The objective is **not achieved**. The smallest currently feasible next gates are:

1. when one GPU has >17 GiB free, run `CUDA_VISIBLE_DEVICES=<free> reproduction/scripts/run_async_full_model_overlap.sh` and inspect all three repeats rather than accepting its exit code alone;
2. rerun `run_real_executor_pilot.sh` and `run_multirequest_ownership.sh` against the asynchronous executor;
3. rerun corrected `run_synthetic_gpu_replay.sh`, preserving complete timestamps and excluding warmup;
4. update `REPORT.md` and this audit from the resulting raw artifacts.

Exact headline work remains blocked pending exact sub-second boundaries plus scaling/context map, task artifacts/prompts/metrics, model revisions, target-PDF controller configs, and paper-equivalent hardware. These blockers do not excuse the three feasible GPU reruns above once uncontended capacity is available.
