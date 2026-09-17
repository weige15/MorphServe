# Research log

## 2026-09-16 — checkpoint 1: materials and environment audit

### Hypothesis

The supplied repository may contain only the paper, so exact author-code execution may be blocked; before reconstructing anything, search all public primary repositories and local prior work.

### Actions

- Inspected local Git history, remote, and all top-level files.
- Read and visually checked the full 19-page supplied PDF through a delegated paper audit; extracted the exact arXiv v2 source and all reference tables.
- Queried and preserved GitHub API state for `ds2-lab/MorphServe` and `MorphServe/MorphServe`.
- Pinned and archived SwiftLLM commit `682cf9a...`.
- Audited `/nfs/home/s314511048/precision-batching` to avoid repeating existing SwiftLLM/asset discovery.
- Captured hardware, software, PCIe, memlock, process, disk, and model-file hash evidence in `results/raw/environment.json`.

### Commands/evidence

See `docs/paper-evidence-brief.md`, `docs/author-artifact-audit.md`, `docs/source-map.md`, and `sources/remote-audit/`. Environment command is reproducible via `scripts/capture_environment.py`.

### Observed outcome

- Official `ds2-lab/MorphServe` remains README-only and explicitly promises a future release.
- An earlier `MorphServe/MorphServe` project-account repository contains a partial implementation closely aligned with the paper, but no primary author/lab link verifies ownership.
- The candidate artifact lacks profiler, experiments, configs, tests, controller modes, and exact inputs; it also has packaging/config defects and synchronous behavior around nominally asynchronous copies.
- The supplied PDF has Tables 1–8 only. There is no Table 9. The objective's Table 4 numeric target is actually PDF Table 2.
- Local hardware is 7× RTX 3090 24 GB with 125 GiB RAM, not the L4/A100 systems in the paper.
- Local Llama 3.1 8B base and AWQ W4 assets are available; exact paper model revisions are not.

### Resource cost

No GPU experiments. Network traffic was limited to public source/API downloads. Approximately 30 MiB of paper/source/code evidence was added. No paid resources or external uploads.

### Decision / next step

Treat checkpoint 1 as complete. Prioritize the candidate artifact's unmodified build/import viability, record negative evidence, and make any compatibility repair only in a clearly separate reconstruction tree. Do not start headline serving runs before correctness gates.

## 2026-09-16 — candidate artifact smoke, initial run

### Hypothesis

The candidate source will not install/import unmodified because of package naming and config plumbing, while its C++ source may still build.

### Command

```bash
reproduction/scripts/run_candidate_artifact_smoke.sh
```

### Observed outcome

The editable install, all documented Python package imports, and CLI-to-dataclass construction failed exactly at the predicted seams. The C++ extension compiled successfully. Direct extension import failed because `libc10.so` was not loaded; since the probe did not import PyTorch first, this is not yet a binary-incompatibility result. Both vendor-manifest checks passed. Full evidence is under `experiments/candidate-artifact-smoke/results/` and summarized in `analysis.md`.

### Resource cost and next step

No GPU work or model load. Run one changed-hypothesis follow-up that preloads PyTorch before importing the extension; do not retry any unchanged packaging/config failure.

### Follow-up outcome

The frozen follow-up `import torch; import swiftllm_c` succeeded against the freshly built extension and exposed 13 bindings, including layer registration/replacement and reclaimed-KV operations. This narrows the failure: the C++ source is buildable/importable, while the released Python package and CLI remain non-runnable. The repeat runner's venv-creation step returned 2 because the venv already existed; dependency and build probes still ran, and this harness idempotence issue was fixed without changing the observed mechanism result.

## 2026-09-16 — Algorithm 1 conditioned-MDS selection

### Hypothesis and seam

A three-layer worked example with state-dependent MDS will distinguish the paper's greedy conditioned algorithm from a one-time/static MDS ranking. The public seam is `rank_layers(lts, lrs, mds)` plus canonical `save_profile`.

### Command and outcome

```bash
PYTHONPATH="$PWD/reproduction/runtime" python3 -m unittest reproduction.tests.test_profiling -v
```

The retained red/green loop ends with 4/4 passing tests. The conditioned algorithm makes six MDS calls, selects `[0,2,1]`, while a static-MDS comparator selects `[0,1,2]`. Literal LIS weights, `argmax`, deterministic reconstructed tie-breaking, validation, and full-history JSON are covered.

### Resource cost and next step

CPU-only and zero GPU experiment time. Next, freeze model representation/calibration choices and implement real WikiText-2 metric collection; selection-logic success alone is not a model-profile reproduction.

## 2026-09-16 — candidate C++ memory-manager GPU check

### Hypothesis

A synthetic FP16-sized owner and packed replacement will demonstrate whether the candidate extension converts reclaimed weight bytes into actual bounded KV storage rather than counters.

### Command and observed outcome

```bash
CUDA_VISIBLE_DEVICES=0 reproduction/scripts/run_candidate_memory_manager_test.sh
```

Both tests completed their assertions. The packed view reused the owner base; a 3,072-byte tail exposed 12 correctly bounded K/V blocks; writes changed the owner; allocator delta was zero; undersized allocation failed; restoration recovered original bytes. After unittest printed `OK`, however, Python segfaulted and returned 139.

Targeted controls plus `cuda-gdb` localized the crash to global destruction of static `pybind11::object` tensor metadata after Python finalization, not to CUDA storage. The diagnosis and minimal repair are in `doc/debug-report.md`. This is an evidence-backed negative result for the released extension's process lifetime.

### Resource cost and next step

Sub-second synthetic kernels ran on one RTX 3090 across isolated probes; no model load. Implement the diagnosed native-metadata repair only in a separate reconstruction source, then rerun the identical gate. Do not proceed to full-model or timing claims while exit safety fails.

## 2026-09-16 — native metadata lifetime repair

### Hypothesis and scoped change

Parsing registration dictionaries into native `TensorInfo` records, while leaving all copy/address/KV behavior unchanged, should eliminate the confirmed shutdown crash.

### Command and outcome

```bash
CUDA_VISIBLE_DEVICES=0 \
MORPHSERVE_CSRC="$PWD/reproduction/runtime/candidate-csrc" \
MORPHSERVE_TEST_RESULT_DIR="$PWD/reproduction/experiments/candidate-memory-manager-repair/results" \
reproduction/scripts/run_candidate_memory_manager_test.sh
```

The same two GPU tests passed and the process exited 0. The extension also imported after Torch preload with all 13 bindings. Address, 12-block capacity, writable owner storage, undersized-tail failure, same-base restore, and zero allocator delta were preserved. Vendor manifests remained unchanged.

### Resource cost and next step

Primary verification used 2.94 s process wall time on one RTX 3090, dominated by startup; CUDA-event operations were each under 1.5 ms for tiny buffers and are not paper-comparable. Next test the candidate Triton mapping across at least two non-contiguous reclaimed layer regions before model integration.

## 2026-09-16 — two-region Triton KV mapping

### Hypothesis

Candidate pointer arithmetic should resolve original cache plus two reclaimed tails when layers are contiguous/equal-stride and swapped back-to-front.

### Command and outcome

```bash
CUDA_VISIBLE_DEVICES=0 reproduction/scripts/run_candidate_kv_mapping_test.sh
```

The process exited 0. Prefill and decode writes landed exactly in virtual blocks `[0,2,14,15]`; packed prefixes remained intact; allocator delta was zero; and combined PagedAttention matched an independent 13-token dense PyTorch oracle with max absolute error 0.0.

### Interpretation and resource cost

This is positive modified-condition mechanism evidence, not proof of arbitrary non-contiguous allocation. The candidate derives group pointers by fixed layer-stride subtraction. First-call wall time was 8.06 s and PagedAttention printed 877 ms because Triton compiled on demand; those values are initialization evidence and excluded from timing claims. Next gates are occupied/in-flight shrink and repeated adaptation, then model integration.

## 2026-09-16 — in-flight restore and repeated adaptation

### Hypothesis

Candidate restore has no event barrier for reclaimed storage, so a delayed non-default-stream write can complete after FP16 restoration and corrupt weights. Explicit waiting should prevent corruption; synchronized cycles should remain stable.

### Command and outcome

```bash
CUDA_VISIBLE_DEVICES=0 reproduction/scripts/run_candidate_lifetime_test.sh
```

The counterexample was confirmed: restore returned while the writer event was incomplete, and the later write left 1,530 bytes different from original. The explicit-wait control had 0 corrupt bytes. Separately, five synchronized quantize/reclaim/write/release/restore cycles reused one base, restored exact bytes each time, and had zero allocator deltas.

### Interpretation and next step

The candidate's free-block check is insufficient for paper-style asynchronous overlap. Add explicit per-region CUDA lifetime events in the reconstruction before claiming safe restore or overlap. Full-model integration remains blocked until this correctness condition is enforced.

## 2026-09-16 — explicit reclaimed-region event barrier

### Hypothesis and scoped change

Recording a timing-disabled event on every stream using a reclaimed region and making restore wait on all recorded events should eliminate the confirmed race without changing address/capacity behavior.

### Command and outcome

```bash
CUDA_VISIBLE_DEVICES=0 \
MORPHSERVE_TEST_RESULT_DIR="$PWD/reproduction/experiments/candidate-lifetime-barrier/results" \
reproduction/scripts/run_candidate_lifetime_test.sh
```

The unrecorded sensitivity case still corrupted 1,530 bytes. In the recorded case the writer was unfinished before restore, restore waited ~103 ms, and final corruption was 0. Three event-protected immediate restore cycles had stable addresses, exact bytes, and zero allocator deltas. Prior memory-manager and two-region Triton regressions also passed.

### Scope and next step

The C++ primitive is correct in this modified synthetic condition. It does not help unless all Python/Triton call sites record use. Integrate that call at the model executor boundary, then normalize package/config/checkpoint loading and run no-morph FP16 parity.

## 2026-09-16 — FP16 parity attempt 1 setup failure

The first bounded run did not load a model. The runner's generic `${MODEL_PATH}` override collided with an inherited `MODEL_PATH=meta-llama/Meta-Llama-3.1-8B`, producing a nonexistent local path and an `HFValidationError`. Dependencies/extension built, but GPU experiment time was zero and no metrics were emitted. The changed retry will use `MORPHSERVE_MODEL_PATH` and an early `config.json` existence check. See `doc/debug-report-fp16-path.md`.

### Attempt 2

The corrected local snapshot loaded and the Transformers reference executed. Candidate import then failed because core model code imports constants from `swiftllm.utils`, which eagerly imports optional package `evaluate` used only by ROUGE evaluation. No candidate model loaded and no parity metric was computed. The changed retry will lazy-import `evaluate` inside `rouge_calculate`; this changes no serving or metric semantics.

### Attempt 3 result

The no-morph Llama 3.1 8B parity gate passed: top-1/top-5 exact, relative logit L2 0.0020401, max absolute error 0.0234375, and 291/291 loaded tensors exact. This is modified-condition numerical evidence with KV disabled. GPU 0 had another 6.9-GiB/100%-utilization process before and after, so all wall timings are excluded from performance claims. Candidate process-local peak was 16.17 GB.

## 2026-09-16 — real static AutoAWQ W4 baseline

### Hypothesis and command

The local AutoAWQ W4 G128 zero-point asset should prove real packed low-bit execution and exact repeated logits.

```bash
CUDA_VISIBLE_DEVICES=1 reproduction/scripts/run_static_autoawq_baseline.sh
```

### Outcome

The mechanism half was confirmed: 224 `WQLinear_GEMM` modules, INT32 qweight/qzeros, FP16 scales, and 3,625,975,808 decoder bytes including metadata. FP16 and W4 shared top-1 token 505; W4-vs-FP16 relative logit L2 was 0.35126.

The exact-repeat gate failed. Five repeats had stable top-1/top-5 but relative logit differences up to 0.00493 and max absolute differences up to 0.05859. Source inspection identifies eight-way split-K plus `tl.atomic_add` in AutoAWQ's small-input Triton path. This remains an evidence-backed negative for bit-exact repeatability, not a reason to replace the real fused path with fake quantization.

### Next step

Use the independently measured repeat envelope to predeclare tolerances for controlled mixed-precision history tests. Build an explicitly labeled AutoAWQ adapter; do not present it as execution of the candidate's mismatched llm-awq interface.

## 2026-09-16 — one-layer real AutoAWQ in-place switch

### Hypothesis and command

One layer's qweight/qzeros/scales can be packed into pinned CPU memory, copied into its existing FP16 GPU region, executed via `WQLinear_GEMM`, and restored without duplicate GPU allocation.

```bash
CUDA_VISIBLE_DEVICES=1 reproduction/scripts/run_autoawq_layer_switch.sh
```

### Outcome

All gates passed. Layer 31 used 113,328,128 packed bytes inside a 436,224,000-byte FP16 region, leaving 322,895,872 bytes (74.02%) physically reclaimable after metadata. Twenty-three packed tensors and eight restored FP16 tensors were exact; seven real low-bit modules ran with zero allocator delta. Mixed logits differed from FP16 (relative L2 0.04193), repeated W4 stayed within the frozen split-K envelope (0.00106), and restored FP16 logits were bit-exact.

Blocking operation walls were 15.67 ms W4 and 58.55 ms FP16 on RTX 3090, substantially slower than the paper's different-model/hardware 4/16 ms transfer examples and ~6 ms complete W4 swap. No overlap occurred, so timing claims are not reproduced.

### Next step

Extend the pilot to active KV/request state with the same precision history and reclaimed block attachment, then implement multiple-layer ordering/profiling. Keep this adapter labeled independent reconstruction.

## 2026-09-16 — active-KV same-history switch

### Hypothesis and command

An active request can use a reclaimed block during four W4 decode steps, migrate it safely, restore FP16, and continue without re-prefill while matching a reference with the same precision/token history.

```bash
CUDA_VISIBLE_DEVICES=1 reproduction/scripts/run_active_kv_switch.sh
```

Attempt 1 reached reference decode then failed on a test-harness inference-tensor reset outside `torch.inference_mode`; no system result. The changed retry fixed only reset/migration context.

Attempt 2 passed all gates. The fourth W4 decode allocated reclaimed virtual block 3 (615 tail slots available). W4 step relative logit L2 was 0.00058–0.00088 with matching top-1; reclaimed K/V relative L2 was 0.000095/0.000207; byte-exact migration retained block table `[0,1,2,3]`; final FP16 top-1 matched at relative L2 0.000354. No re-prefill, eviction, model reload, or scheduler restart occurred.

### Interpretation

This is strong modified-condition state-preservation evidence for one request/layer/block. Occupied-block migration is a reconstruction choice, and first-call multi-second logs are Triton JIT initialization, not steady-state latency.

## 2026-09-16 — real WikiText-2 conditioned LIS pilot

### Hypothesis and command

The tested Algorithm 1 core can drive real FP16/W4 metric collection for layers 29–31 on one exact 2,048-token WikiText-2 sequence.

```bash
CUDA_VISIBLE_DEVICES=1 reproduction/scripts/run_lis_real_pilot.sh
```

All gates passed. The profiler executed six candidate calls, retained LTS/LRS/MDS histories, selected `[29,30,31]`, restored 56 tensors and final FP16 logits exactly, and saved an immutable profile. Inner work took 14.83 s; runner wall with model/data load was 45.22 s. This is a bounded modified-condition pilot, not a full 32-layer order or under-15-minute reproduction.

## 2026-09-16 — expanded eight-layer real LIS

Attempt 1 completed all 36 model evaluations but the generic runner retained three-layer gate constants and exited 1; raw model results were preserved. After generalizing only the verifier counts, the identical frozen protocol passed and selected `[25,24,26,27,28,29,30,31]`. The shared tail kept `[29,30,31]`; 336 restores and final FP16 logits were exact. Inner work took 58.14 s, full wall 92.78 s.

A byte audit found all 32 decoder FP16+W4 variants require 17,585,668,096 pinned bytes versus a 16,844,414,976-byte memlock limit, exceeding it by 741,253,120 bytes before staging/runtime overhead. This blocks the paper-style simultaneous pinning condition locally; limits will not be raised autonomously.

## 2026-09-16 — reconstructed controller core

Because the paper gives only example 85% KV/100-ms queue thresholds, the protocol froze explicit non-author settings before evaluation: EMA 0.25; 3-sample pressure; 5-sample recovery; 1/2/4 layers per accuracy/default/performance action with maxima 4/8/16. Six tests pass for complete signal collection, persistence, hysteresis, mode bounds, oscillation resistance, coordinated KV commands, and invalid inputs. This is CPU policy-core evidence only; engine/GPU integration is next.

## 2026-09-16 — controller/executor integration seam

Five fake-executor tests pass. Pressure follows the frozen profile and copies W4 before KV expansion; expansion failure/exception rolls back FP16 in reverse order; recovery shrinks KV before restore and defers if unsafe; FCFS request order never changes. This validates command semantics/rollback only. Real GPU executor integration and concurrent allocation failures remain open.

## 2026-09-16 — arbitrary profiled-order KV counterexample

The real LIS prefix `[25,24,26]` was applied to a four-region synthetic layout. Candidate group 2 computed `layer25_base - 2*stride`, left registered layer 26 unchanged, and wrote the known K/V token into layer 23. The process exited 0, proving deterministic silent misrouting rather than a crash. Prior positive mapping evidence is narrowed to descending contiguous layers.

### Explicit-region repair

The independent runtime now masks/remaps one block table per region and repairs the candidate multi-kernel attention helper. The same `[25,24,26]` test writes layer 26, leaves layer 23 untouched, and matches a dense oracle with max error 0. The descending fused regression remains max-error 0. This restores correctness but adds launches and is not yet a performance-equivalent paper reproduction.

## 2026-09-16 — transactional real GPU executor

Attempt 1 completed all actions but used an incorrect free-block assertion for a no-request pool; raw results were preserved. The corrected run passed: injected first expansion failure rolled back layer 25/capacity/state; three accuracy-mode actions activated `[25,24,26]`, attached 615 blocks each (4→1,849), and selected explicit mapping; recovery shrank/restored `26→24→25`; final FP16 logits/capacity were exact and FCFS queues unchanged.

## 2026-09-16 — real two-request reclaimed ownership

Attempt 1 exposed an inference-mode context bug in executor shrink; after the scoped fix, request 0/1 allocated `[0,1,2,3,4]` and `[5]`, placing IDs 4/5 in reclaimed group 0. Recovery removed free groups 26/24, refused occupied layer 25 while preserving counts, rows, K/V sentinel, state and FCFS, then succeeded after real BlockManager frees. Final FP16/capacity were exact and FCFS/swapped queues were unchanged. The static pilot has no cumulative preemption counter, so ordinary scheduler preemptions were not measured; model decode concurrency and timed arrivals remain untested.

## 2026-09-16 — replay harness and primary trace recovery

Four replay tests pass for independent scheduled submissions, complete error/timeout/token accounting, explicit TTFT/TPOT and type-7 percentiles, and raw-summary regeneration. Primary Azure Code/Conversation and BurstGPT v1.1 files were recovered and hashed. Exact paper replay remains blocked: Azure file, both 72-s offsets, scaling operation and context mapping are absent. First/densest Burst windows contain 2/1,666 requests, proving that an unsourced window choice would dominate outcomes.

## 2026-09-17 — frozen synthetic GPU replay attempts

Attempt 1 passed accounting but prefill-only warmup leaked decode Triton JIT into timed TPOT; preserved as initialization evidence. The corrected warmup adds a cached decode step. Attempt 2 then encountered a transient external-GPU OOM before model load (10.32 GiB occupied; 12.95 GiB free versus 14.96-GiB model allocation). All GPUs became occupied, so no safe unchanged retry was possible without evicting others or changing the model. Retry remains queued for sufficient headroom.

## 2026-09-17 — independent asynchronous layer-copy seam

The candidate's synchronized-per-call copy path was retained, and a labeled independent path was added using a non-owning registered-region view, pinned copies on one persistent morph stream, prebuilt precision wrappers, a model-wide last-forward event, and a just-in-time wait before the affected layer. Five small CUDA tests pass: prior use was unfinished when 4 MiB enqueue returned (0.206 ms), copy took 0.578 ms, bytes/address were exact, model use observed the copy, partial expansion published a rollback barrier, redundant blocking-path events were suppressed, and malformed sources failed closed. A three-repeat full-layer same-history overlap pilot is frozen, but every GPU remains externally occupied and none has the >17 GiB headroom required for the 8B model. The official lab repository was rechecked and remains at README-only commit `1c42999...` with no tags/releases.

## 2026-09-17 — Figure 1 trace-window inference

Approximate Figure 1b token-volume shapes were digitized and matched against every dense source window without using serving outcomes. Figure 1a independently resolves Azure Code (zeros and ≈1.4M context tokens/min). Both request-count and token rankings select Azure relative second 1073 (443 requests in 72 s; context Pearson 0.8793, runner-up 0.5706) and BurstGPT v1.1 second 1,781,278 (214 requests; prompt Pearson 0.7356, runner-up 0.6173). Section 5 explicitly says evaluation uses the snippets in Figure 1. The offsets are still approximate plot-derived settings because sub-second boundaries, scaling/thinning and context mapping remain unknown.

## 2026-09-17 — task source audit

Primary GovReport, QMSum, DuReader, Multi-News, LongBench and pre-paper BookSum source revisions were pinned. The paper-linked `baidu/DuReader@c625076...` recursive tree contains 303 paths but no English/translation artifact and has no releases, contradicting the practical implication of Appendix C's “hosted at” statement. Public task sources remain usable only for modified-condition pilots because exact splits, sampled rows, prompts, mapping seeds, decoding and metric versions are absent.

## 2026-09-17 — full async attempt 1 and transactional review

Three W4/FP16 cycles ran on Llama 3.1 8B. Host enqueue was nonblocking; W4 same-history relative L2 stayed 0.00049–0.00102 with top-1 matches; final FP16 bytes/state were exact. The run is rejected as steady-state overlap evidence: first W4 decode included PagedAttention JIT (3.67 s), separate FP16 requests were not bit-exact despite rel-L2 0.00074–0.00130/top-1 matches, and interval intersection could count layer-wait idle time. Review also found partial expansion/shrink/morph transaction defects and alignment underflow. Repairs now pass five CUDA seam, ten transaction/controller and three C++ alignment tests; the retry adds real decode warmup and a pre-layer-wait event to measure actual compute/copy intersection.

A source-wide check also found that objective-added LLM-PQ (41.3%/82.3%) and PyramidKV (1.73×/2.4×) comparisons are absent from the supplied v2 PDF and LaTeX. PyramidKV occurs only in related work and LLM-PQ not at all. Values were moved out of paper headline references into an objective-only namespace and are not reproduction targets for v2.

Added the first raw-driven visual artifact: `figures/gen_fig_transfer_diagnostics.py` reads rejected full-model attempt 1, refuses a non-rejected source, and plots only the separately valid 3× W4/FP16 transfer durations and effective rates. CSV/PDF/PNG regenerated byte-identically in a clean pinned analysis environment; the caption explicitly excludes overlap and paper-agreement claims.

A follow-up read-only review still blocked the async executor: post-shrink restore failure could lose KV groups, pressure rollback ignored restore failure, double rollback-copy failure escaped without a poison state, and the pre-layer event interval was not actual activity evidence. Repairs add atomic KV snapshot/reattach recovery, truthful coordinator/executor synchronization, fail-closed poisoning, exact allocator/refusal gates, and three Chrome-trace analyzer tests. The pending full run now requires size-matched H2D activity to intersect an actual kernel on another stream; event intervals are supporting bounds only. Sixteen transaction/controller tests and four fail-closed activity-trace tests pass. Review findings are preserved at `results/raw/transaction-followup-review.md`. Re-review then found missing-stream metadata could falsely pass, poisoned recovery reattached invalid KV views, and swapped-queue length was mislabeled as a preemption count. The analyzer now requires explicit distinct streams, poisoned recovery leaves truthful partial state without reattachment, and both ownership/replay pilots record scheduler preemptions as not measured. The re-review is preserved at `results/raw/transaction-rereview.md`.

## 2026-09-16 — final pre-rerun audit and expansion-preflight repair

The claim verifier was found to treat duplicate pre-atomic `real-executor/results/` and `multirequest-ownership/results/` copies as current positive gates. The duplicates were removed, all H8/H25/H26/H28 links now point to explicitly classified `results-before-atomic-repair/`, and verifier output separates four current gates from two supporting historical gates. Current atomic CPU transaction artifacts are linked independently; the claim map now hashes/Git-audits 75 files.

A read-only readiness review then found that public `expand_kv()` could acquire inactive/duplicate/already-grouped or overlapping registered regions, potentially zeroing live FP16 storage or double-counting aliases. New tests first failed for all six invalid batch cases. The executor now materializes and rejects invalid batches before snapshot, wait, counter increment or native acquisition; tests verify zero acquisitions/waits and exact allocator/cache/group non-mutation. All 18 executor/coordinator transaction tests pass in 0.063 s. The same review found source-status omissions; all four pending GPU runners now cover the dependency lock, and synthetic replay also covers its executed JSON config. A focused independent rereview of commit `58537b3...` returned PASS with no supported-flow bypass or regression.

```bash
PYTHONPATH="$PWD/reproduction/runtime:$PWD/reproduction/runtime/candidate-python" \
  reproduction/.venv/bin/python -m unittest \
  reproduction.tests.test_real_executor_transactions \
  reproduction.tests.test_controller_integration \
  reproduction.tests.test_trace_analysis -v
python3 reproduction/scripts/verify_report_artifacts.py
```

The combined 22-test command and report verifier pass. GPU 4 initially remained at 16,501 MiB free and 98–100% utilization. At 21:36 UTC it became fully idle with 24,124 MiB free, so the current-revision real-executor runner was launched immediately. Before its first guard executed, a new external process claimed 9,106 MiB; the runner observed 15,009 MiB and safely exited 75 before deleting prior output, installing, or loading the model. The before/after snapshots and exact command are preserved under `results/raw/pending-gpu-*`. All devices were again below the frozen 17,408-MiB threshold, so no unchanged full-model retry was made.

The 4-MiB CUDA seam requires negligible capacity, so it was rerun on the occupied GPU solely for current-revision correctness. Five tests passed at source revision `14c006f...` with an empty source-status sidecar: host enqueue returned while prior use was unfinished, bytes/address were exact, the model consumed the ready event immediately before the affected layer, the async path accumulated no redundant native events, and partial expansion rollback published its barrier under the new preflight invariants. Contended timings (0.244-ms host enqueue, 0.562-ms CUDA-event interval) are supporting diagnostics only; measured test-process wall was 11.229 s including import/teardown and excluding install/build.

## 2026-09-17 — current-revision GPU closure

The available GPU survey found all seven RTX 3090s at 24,124 MiB free with no compute processes. After current CPU/provenance checks, the four frozen full-model commands were run serially on GPU 0, preserving all prior numbered attempts and recording before/pre-run/after snapshots plus vendor-manifest checks. The real executor passed injected partial-expansion rollback, real W4 module replacement, explicit reclaimed-KV expansion, shrink-before-restore LIFO recovery and exact final FP16 state. The ownership runner passed two-request reclaimed-block ownership, occupied-region refusal without mutation, post-release recovery, sentinels/counts/rows and final restoration. Both runs returned 0.

The first new async full-model run returned 1 only because the analyzer counted PyTorch's mirrored `gpu_user_annotation` trace records as duplicate phase markers; its raw trace already contained size-matched H2D/kernel intersections. A minimal parser repair selecting the single host `user_annotation`, plus a focused test update, passed four trace-analysis tests. The rerun then passed three W4/FP16 timing repeats and the raw activity gate: stream 17 H2D copies intersected stream 7 kernels by 288.290 μs and 290.210 μs, with same-history numerical gates and exact final FP16 bytes. The corrected synthetic GPU replay passed independently submitted 0/10/20-ms arrivals, 3/3 completion, 9/9 tokens, raw timestamp/TTFT/TPOT/KV accounting, warmup exclusion and final 8/8 KV release. It returned 0.

The current revision therefore closes implementation-level GPU verification for all four requested surfaces. It does not reproduce paper headline latency/quality/throughput or exact tables; those remain blocked by exact author inputs, task/trace/model settings and hardware.

## 2026-09-16 — bounded completion audit

The prior prompt-to-artifact audit rechecked every requirement group A–H against the then-current artifacts and correctly classified the four GPU surfaces as resource-blocked at that time. It remains historical context; the current closure audit supersedes its pending-rerun conclusion. Exact author-code, trace/task/controller/model/hardware conditions remain independently blocked, while the current-revision implementation GPU gates now pass.
