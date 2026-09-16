# Independent readiness review before expansion-preflight repair

Audited commit: `fac0fe5391d8f6e22d8a9cb305910d4ea805606a`  
Classification: preserved read-only review; its BLOCK verdict motivated the subsequent `expand_kv` preflight and runner-provenance repairs. It is not a review of the repaired descendant.

## Review
**Audited target:** clean commit `fac0fe5391d8f6e22d8a9cb305910d4ea805606a`, the supervisor-designated final descendant of the originally requested `e74d625`. Read-only; no files modified.
### Findings
- **P1 — `expand_kv()` lacks preflight validation and can corrupt FP16 weights or duplicate live KV regions.**  
  `reproduction/runtime/morphserve/real_executor.py:122-146` accepts any layer list. Unlike morph/restore, it does not require unique layers, require layers to be active W4 layers, reject layers already represented in `kv_groups`, or detect overlapping registered physical regions.
  - Calling it for an inactive FP16 layer reaches `acquire_new_kvcache()` and zeroes the area after the registered packed size (`reproduction/runtime/candidate-csrc/src/memory_manager.cpp:143-220`), even though FP16 weights still occupy that storage.
  - Passing a duplicate layer, or re-expanding an existing group, acquires the same region again. The equal-block-count check at `real_executor.py:134-135` passes, while lines 136-145 append duplicate aliases and double-count allocator capacity.
  - Registration itself overwrites entries without overlap checks at `memory_manager.cpp:45-62`.
  - Existing transaction tests contain no expansion precondition/duplicate-region case (`reproduction/tests/test_real_executor_transactions.py:30-111`).
  **Smallest fix:** before any wait/acquisition, materialize and validate the batch: unique layers, every layer active, no layer already grouped, and non-overlapping registered GPU ranges across existing and requested groups. Add rejection tests asserting zero acquisitions and unchanged allocator/cache/group state.
- **P2 — GPU source-status sidecars omit some executed inputs.**  
  All four pending runners correctly save `source-revision.txt` and relevant runtime/script status, but the status scopes omit `configs/fp16-requirements-lock.txt`; the synthetic runner additionally executes `configs/synthetic-gpu-replay.json` without including it in `source-status.txt` (`reproduction/scripts/run_synthetic_gpu_replay.sh:8-21`). A dirty version of either file could therefore be executed while the sidecar only reports clean runtime/scripts. The synthetic payload does preserve parsed config values, reducing but not eliminating exact file provenance risk.
  **Smallest fix:** include the lock file and any executed config in the explicit `git status --short -- ...` path list.
### Correct
- Morph/restore rollback and poisoned-state handling are coherent and truthful at the Python object/state level (`real_executor.py:77-119,194-240`).
- Atomic recovery snapshots and restores allocator counts, free bitmap, K/V lists, groups, and group size; poisoned rollback does not reattach uncertain regions (`real_executor.py:47-74,243-266`).
- Recovery orders compensated W4 copies before snapshot reattachment (`real_executor.py:259-264`).
- Shrink validates all occupancy before its first mutation (`real_executor.py:165-190`).
- Partial expansion failure publishes a CUDA barrier before restoring metadata (`real_executor.py:149-161`).
- Alignment-past-region and sub-block tails fail closed (`memory_manager.cpp:149-169`; `tests/test_candidate_memory_manager.py:114-133`).
- The coordinator uses `recover_fp16`, checks pressure rollback results, synchronizes executor/controller state, and poisons on divergence (`integration.py:55-129`).
- CUDA lifetime ordering is explicit: the copy stream waits on prior copy/model events, while model execution waits immediately before the affected layer (`autoawq_adapter.py:42-58`; `candidate-python/swiftllm/worker/model.py:307-313`).
- Overlap acceptance now requires exact-size H2D activity, non-null distinct streams, and positive kernel/copy intersection (`trace_analysis.py:55-108`; `async_full_model_overlap.py:95-104`).
- All four pending GPU runners capture revision/status and perform two checks for at least 17,408 MiB: before setup and immediately before model launch. Representative locations are `run_real_executor_pilot.sh:9-30`, `run_multirequest_ownership.sh:6-22`, `run_async_full_model_overlap.sh:5-21`, and `run_synthetic_gpu_replay.sh:5-20`.
- Historical executor/ownership artifacts are now explicitly isolated and classified as pre-atomic, both in the claim map and report (`claim-evidence-map.json:12,29-32`; `REPORT.md:100-122,141,158-161`).
- Ordinary scheduler preemption is explicitly recorded as unmeasured rather than zero (`scripts/multirequest_ownership.py:125`; `REPORT.md:118-120`).
- The verifier distinguishes current positive gates from supporting historical gates and documents its limits (`scripts/verify_report_artifacts.py:34-70`; `docs/completion-audit.md:148-154`).
### Validation
No commands were executed: this review environment exposed read-only file inspection but no shell/test runner. I inspected the saved evidence rather than representing it as a fresh run:
- Transaction log: 16 tests passed.
- Trace-analysis log: 4 tests passed.
- `results/report-verification.log`: PASS, with historical gates separately labeled.
- Current GPU reruns remain resource-blocked/missing evidence, not failed evidence.
## Merge verdict: **BLOCK**
Do not spend the pending GPU rerun budget yet. Fix `expand_kv()` preflight/duplicate-region defenses first; otherwise the reruns would validate a revision that still permits allocator and backing-storage corruption through a public executor operation. After that fix and CPU tests, the pending GPU protocols themselves are ready.
```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "P1 at reproduction/runtime/morphserve/real_executor.py:122-146 and P2 at reproduction/scripts/run_synthetic_gpu_replay.sh:8-21, with source-backed remediation."
    }
  ],
  "changedFiles": [
    "reproduction/REPORT.md",
    "reproduction/configs/claim-evidence-map.json",
    "reproduction/docs/completion-audit.md",
    "reproduction/results/raw/claim-evidence-provenance.json",
    "reproduction/results/raw/resource-usage-summary.json",
    "reproduction/results/report-verification.log",
    "reproduction/scripts/verify_report_artifacts.py"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "PYTHONPATH=\"$PWD/reproduction/runtime:$PWD/reproduction/runtime/candidate-python\" reproduction/.venv/bin/python -m unittest reproduction.tests.test_real_executor_transactions reproduction.tests.test_controller_integration reproduction.tests.test_trace_analysis -v",
      "result": "not-run",
      "summary": "No shell execution tool was available; saved logs were inspected separately."
    },
    {
      "command": "python3 reproduction/scripts/verify_report_artifacts.py",
      "result": "not-run",
      "summary": "Fresh execution unavailable; committed results/report-verification.log records PASS."
    }
  ],
  "validationOutput": [
    "HEAD verified as fac0fe5391d8f6e22d8a9cb305910d4ea805606a.",
    "Saved transaction log records 16 passing tests.",
    "Saved trace-analysis log records 4 passing tests.",
    "Saved report verifier records current_positive_gates separately from supporting_historical_gates."
  ],
  "residualRisks": [
    "P1: expand_kv permits inactive, duplicate, already-grouped, or physically overlapping regions.",
    "P2: runner source-status scopes omit dependency lock files and the synthetic replay config.",
    "Current-revision real executor, ownership, overlap, and synthetic replay GPU evidence remains pending sufficient GPU headroom.",
    "No fresh commands were run in this read-only tool environment."
  ],
  "noStagedFiles": true,
  "diffSummary": "Post-e74 audit commits isolate pre-atomic GPU artifacts, relink claims, separate historical verifier gates, and refresh provenance/report artifacts; runtime executor code remains unchanged.",
  "reviewFindings": [
    "blocker: reproduction/runtime/morphserve/real_executor.py:122-146 - expand_kv lacks active/group/duplicate/physical-region preflight and can corrupt backing storage or double-count aliases.",
    "note: reproduction/scripts/run_synthetic_gpu_replay.sh:8-21 - provenance status scope omits executed config and lock inputs.",
    "resource-blocked: current-revision GPU reruns have not yet produced evidence."
  ],
  "manualNotes": "Final audit target was changed by the supervisor from the originally requested e74d625 to its clean descendant fac0fe5. No files were modified during review."
}
```
