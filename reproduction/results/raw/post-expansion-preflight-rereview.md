# Independent post-expansion-preflight rereview

Audited commit: `58537b336444aaabf685fa6e85f6c2e7c3fc5e2a`  
Classification: read-only readiness review of the repaired executor and runner provenance scopes.

## Review
- **Correct — P1 fully fixed for the supported executor flow.** `RealMorphingExecutor._expand_rejection_reason()` rejects empty, non-integer, duplicate, inactive, already-grouped, unregistered, invalid-tail, and overlapping-region batches before any acquisition (`reproduction/runtime/morphserve/real_executor.py:76-113`). `expand_kv()` invokes it before snapshotting, waiting, incrementing counters, or calling the extension (`real_executor.py:161-177`).
- **Correct — tests prevent mutation or acquisition on rejection.** The table-driven test covers six invalid batch classes and verifies unchanged groups, cache lists, allocator counts/bitmap, waits, acquisition calls, and expansion counters (`reproduction/tests/test_real_executor_transactions.py:41-61`). The overlap test confirms rejection before acquisition (`test_real_executor_transactions.py:63-70`).
- **Correct — no production regression found.** Normal runner inputs use integer, unique, active W4 layers backed by `backup_fp16_layer()` metadata. Converting the input iterable to a list also avoids generator reuse problems (`real_executor.py:161-165`).
- **Correct — P2 fully fixed.** All four runners now include the dependency lock in `source-status.txt`; synthetic replay additionally includes its executed JSON configuration:
  - `run_real_executor_pilot.sh:12-13`
  - `run_multirequest_ownership.sh:9-10`
  - `run_async_full_model_overlap.sh:8-9`
  - `run_synthetic_gpu_replay.sh:8-9`
- **Correct — saved validation artifact is updated.** `reproduction/experiments/async-layer-transfer/transaction-results/test.log:1-25` records all 18 transaction/coordinator tests passing, including both new preflight tests; exit code is `0`.
- **No bypass found in the supported construction path.** The region check trusts `backups[*].base/size`, which are produced alongside native registration by `backup_fp16_layer()`. Arbitrary external mutation of the process-global extension registry after executor construction remains outside the tested lifecycle.
**No issues found.**
### Readiness verdict: **PASS**
HEAD `58537b336444aaabf685fa6e85f6c2e7c3fc5e2a` is ready for the pending GPU reruns. Current-revision GPU outcomes remain missing evidence, not a code defect. Each resulting `source-status.txt` should be inspected and required to be empty before treating a run as clean-revision evidence.
No commands were freshly executed because this revived review environment still provides no shell runner. The committed 18-test artifact and source were inspected directly.
```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "No defects found: expansion preflight is implemented at reproduction/runtime/morphserve/real_executor.py:76-113 and invoked before mutation at lines 161-177; rejection tests are at reproduction/tests/test_real_executor_transactions.py:41-70; all four runner status scopes include their executed lock/config inputs."
    }
  ],
  "changedFiles": [
    "reproduction/runtime/morphserve/real_executor.py",
    "reproduction/tests/test_real_executor_transactions.py",
    "reproduction/scripts/run_real_executor_pilot.sh",
    "reproduction/scripts/run_multirequest_ownership.sh",
    "reproduction/scripts/run_async_full_model_overlap.sh",
    "reproduction/scripts/run_synthetic_gpu_replay.sh",
    "reproduction/experiments/async-layer-transfer/transaction-results/test.log",
    "reproduction/docs/completion-audit.md",
    "reproduction/results/raw/claim-evidence-provenance.json"
  ],
  "testsAddedOrUpdated": [
    "reproduction/tests/test_real_executor_transactions.py",
    "reproduction/experiments/async-layer-transfer/transaction-results/test.log"
  ],
  "commandsRun": [
    {
      "command": "PYTHONPATH=\"$PWD/reproduction/runtime:$PWD/reproduction/runtime/candidate-python\" reproduction/.venv/bin/python -m unittest reproduction.tests.test_real_executor_transactions reproduction.tests.test_controller_integration -v",
      "result": "not-run",
      "summary": "No shell execution tool was available. The committed artifact for this exact command records 18 tests passing in 0.063s with exit code 0."
    }
  ],
  "validationOutput": [
    "HEAD verified as 58537b336444aaabf685fa6e85f6c2e7c3fc5e2a.",
    "Saved transaction artifact records 18/18 tests passing, including invalid-batch and overlapping-region preflight tests.",
    "All four GPU runners capture HEAD and relevant runtime, runner, dependency-lock, and synthetic-config worktree status."
  ],
  "residualRisks": [
    "Current-revision real-executor, ownership, asynchronous-overlap, and synthetic-replay GPU results remain pending sufficient GPU resources.",
    "Runner source-status files capture dirty inputs but do not themselves abort on a non-empty status; inspect and require empty files when accepting rerun evidence.",
    "Expansion region validation assumes backup metadata and native registrations are not externally overwritten after executor construction."
  ],
  "noStagedFiles": true,
  "diffSummary": "Adds fail-closed KV expansion preflight with zero-mutation tests and broadens all pending GPU runner source-status scopes to cover dependency locks and the synthetic replay config.",
  "reviewFindings": [
    "no blockers"
  ],
  "manualNotes": "Read-only rereview of the stable clean target; no files modified. Fresh test execution was unavailable, so the committed command, log, and exit code were inspected."
}
```
