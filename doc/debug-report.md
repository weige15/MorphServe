# Debug Report

## Symptom

The first v11 Phase A plan invocation stopped before creating a run directory or initializing the model.

## Reproduction Command

Working directory: `/nfs/home/s314511048/MorphServe/swiftLLM`
Shell: `/bin/bash`
Runtime: `Python 3.12.3`
Environment: `/nfs/home/s314511048/.venvs/morphserve-vllm0112`
Relevant environment variables:
```text
CUDA_VISIBLE_DEVICES=5
PYTHONPATH=/nfs/home/s314511048/.cache/morphserve/swiftllm-c-torch29/lib:/nfs/home/s314511048/MorphServe/swiftLLM
```

```bash
cd swiftLLM && CUDA_VISIBLE_DEVICES=5 PYTHONPATH="$AWQ_EXT:$PWD" "$AWQ_ENV/bin/python" -m benchmark.run_throughput_confirmation_plan --plan ../benchmark-results/throughput-confirmation-v11/phase-a-run-plan.json ...
```

## Expected Behavior

The plan runner should validate the repo-relative workload and launch the first fresh FP16 process.

## Actual Behavior

The child condition received `benchmark-results/release-side-runtime-v10/input/low-only.jsonl` while its working directory was `swiftLLM/`, so preregistration validation failed before model initialization and before a run directory was created.

## Error Log

```text
FileNotFoundError: [Errno 2] No such file or directory:
'benchmark-results/release-side-runtime-v10/input/low-only.jsonl'
```

Full log: `benchmark-results/throughput-confirmation-v11/logs/v11-phase_a-low-00-fp16.log`.

## Failure Layer Classification

Most likely layer:

* Command problem: yes
* Permission problem: no
* Shell/script invocation problem: yes
* Environment problem: no
* Dependency problem: no
* Python/package/import problem: no
* GPU/CUDA problem: no
* Distributed/torchrun problem: no
* Filesystem/path problem: yes
* Data/checkpoint/model file problem: no
* Code logic problem: no
* Configuration problem: no
* Resource problem: no
* Concurrency/race problem: no
* Unknown/insufficient evidence: no

Final classification: wrong working directory for repo-relative paths in the raw regeneration command.

## Hypotheses

### Hypothesis 1: wrong working directory

Why it could explain the symptom: plan rows intentionally store paths relative to the repository root, while the command changed into `swiftLLM/`.
Evidence for: the missing relative path exists from the repository root; traceback fails in `sha256_file` before any CUDA/model work.
Evidence against: none.
How to verify: rerun the same module from the repository root with `PYTHONPATH="$AWQ_EXT:$PWD/swiftLLM"`.

### Hypothesis 2: missing workload artifact

Why it could explain the symptom: `FileNotFoundError` can indicate an absent input.
Evidence for: the immediate exception names the workload.
Evidence against: `benchmark-results/release-side-runtime-v10/input/low-only.jsonl` exists and its SHA-256 passed manifest creation from the repository root.
How to verify: hash the file from the repository root.

## Most Likely Root Cause

The recorded v11 command used the correct module environment but the wrong working directory. The implementation and workload are intact; this was an administrative pre-measurement invocation failure, not a serving or scientific failure.

## Minimal Fix

Run from the repository root and set `PYTHONPATH="$AWQ_EXT:$PWD/swiftLLM"`. Update only the v11 regeneration command; do not alter the frozen plan, protocol, controller, workload, or v10.

## Verification

```bash
CUDA_VISIBLE_DEVICES=5 PYTHONPATH="$AWQ_EXT:$PWD/swiftLLM" "$AWQ_ENV/bin/python" -m benchmark.run_throughput_confirmation_plan --plan benchmark-results/throughput-confirmation-v11/phase-a-run-plan.json ... --resume
```

Expected verification result:

```text
run 1/24: v11-phase_a-low-00-fp16 attempt 0
...
v11 phase_a run plan complete: 24/24 valid
```
