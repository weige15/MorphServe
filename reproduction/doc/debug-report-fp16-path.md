# Debug Report

## Symptom

The first normalized FP16 parity run stopped before model loading because the runner used an unrelated inherited `MODEL_PATH=meta-llama/Meta-Llama-3.1-8B` instead of its local snapshot default.

## Reproduction Command

Working directory: `/nfs/home/s314511048/MorphServe`
Shell: Bash
Runtime: Python 3.12.3 in `reproduction/.venv`
Environment: uv-managed isolated environment
Relevant environment variables:
```text
MODEL_PATH=meta-llama/Meta-Llama-3.1-8B
CUDA_VISIBLE_DEVICES=0
```

```bash
CUDA_VISIBLE_DEVICES=0 reproduction/scripts/run_candidate_fp16_baseline.sh
```

## Expected Behavior

Use local snapshot `d04e592...`, load Transformers and normalized candidate sequentially, and emit parity metrics.

## Actual Behavior

The commands artifact records `--model-path "meta-llama/Meta-Llama-3.1-8B"`. The Python script resolved it relative to the repository and Transformers rejected the resulting nonexistent local path before any model/GPU work.

## Error Log

```text
HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name':
'/nfs/home/s314511048/MorphServe/meta-llama/Meta-Llama-3.1-8B'
```

## Failure Layer Classification

Most likely layer:

* Command problem: no
* Permission problem: no
* Shell/script invocation problem: no
* Environment problem: yes
* Dependency problem: no
* Python/package/import problem: no
* GPU/CUDA problem: no
* Distributed/torchrun problem: no
* Filesystem/path problem: yes
* Data/checkpoint/model file problem: no
* Code logic problem: yes (runner variable is too generic)
* Configuration problem: yes
* Resource problem: no
* Concurrency/race problem: no
* Unknown/insufficient evidence: no

Final classification: inherited environment-variable collision in the runner.

## Hypotheses

### Hypothesis 1: generic `MODEL_PATH` collision
Why it could explain the symptom: runner uses `${MODEL_PATH:-local-default}` and the shell already defines `MODEL_PATH`.
Evidence for: `commands.txt` records the inherited value; the local default exists but was not selected.
Evidence against: none.
How to verify: rename the override to `MORPHSERVE_MODEL_PATH` and inspect the regenerated command before load.

### Hypothesis 2: missing local checkpoint
Why it could explain the symptom: invalid paths can trigger the same Transformers error.
Evidence for: the selected path is nonexistent.
Evidence against: the intended snapshot exists and was hashed in `results/raw/environment.json`.
How to verify: `test -f <snapshot>/config.json`.

## Most Likely Root Cause

The runner's generic `MODEL_PATH` override collided with an unrelated session variable. No candidate code or GPU model execution was reached.

## Minimal Fix

Use only `MORPHSERVE_MODEL_PATH` as the optional override and fail early unless `${MODEL}/config.json` exists.

## Verification

```bash
CUDA_VISIBLE_DEVICES=0 reproduction/scripts/run_candidate_fp16_baseline.sh
```

Expected verification result:

```text
commands.txt contains snapshot d04e592...
metrics.json exists
run.exitcode reflects the parity gate rather than path setup
```
