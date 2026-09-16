# Debug Report

## Symptom

Repeated forwards through the same static packed AutoAWQ W4 model and identical token IDs produce non-bit-exact logits, although top-1/top-5 remain stable.

## Reproduction Command

Working directory: `/nfs/home/s314511048/MorphServe`
Shell: Bash
Runtime: Python 3.12.3; PyTorch 2.4.0+cu121; AutoAWQ 0.2.9; Triton 3.0.0
Environment: `reproduction/.venv`
Relevant environment variables:
```text
CUDA_VISIBLE_DEVICES=1
```

```bash
CUDA_VISIBLE_DEVICES=1 reproduction/scripts/run_static_autoawq_baseline.sh
```

## Expected Behavior

Five identical W4 forwards should produce exactly equal logits under the added deterministic verification gate.

## Actual Behavior

All top-k predictions are stable, but relative L2 difference from the first run ranges from 0.002256 to 0.004930 and max absolute difference reaches 0.058594. The command exits 1 because `repeat_logits_exact=false`.

## Error Log

```text
repeat_logits_exact: false
repeat top1: [505, 505, 505, 505, 505]
max abs vs first: [0, 0.058594, 0.039062, 0.058594, 0.035156]
```

## Failure Layer Classification

Most likely layer:

* Command problem: no
* Permission problem: no
* Shell/script invocation problem: no
* Environment problem: no
* Dependency problem: no
* Python/package/import problem: no
* GPU/CUDA problem: no
* Distributed/torchrun problem: no
* Filesystem/path problem: no
* Data/checkpoint/model file problem: no
* Code logic problem: no
* Configuration problem: yes (split-K reduction choice)
* Resource problem: no
* Concurrency/race problem: yes (atomic reduction order)
* Unknown/insufficient evidence: no

Final classification: expected floating-point nondeterminism from the selected fused split-K Triton GEMM path.

## Hypotheses

### Hypothesis 1: split-K atomic accumulation order
Why it could explain the symptom: parallel partitions atomically add FP partial sums in nondeterministic order.
Evidence for: AutoAWQ's small-input `WQLinearMMFunction` calls `awq_gemm_triton(... split_k_iters=8)`; `awq_gemm_kernel` uses `tl.atomic_add` whenever `SPLIT_K != 1`. Observed differences are small while top-k is stable.
Evidence against: no competing source with stronger evidence.
How to verify: run a separately labeled `SPLIT_K=1` diagnostic; do not substitute it for the paper's fused path.

### Hypothesis 2: mutable model/cache state
Why it could explain the symptom: repeated inference might mutate hidden cache state.
Evidence for: autoregressive models often cache K/V.
Evidence against: every call sets `use_cache=False`; top-k is stable; source-level atomic reduction directly explains differences.
How to verify: no further retry needed unless deterministic execution becomes a requirement.

## Most Likely Root Cause

AutoAWQ 0.2.9's real low-bit small-input Triton kernel uses eight split-K partitions and nondeterministic atomic floating-point accumulation. This is not fake quantization or checkpoint corruption.

## Minimal Fix

Do not “fix” the reproduction path merely to satisfy bit-exactness. Preserve the fused W4 execution and compare controlled histories using a predeclared numerical envelope and stable token/top-k criteria. If a deterministic diagnostic is specifically needed, use split-K 1 in a separate experiment and report its kernel deviation/performance cost.

## Verification

```bash
python - <<'PY'
import json
m=json.load(open('reproduction/experiments/static-autoawq/results/metrics.json'))
print(m['repeat_diagnostics'])
PY
```

Expected verification result:

```text
all top1/top5 stable; nonzero bounded logit differences; exact-repeat gate remains failed
```
