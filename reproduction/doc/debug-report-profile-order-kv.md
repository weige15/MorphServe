# Debug Report

## Symptom

A virtual KV block assigned to the third reclaimed group in profile order `[25,24,26]` is written into unregistered layer-23 memory rather than layer 26.

## Reproduction Command

Working directory: `/nfs/home/s314511048/MorphServe`
Runtime: Python 3.12.3, PyTorch 2.4.0+cu121, candidate Triton kernel
Environment: `reproduction/.venv`, `CUDA_VISIBLE_DEVICES=1`
Relevant environment variables:
```text
MORPHSERVE_CANDIDATE_PYTHON=reproduction/vendor/author-morphserve/MorphServe
```

```bash
# Exact command saved in:
reproduction/experiments/noncontiguous-profile-order/results/commands.txt
```

## Expected Behavior

Virtual group 2 maps to the registered reclaimed tail for layer 26.

## Actual Behavior

Layer 26 remains zero; the known K/V token appears in layer 23. Expected and computed pointers differ by 12,288 bytes.

## Error Log

```text
group2 fixed-stride prediction: 136328706524160
registered layer26 address:      136328706536448
expected_layer26_unchanged: true
unintended_layer23_received_token: true
```

## Failure Layer Classification

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
* Code logic problem: yes
* Configuration problem: no
* Resource problem: no
* Concurrency/race problem: no
* Unknown/insufficient evidence: no

Final classification: Triton/C++ address-remapping logic assumes descending contiguous layer order.

## Hypotheses

### Hypothesis 1: fixed negative-stride mapping cannot represent LIS order
Why it could explain the symptom: kernels compute `first_pointer - group_index*layer_stride`.
Evidence for: groups 0/1 correspond to layers 25/24, while group 2 computes layer 23; the measured profile selects layer 26.
Evidence against: none.
How to verify: the synthetic test directly observes both predicted and actual addresses.

### Hypothesis 2: incorrect virtual block ID
Why it could explain the symptom: a wrong group number would target the wrong tail.
Evidence for: none.
Evidence against: block 26 equals `2 original + 2*12`, exactly group 2 block 0.
How to verify: metrics record the block/order/address arithmetic.

## Most Likely Root Cause

The candidate passes only `k_cache_new[0]/v_cache_new[0]` into combined kernels and reconstructs every later region by fixed subtraction. That optimization is incompatible with arbitrary offline profile order and can overwrite unrelated layer memory.

## Minimal Fix

Use explicit per-group base pointers/indirection, or the candidate's multi-kernel remap design that dispatches each group tensor with a local block table. Preserve a fast fixed-stride path only when runtime validation proves the selected layers are descending contiguous.

## Verification

Rerun the counterexample plus the existing two-tail dense-attention oracle. A repaired implementation must write layer 26, leave layer 23 unchanged, and preserve attention output.

## Applied Repair and Verification

`runtime/candidate-python` now provides explicit-region store dispatch with negative-ID guards and repairs the candidate multi-kernel attention helper to use passed region tensors/tables. The `[25,24,26]` repair test writes layer 26, leaves layer 23 unchanged, and matches the dense attention oracle with max error 0. The descending combined-path regression also remains max-error 0. Evidence: `experiments/noncontiguous-mapping-repair/results/`.
