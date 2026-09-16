# Arbitrary profiled-order KV mapping result

## Negative result

The candidate fused mapping does **not** support the measured LIS prefix `[25,24,26]`.

- Group-0 K pointer (layer 25): `136328706532352`.
- Group-1 layer 24 is one negative 4,096-byte stride and works.
- Group-2 fixed-stride prediction is `136328706524160` (layer-23 tail).
- Actual registered layer-26 K pointer is `136328706536448`.
- Storing to virtual group-2 block left expected layer 26 unchanged and wrote the known K/V token into unintended layer 23.

The test process exited normally; this is deterministic address corruption, not a crash.

## Root cause

Candidate combined Triton kernels receive only the first reclaimed pointer and derive group `j` as `base - j*org_layer_param_size`. This is valid only for monotonically descending contiguous layers. Offline LIS produces arbitrary orders, and the current measured order breaks the assumption at the third layer.

## Classification

**Tested-not-reproduced:** arbitrary non-contiguous/profile-ordered KV mapping. Prior two-tail positive evidence is narrowed to equal-stride descending layouts.

A repair must pass explicit per-group pointers/indirection or dispatch one remapped kernel per region. Real controller/executor integration is blocked until this is fixed; otherwise following LIS can silently corrupt unrelated FP16 weights.
