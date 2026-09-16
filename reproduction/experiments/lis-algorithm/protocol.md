# Protocol: exact conditioned-MDS LIS selection

Status: pre-registered before implementation/tests.

## Public seam

`morphserve.profiling.rank_layers(lts, lrs, mds)` returns an ordered profile plus every candidate score at every greedy step. `save_profile(profile, path)` persists it as canonical JSON. This seam directly represents Algorithm 1 and is independent of model-loading details.

## Prediction

A correct implementation will evaluate MDS for every remaining layer at each step and can produce a different order than any implementation that computes MDS only once at `Q=∅`. Ties will be resolved by lowest integer layer index as an explicitly labeled reconstruction choice because the paper is silent.

## Locked tests

1. Use a worked three-layer example whose conditioned MDS changes after the first selection; assert the exact order and all six MDS calls (`3+2+1`).
2. Assert the specified LIS weights `0.25/0.25/0.5` and literal candidate scores from the worked example.
3. Assert deterministic lowest-index tie breaking and record that policy in output metadata.
4. Reject mismatched/missing layer metrics and non-finite values.
5. Save/load canonical JSON and verify the full history, not just final order.
6. Include a deliberately static-MDS comparator in the test only and show it produces a different order; this prevents an unconditioned implementation from passing.

## Boundaries

- This experiment verifies selection logic only. It does not claim model activations, cosine reduction, quantization, or profiling time.
- Use the Python standard library only.
- Model-specific metric collection will be a later protocol after representation/sample choices are frozen.
