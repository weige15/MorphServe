# Protocol: expanded real WikiText-2 LIS profile (layers 24–31)

Status: pre-registered before execution.

## Objective

Repeat the frozen three-layer real-W4 profiler unchanged for decoder layers `24,25,26,27,28,29,30,31`. This tests whether the pilot order `[29,30,31]` is stable when additional candidates and conditioned interactions are introduced.

## Frozen conditions

- Same local Llama 3.1 8B FP16/AutoAWQ assets and RTX 3090 environment.
- Same WikiText-2 row rule, exact 2,048-token sequence, text/token hashes, residual-stream/logit representations, flattened float32 cosine, weights 0.25/0.25/0.5, zero-based indices, lowest-index tie break.
- Evaluate every remaining candidate at each step: exactly `8+7+...+1 = 36` MDS calls and 36 unique non-empty quantized sets.
- Cache each set once; restore all FP16 bytes after evaluation.

## Gate

1. Calibration text/token hashes exactly match the three-layer pilot.
2. Eight LTS and eight LRS values retained.
3. Exactly 36 conditioned MDS calls and unique evaluated sets.
4. Saved profile contains every candidate exactly once and its candidate count decreases 8→1.
5. Every restored tensor check and final FP16 logits are exact.
6. Report whether the relative order of common layers 29/30/31 remains `[29,30,31]`; this is an observation, not a pass criterion.
7. Process exits 0; no under-15-minute/full-profile claim.

## Resource bound

Eight FP16 backups plus W4 variants require about 4.39 GB pinned memory, below the 16.84-GB memlock limit. A separate byte audit documents why all 32 simultaneous decoder variants exceed that limit.
