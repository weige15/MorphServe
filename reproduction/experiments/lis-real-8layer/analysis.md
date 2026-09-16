# Expanded eight-layer real LIS result

## Result

The corrected expanded run passed every gate and produced order:

`[25, 24, 26, 27, 28, 29, 30, 31]`

- Exact same calibration text/token hashes as the three-layer pilot.
- Eight LTS and LRS metrics retained.
- Exactly 36 conditioned MDS calls and 36 unique evaluated sets.
- Candidate counts decreased `[8,7,6,5,4,3,2,1]`.
- 336 restored-tensor checks and final FP16 logits were exact.
- Inner profile work: 58.14 s; full runner wall including load: 92.78 s.

The common layers preserve relative order `[29,30,31]`, but the expanded search places layers 25/24/26/27/28 before them. This demonstrates why a three-layer suffix cannot stand in for the full model order.

## Interaction observation

MDS is genuinely conditioned: for example layer 29 changes from `0.9983513` at `Q=∅` to `0.9984072` after layer 25 and `0.9984013` after layers 25/24/26/27/28. The order still largely follows LTS because local transformation similarities vary much more than MDS in this subset.

## Classification

**Approximate/modified-condition eight-layer profiling reproduction.** This materially strengthens Algorithm 1 evidence but is not the paper's 32-layer profile or ordering/perplexity table.

The paper-style simultaneous pinned FP16+W4 decoder variants for all 32 local layers require 17,585,668,096 bytes, exceeding the 16,844,414,976-byte memlock limit by 741,253,120 bytes before staging/runtime overhead. A full run therefore requires a documented staged/pageable variant strategy or additional authorization; limits will not be raised autonomously.

## Evidence

- `results/metrics.json`, `verification.log`
- `profiles/llama31-8b-wikitext2-layers24-31.json`
- `results/raw/full-profile-memory-feasibility.json`
