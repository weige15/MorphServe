# Conditioned-MDS LIS algorithm result

## Result

The Algorithm 1 selection core is implemented at `runtime/morphserve/profiling.py` and verified through the public `rank_layers` seam.

- Every remaining candidate is evaluated at each step: the three-layer worked example records exactly six MDS calls (`3+2+1`).
- MDS receives the current ordered quantized set and changes the chosen order from static `[0,1,2]` to conditioned `[0,2,1]`.
- Scores use the paper's literal `0.25*LTS + 0.25*LRS + 0.5*MDS` formula and `argmax` direction.
- Lowest layer index is the deterministic tie break, explicitly labeled as a reconstruction choice.
- The full candidate history, inputs, scores, selected layers, weights, and tie rule are saved as canonical JSON.
- Missing/mismatched and non-finite metric values fail closed.

Four tests pass. Red/green logs are retained under `results/`.

## Scope limit

This reproduces the greedy selection logic only. It does not yet establish the paper's model-specific activation representations, cosine reduction axes, WikiText-2 sample identity, packed AWQ execution, or profiling time. Claim H23 is therefore only partially reproduced.

## Command

```bash
PYTHONPATH="$PWD/reproduction/runtime" \
  python3 -m unittest reproduction.tests.test_profiling -v
```

## Resource cost

CPU-only; no model load and zero GPU experiment time.
