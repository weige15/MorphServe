# Real WikiText-2 conditioned-MDS LIS pilot result

## Result

The bounded real-W4 pilot passed all gates and produced frozen order **[29, 30, 31]**.

Calibration provenance:

- locally cached Salesforce WikiText-2 raw v1 train split;
- first exactly 2,048 tokenizer tokens from ordered non-empty rows `[1,3,4,5,7,9,10,11,13,15,16,17,19,21]`;
- text SHA-256 `a590a32e...e8c8`; token-ID SHA-256 `581a6f47...5b85`;
- no random seed/sample selection.

Metrics:

- LTS: layer 29 `0.949241`, 30 `0.895856`, 31 `0.556244`;
- LRS: 29 `0.999717`, 30 `0.999221`, 31 `0.998399`;
- six conditioned MDS calls were executed (`3+2+1`), including changed values after Q grew;
- step-0 LIS: 29 `0.986415`, 30 `0.973079`, 31 `0.887672`;
- all 56 restored-tensor checks and final FP16 logits were exact.

The immutable profile is `profiles/llama31-8b-wikitext2-layers29-31-pilot.json`.

## Classification

**Approximate/modified-condition profiling reproduction (bounded pilot).** This validates real activation collection, explicit flattened cosine semantics, real W4 layer sets, every-candidate conditioned MDS, tested `argmax` selection, saved history/order, and exact restore.

It does **not** reproduce the paper's full 32-layer profile/order/perplexities or under-15-minute claim. Inner profile work for three layers/seven forward sets was 14.83 s; full runner wall was 45.22 s including model/data load. No full-profile extrapolation is reported.

## Limitations

- layers 29–31 only;
- one deterministic calibration sequence and last-token logits as model output;
- local Llama 3.1 8B/AutoAWQ/RTX 3090;
- long-prefill AutoAWQ may dequantize packed weights for matmul under its heuristic, though stored variants are real W4.
