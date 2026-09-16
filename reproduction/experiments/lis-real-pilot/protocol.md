# Protocol: real WikiText-2 conditioned-MDS LIS pilot

Status: pre-registered before implementation/execution.

## Scope

Run a bounded three-layer (`29,30,31`) pilot of the paper's exact Algorithm 1 on local Llama 3.1 8B with real AutoAWQ W4 variants. This validates real metric collection and conditioned selection; it is not a full 32-layer paper profile.

## Frozen calibration/representation choices

- Dataset: locally cached Salesforce WikiText-2 raw v1 training split.
- Sample rule: concatenate non-empty rows in original order with blank-line separators and take the tokenizer's first exactly 2,048 tokens. No random sampling; seed is `null`. Save row indices, text SHA-256, token IDs/hash, tokenizer snapshot.
- Layer index convention: zero-based decoder indices.
- Layer input/output: logical residual-stream state immediately before/after the complete candidate decoder layer (`input_embds + residual_buf`, `ffn_out + residual_buf`).
- LTS/LRS: one float32 cosine after flattening all 2,048×hidden elements.
- Model output for MDS: float32 cosine over flattened last-token vocabulary logits.
- Weights: LTS 0.25, LRS 0.25, MDS 0.5; select `argmax`; ties use lowest layer index as the documented reconstruction choice.
- Cache each evaluated quantized set once to avoid adding extra split-K variance.

## Gate

1. Sequence length exactly 2,048 and all calibration provenance saved.
2. Real W4 tensors/modules and same-base in-place copies for every evaluated layer set.
3. Local LTS/LRS retained for all three candidates.
4. At each greedy step, evaluate every remaining candidate's MDS conditioned on current Q: exactly `3+2+1=6` candidate-set evaluations (with singleton results reusable from local metrics).
5. Feed collected values through the already-tested `rank_layers` seam; saved order/history must match evaluation history.
6. Restore all FP16 layer bytes and final FP16 logits exactly; process exits 0.
7. Record wall/GPU conditions, but make no under-15-minute or quality claim from this pilot.
