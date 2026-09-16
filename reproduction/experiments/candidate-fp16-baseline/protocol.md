# Protocol: normalized candidate no-morph FP16 parity

Status: pre-registered before runtime normalization and GPU execution.

## Question

Can a minimally normalized copy of the candidate Python source reproduce matched Transformers FP16 next-token behavior for the available Llama 3.1 8B base checkpoint, with all morphing disabled and no KV cache?

## Scoped normalization

Create `runtime/candidate-python/swiftllm` from the immutable candidate `MorphServe/` tree, excluding checked-in bytecode. Changes allowed before this run:

- package name/import normalization and a non-eager package `__init__`;
- complete `EngineConfig` defaults/CLI field mapping;
- correct Llama 3 RoPE construction;
- correct tied/untied `lm_head` selection;
- an evidence-only `return_logits` option that leaves default greedy behavior unchanged;
- C++ lifetime/event repairs already tested separately.

No quantized layer, controller, or KV resizing path may be used.

## Frozen model/input

- Local `meta-llama/Llama-3.1-8B` snapshot `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b` (base, not instruct).
- Prompt: `Life blooms like a flower, far away` tokenized once by the local tokenizer; identical IDs go to both paths.
- Transformers 4.51.3 eager attention, FP16, `use_cache=False`.
- Normalized candidate FP16, `ignore_kvcache=True`, one request, logits returned for the last prompt token.

## Predeclared gate

Pass requires:

1. finite logits with identical vocabulary shape;
2. exact top-1 token match;
3. relative L2 logit error `<0.005` (taken from the previously measured local FP16 envelope, not tuned against MorphServe paper values);
4. exact loaded tensor audit for all top-level/layer weights, including `[up,gate]` contiguous storage;
5. process exit 0 and unchanged vendor manifests.

Report maximum absolute error but do not use it as a scale-independent pass gate. This is an RTX 3090/local-checkpoint modified-condition baseline, not paper-hardware evidence.

## Failure policy

If packaging/dependency/load/numerical behavior fails, preserve the artifact and diagnose the earliest failure before any AWQ or serving experiment. Do not weaken the gate after seeing results.
