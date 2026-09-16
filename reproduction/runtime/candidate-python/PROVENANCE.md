# Candidate Python reconstruction provenance

Base: `vendor/author-morphserve/MorphServe/` from `MorphServe/MorphServe@85c4fbf753b6eb47613bdd7ea423c38b48054544`, copied without checked-in bytecode and renamed to the import name used internally: `swiftllm`.

Current scoped changes:

- lazy package exports and lazy ROUGE dependency loading so FP16 core import does not require optional morphing/evaluation packages;
- complete `EngineConfig` field/default/CLI mapping;
- Llama 3 RoPE construction matching Transformers' wavelength cutoffs/interpolation;
- tied/untied `lm_head` key selection from checkpoint metadata;
- evidence-only `return_logits` option; default greedy behavior is unchanged;
- one `record_layer_memory_use` call per reclaimed region at the end of a forward for the candidate blocking restore path, after the default stream's waits on decode-attention streams;
- one Python CUDA completion event per forward plus a just-in-time wait immediately before a layer whose independent asynchronous replacement is pending; the async executor disables redundant C++ event accumulation.

Controller policy, AutoAWQ adaptation, and asynchronous copy orchestration remain isolated under `runtime/morphserve`; no experiment-tuned threshold is embedded here.
