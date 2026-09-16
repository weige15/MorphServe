# Candidate Python reconstruction provenance

Base: `vendor/author-morphserve/MorphServe/` from `MorphServe/MorphServe@85c4fbf753b6eb47613bdd7ea423c38b48054544`, copied without checked-in bytecode and renamed to the import name used internally: `swiftllm`.

Current scoped changes:

- lazy package exports to avoid importing optional morphing dependencies for FP16-only use;
- complete `EngineConfig` field/default/CLI mapping;
- Llama 3 RoPE construction matching Transformers' wavelength cutoffs/interpolation;
- tied/untied `lm_head` key selection from checkpoint metadata;
- evidence-only `return_logits` option; default greedy behavior is unchanged;
- one `record_layer_memory_use` call per reclaimed region at the end of a forward, after the default stream's waits on decode-attention streams.

No controller-mode policy, AWQ compatibility layer, or experiment-specific threshold has been added yet.
