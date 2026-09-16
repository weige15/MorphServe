# Attempt 1 — inference-tensor reset failure

The reference branch completed enough decode work to JIT-compile candidate PagedAttention, then the branch reset failed before the in-place branch. `LlamaModel.init_kvcache_and_swap` is decorated with `torch.inference_mode`, so its cache tensors are inference tensors; direct `copy_` outside an inference-mode context is forbidden.

No parity/state result was produced. The changed retry will wrap only direct cache reset/migration mutations in `torch.inference_mode`; model or algorithm behavior is unchanged. The large first-call attention times in the log are JIT compilation and excluded from performance evidence.
