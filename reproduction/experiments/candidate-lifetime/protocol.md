# Protocol: in-flight restore safety and repeated adaptation

Status: pre-registered before execution.

## Questions

1. Does the candidate restoration path wait for in-flight consumers/writers of reclaimed KV storage before copying FP16 weights back into the same region?
2. After the native metadata repair, can an otherwise synchronized layer complete repeated quantize → reclaim → use → release → restore cycles without address drift or content loss?

## Prediction

The candidate has no event registration/wait between reclaimed KV users and `replace_layer_quant2org`, so a delayed non-default-stream write launched before restore will complete after restore and corrupt restored bytes. An explicit wait before restore will be a clean control. Fully synchronized repeated cycles should pass.

## Frozen GPU race

- One 4,096-byte layer region, 1,024-byte packed prefix, and reclaimed K/V views as in the prior gate.
- Queue a delayed `k_cache.fill_(7)` on a separate PyTorch CUDA stream using `torch.cuda._sleep(100_000_000)`.
- Confirm its completion event is not ready, immediately call candidate restore (which uses its own newly created CUDA stream and synchronizes only that stream), then wait for the writer.
- Report byte differences from original after the writer. The test passes as a **negative finding** only if corruption is observed.
- Control: explicitly synchronize the writer event before restore; final bytes must equal original.

## Repetition gate

Run five synchronized quantize/reclaim/write/release/restore cycles on one registered layer. Each cycle must reuse the same base, restore exact bytes, keep allocator-view delta at zero, and exit cleanly.

## Boundaries

This is a synthetic writer race, not a full attention kernel or scheduler. It proves the absence/presence of a general lifetime barrier at the storage boundary. It does not quantify real serving race probability. No workaround such as `os._exit` is allowed.
