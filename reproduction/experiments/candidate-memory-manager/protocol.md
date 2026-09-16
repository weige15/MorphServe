# Protocol: candidate C++ in-place storage and reclaimed-KV memory

Status: pre-registered before GPU execution.

## Question

Do the candidate artifact's compiled C++ bindings actually copy packed bytes into the registered FP16 GPU base address and expose the unused tail as bounded, writable FP16 K/V block storage without allocating a second GPU layer?

## Prediction

For a synthetic contiguous layer region, `replace_layer_org2quant` will preserve the registered GPU base address, `acquire_new_kvcache` will create K/V views wholly inside the unused tail, and writes through those views will modify the owning layer buffer. An undersized tail will raise an allocation error. Restoration will work only after the test releases reclaimed views; this protocol does not claim in-flight safety.

## Frozen setup

- Candidate source commit `85c4fbf...`, compiled unmodified with PyTorch 2.4.0+cu121 and CUDA toolkit 12.4.
- One free RTX 3090 exposed as logical CUDA device 0.
- Synthetic original region: 4,096 bytes; packed replacement: 1,024 bytes.
- KV metadata: 2 layers, 1 KV head, block size 4, head dimension 8, FP16. One full K+V block is 256 bytes, so the 3,072-byte tail must expose 12 blocks.
- Source buffers must be pinned CPU tensors. Keep the owning GPU tensor live for the entire view lifetime.

## Required checks

1. Packed bytes match after copy and returned tensor base equals the registered GPU base.
2. K/V shapes are `[12,2,1,4,8]`; K starts at byte 1,024, V follows K, and V ends exactly at byte 4,096.
3. K/V writes alter the owner tail, demonstrating physical storage rather than counters.
4. Acquiring from a tail smaller than one full KV block raises `RuntimeError`.
5. Releasing K/V views before restore allows original bytes to be copied back at the same base.
6. CUDA-event timings and allocator bytes are recorded as mechanism evidence only, with no paper-latency comparison.

## Boundaries

- This does not test Triton block remapping, scheduler occupancy, in-flight CUDA lifetime, multiple reclaimed layers, AWQ GEMM, or serving overlap.
- It is supporting synthetic GPU evidence, not a full-model result.
