# Protocol: explicit reclaimed-region CUDA lifetime barrier

Status: pre-registered before implementation.

## Hypothesis

If every stream that uses reclaimed storage records a layer-use event after its final queued operation, and FP16 restore waits on all recorded events before copying, the previously confirmed delayed-writer corruption will fall from 1,530 bytes to 0. An otherwise identical unrecorded writer must still corrupt bytes, proving test sensitivity.

## Scoped reconstruction

Add one C++ binding to `runtime/candidate-csrc`:

- `record_layer_memory_use(layer_id)` records a timing-disabled CUDA event on PyTorch's current CUDA stream;
- completed old events may be pruned;
- `replace_layer_quant2org` makes its copy stream wait for every outstanding event for that layer, synchronizes the copy as before, then destroys/clears those events;
- no vendor file is edited and no copy/address/capacity behavior changes.

## Frozen verification

1. Re-run the unrecorded delayed writer and require corruption (`>0` bytes).
2. Queue the same delayed write, record use on that stream, immediately restore, and require restore to wait and final corruption to be exactly 0.
3. Require the writer event to be unfinished before restore, so the barrier—not lucky completion—explains success.
4. Repeat event-protected use/restore three times with stable base and exact bytes.
5. Require process exit 0, unchanged vendor manifests, and preserved prior memory-manager/two-region tests.

## Boundary

The binding establishes a usable event/lifetime primitive. Full correctness still requires every candidate/reconstruction kernel call site that can touch a reclaimed region to record the event. This test does not make the candidate's synchronous copies overlap decoding or establish full-model integration.
