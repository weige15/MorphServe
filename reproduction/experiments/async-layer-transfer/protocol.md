# Protocol: asynchronous in-place layer transfer and overlap

Status: pre-registered before implementation.

## Public seams

1. `swiftllm_c.get_layer_memory_org_gpu(layer_id)` returns a non-owning uint8 view of the registered full GPU layer region.
2. `AsyncLayerCopier.enqueue(layer_id, pinned_source, size, tensor_map)` queues pinned-host→same-address-GPU copy on one persistent morphing stream, after the model's last-forward event, and returns typed views without host synchronization.
3. Candidate `LlamaModel` waits for a pending transfer event immediately before the affected layer and records one last-forward lifetime event after all layer/KV use.
4. `RealMorphingExecutor` uses this copier for W4/FP16 replacement; KV expansion waits for W4 readiness before reclaimed bytes become usable.

## Correctness pilot

On a small registered CUDA region, enqueue a delayed prior-use event and a pinned source copy. Require:

- enqueue host latency below the injected GPU delay;
- destination remains same-address and typed views map the declared offsets;
- waiting on the ready event yields byte-exact contents;
- transfer begins only after the prior-use event;
- missing/non-pinned/oversize inputs fail closed.

## Full-model overlap pilot

When GPU headroom permits, use local Llama 3.1 8B layer 25:

- warm FP16 and W4 kernels;
- start an asynchronous W4 copy before a decode step;
- record CUDA events for transfer, decode, and the point immediately before the affected-layer wait on separate streams;
- verify same-history W4 output and exact FP16 restoration;
- report transfer, decode, actual pre-layer-compute overlap, remaining transfer at the layer wait, concurrent wall/exposed stall, and enclosing-interval overlap separately;
- perform at least three timed repeats if resources permit.

## Classification

Independent reconstruction. The candidate API is blocking; this extension is not recovered author code. No paper overlap/latency agreement is claimed without full CUDA timing evidence.
