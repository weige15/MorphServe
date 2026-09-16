# Asynchronous layer-transfer correctness pilot

Three CUDA tests pass on the public seams frozen in the protocol.

- A 4,194,304-byte pinned-host copy was enqueued in 0.222 ms while the injected prior-use event was still unfinished.
- The separate-stream copy itself measured 0.605 ms between CUDA events.
- The returned typed view retained the registered destination address and all bytes matched after the just-in-time wait.
- A synthetic `LlamaModel._forward` waited immediately before the affected layer, observed the copied value, consumed the pending event, and recorded a new end-of-forward lifetime event.
- Pageable and oversize sources failed closed.

This establishes non-blocking host enqueue, event order, same-address views, and the model-use barrier on a small CUDA region. It is supporting correctness evidence only: the GPU was externally busy, the payload is not a decoder layer, and no decode/copy overlap or paper latency result is inferred. The full-model pilot remains pending adequate GPU headroom.
