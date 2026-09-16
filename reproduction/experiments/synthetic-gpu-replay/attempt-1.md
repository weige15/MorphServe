# Attempt 1 — decode JIT leaked into timed replay

Accounting gates passed and arrivals occurred at 0.00014/0.01097/0.02124 s, but the warmup request performed prefill only. The first timed decode therefore compiled PagedAttention kernels, producing multi-second per-layer logs and P99 TPOT 3.64 s. This is valid raw initialization evidence but not steady-state serving latency.

Changed retry: warm one cached decode step after warm prefill, then free the warm request before starting the replay origin. No trace, model, timeout, or metric gate changes.
