# Protocol: scheduled-arrival replay and complete accounting

Status: pre-registered before implementation/tests.

## Objective

Implement a trace-independent asynchronous replay harness that schedules every arrival from a common monotonic origin, independently of prior request completion, and retains every request through completion/error/explicit timeout.

## Record contract

Per request save: request ID, prompt/reference, scheduled and actual submit timestamps, submit lag, queued/first-token/completion timestamps when supplied, emitted token IDs/count, per-token timestamps/TPOT intervals, error/timeout, preemptions, precision changes, KV occupancy/capacity samples, and arbitrary adapter metadata.

## Gate

1. A fake server slower than the inter-arrival spacing still receives all submissions near their schedules rather than serially.
2. Every submitted request appears exactly once in output, including failures/timeouts.
3. TTFT and TPOT are derived from saved timestamps with explicit units/denominators.
4. Percentiles use a documented deterministic definition.
5. Arrival order and request IDs are preserved; invalid/non-monotonic traces fail closed.
6. Raw JSONL and machine-readable summary regenerate from the same records.

## Boundary

Harness/accounting only. Fake-server passing does not establish GPU serving performance. Exact Azure/BurstGPT use remains blocked until source file/window/downscaling/context mapping are recovered or a modified-condition trace is frozen explicitly.
