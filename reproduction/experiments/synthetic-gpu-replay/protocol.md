# Protocol: frozen synthetic GPU replay/accounting baseline

Status: pre-registered with `configs/synthetic-gpu-replay.json` before execution.

## Objective

Connect the tested arrival replay harness to a real normalized Llama 3.1 8B FP16 worker and verify end-to-end request/token/timing/KV accounting. This is explicitly not Azure/BurstGPT or a paper performance comparison.

## Frozen setup

- Three arrivals at 0/10/20 ms, three output tokens each, prompts in the config.
- FP16 only; block size 16; eight GPU KV blocks; one worker.
- Dedicated one-thread GPU executor; async tasks submit independently, while a per-token lock serializes model steps and allows request interleaving.
- Warm one separate request before the timed origin; initialization/warmup saved separately.
- 120-s per-request timeout; account every request and partial token.

## Gate

1. All three actual submissions occur without waiting for earlier completion; submit span <100 ms.
2. All IDs complete exactly once with three tokens and no timeout/error.
3. Raw records contain scheduled/actual/queued/first/completion/token timestamps, TTFT/TPOT, precision, KV occupancy/capacity, preemptions and generated IDs.
4. Summary regenerates from JSONL and reports explicit percentile definition.
5. KV capacity returns fully free after all requests; preemptions/errors are zero.
6. Initialization/warmup excluded from replay timestamps and process exits 0.

## Boundary

Modified-condition instrumentation baseline only: short synthetic inputs, serialized token steps, no controller/morphing, no paper trace/window or latency claim.
