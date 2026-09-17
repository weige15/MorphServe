# Attempt 3 — corrected synthetic GPU replay passed

Command:

```bash
CUDA_VISIBLE_DEVICES=0 reproduction/scripts/run_synthetic_gpu_replay.sh
```

The runner passed at HEAD `67bbdc2d7094c2b6dcaab6e58c745d1a1a0a815b`. GPU 0 had 24,124 MiB free before setup, before model load, and after teardown; no external compute process was present in the snapshots.

The corrected protocol used a separate warmup request and a cached-decode warmup before the timed origin. Three arrivals were submitted independently at the frozen 0/10/20-ms schedule with actual submit span 21.049 ms, rather than waiting for earlier completions. All IDs completed once, all emitted exactly three tokens (9 total), with no error or timeout.

`raw.jsonl` records scheduled, actual submission, queue, first-token, token-level and completion timestamps, TTFT/TPOT, generated IDs/text, precision, KV occupancy/capacity and explicit `preemptions: null` / `scheduler_preemptions_measured: false`. `summary.json` regenerates from those raw records using Hyndman–Fan type 7 percentiles: 3 requests, 3 completions, 0 errors/timeouts, p95 TTFT 0.0597138 s and p99 TPOT 0.0682005 s. `run-metadata.json` records 13.303 s initialization and 3.622 s warmup excluded from replay timing; final KV capacity/free state is 8/8.

## Classification

**Modified-condition instrumentation/replay validation only.** This is not Azure, BurstGPT, a paper trace replay, or paper latency reproduction.
