# Current-revision full-layer asynchronous overlap result

The frozen protocol `protocol.md` was executed with `CUDA_VISIBLE_DEVICES=0 reproduction/scripts/run_async_full_model_overlap.sh` on local Llama 3.1 8B and the local AutoAWQ W4 G128 derivative. GPU 0 had 24,124 MiB free before setup, before model load, and after teardown; no external compute process was present in the snapshots.

The first current-revision run is preserved as `full-model-results-attempt-2/`: it returned 1 because the analyzer counted the profiler's mirrored `gpu_user_annotation` events as duplicate phase markers. The raw trace itself showed the expected copy/kernel activity. A focused parser repair selects the single host `user_annotation` marker, has four unit-test gates passing, and does not weaken copy size, stream identity or interval-intersection checks.

The subsequent current output `full-model-results/metrics.json` passed all gates:

- three unprofiled W4/FP16 repeats ran after warm FP16, W4, prefill and cached-decode kernels;
- host enqueue was nonblocking for all six rows (0.213–0.397 ms; the ready event was unfinished at return);
- W4 H2D transfer was 14.8536–14.8766 ms and FP16 H2D was 53.1627–53.2377 ms;
- transfer/decode/event fields are recorded separately per row, including remaining transfer at the layer wait and exposed decode delta;
- event interval intersection is only supporting evidence; the separate raw Chrome CUDA activity analysis found size-matched H2D copies on morph stream 17 intersecting actual kernels on decode stream 7 by 288.290 μs (W4) and 290.210 μs (FP16);
- same-history W4/FP16 numerical gates passed with top-1 agreement and relative-L2 below 0.005 for all rows;
- the final FP16 region bytes were exact and executor/model state returned to FP16 with no pending transfer events.

Raw `cuda-activity-trace.json`, derived `activity-analysis.json` and `transfer-summary.json`, command, source revision/status, manifest checks, GPU before/pre-run/after snapshots and logs are preserved in `full-model-results/`. The run is a modified-condition implementation/overlap measurement, not a reproduction of the paper's ≈4/16/6 ms examples or headline performance.
