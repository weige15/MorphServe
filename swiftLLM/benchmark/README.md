# SwiftLLM Phase-1 benchmark harness

This harness is an open-loop client around the existing `swiftllm.Engine`.
`benchmark.run` schedules request tasks by wall-clock offsets and does not wait
for any previous request to finish. It supports `fixed` intervals and seeded
`poisson` gaps. Each request uses one fixed prompt and output length.

Run from `swiftLLM/` with the same environment as the validated baseline:

```bash
MODEL=/nfs/home/s314511048/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
VENV=/nfs/home/s314511048/.venv
PYTHONPATH="$PWD:$PWD/csrc" CUDA_VISIBLE_DEVICES=3 \
  "$VENV/bin/python" -m benchmark.run \
  --model-path "$MODEL" --target-rps 0.5 --arrival-mode fixed \
  --request-count 4 --prompt-token-count 8 --output-token-count 4 \
  --seed 2025 --telemetry-interval-s 0.1 \
  --output-dir ../benchmark-results/phase-1/baseline-low-load/runs --run-id fixed-arrivals \
  --expected-num-gpu-blocks 3880
```

The run writes `metadata.json`, `requests.jsonl`, `telemetry.jsonl`, and
`summary.json`. Regenerate the summary using only saved files:

```bash
PYTHONPATH="$PWD" "$VENV/bin/python" -m benchmark.summarize \
  ../benchmark-results/phase-1/baseline-low-load/runs/fixed-arrivals
```

## Timestamp and metric contract

`*_time_ns` values use one process-local `time.perf_counter_ns()` clock.
`arrival_time_ns` is the actual open-loop launch immediately before the engine
accepts the request. `planned_arrival_offset_s` is the offered schedule;
`actual_arrival_offset_s` and `arrival_jitter_s` are retained separately. The
reported inter-arrival rate uses `(N - 1) / (last_arrival - first_arrival)`;
the offered rate uses the configured workload window. The engine records
scheduler eligibility after tokenization and before
`Scheduler.on_requests_arrival`, first prefill immediately before the relevant
`model.forward`, and model output/completion immediately after `forward`
returns. The harness records when the streamed `StepOutput` is received.

The reported request metrics use client-observed streaming timestamps:

- `TTFT = first_stream_token_received - arrival`.
- `queueing_delay = first_prefill - scheduler_eligible`.
- `TPOT = (stream_completion - first_stream_token_received) / (output_tokens - 1)`;
  one-token requests report zero.
- `logical_kv_utilization = num_decoding_gpu_blocks / num_gpu_blocks`.

Engine-side TTFT/TPOT are also retained. Physical GPU memory is optional
telemetry only; logical KV occupancy and scheduler queues are the saturation
signals. Swap-out is counted as preemption in the summary because SwiftLLM's
existing scheduler returns preempted requests as swap-out work.

The only core changes are timestamp/counter observations in `server/engine.py`
and optional metadata fields in `server/structs.py`; `server/scheduler.py` is
unchanged and all scheduling decisions and model arguments remain upstream.
