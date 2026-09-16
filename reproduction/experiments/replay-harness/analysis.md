# Scheduled replay/accounting harness result

Four tests pass:

- arrivals are launched from one monotonic origin independently of slower completions;
- completed, failed, and timed-out requests remain exactly once in output;
- per-token timestamps, TTFT, TPOT intervals, metadata/events, errors and partial tokens are retained;
- raw JSONL regenerates the same summary;
- percentiles use explicit Hyndman-Fan type-7 linear interpolation;
- duplicate IDs/non-monotonic traces fail closed.

## Classification

Supporting CPU harness evidence only. It is ready for a frozen modified-condition GPU adapter, but exact Azure/BurstGPT replay remains blocked by the source ambiguities documented in `docs/trace-audit.md`.
