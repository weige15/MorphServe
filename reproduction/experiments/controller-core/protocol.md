# Protocol: reconstructed monitor/controller core

Status: pre-registered before implementation/tests.

## Source boundary

The paper specifies monitored signals and example pressure thresholds (KV usage 85%, queue delay 100 ms) but not smoothing, persistence, hysteresis, per-mode layer amounts, maxima, or recovery. The candidate artifact has no three-mode controller. All values below are explicit reconstruction choices, not author settings, and are frozen before any workload evaluation.

## Frozen configuration

- Monitor sample interval: 50 ms; EMA alpha 0.25 for GPU memory/KV usage, queue depth/delay, throughput, TTFT, TPOT.
- Pressure condition: smoothed KV usage ≥0.85 **or** queue delay ≥0.100 s for 3 consecutive samples.
- Recovery condition: KV usage ≤0.65 **and** queue delay ≤0.025 s for 5 consecutive samples.
- Any neutral sample resets the opposite persistence counter; pressure dominates recovery.
- Modes:
  - accuracy: add 1 layer/action, maximum 4 W4 layers;
  - default: add 2/action, maximum 8;
  - performance: add 4/action, maximum 16.
- Recovery removes the same per-action amount, never below zero.
- Every positive layer delta carries a coordinated `expand_kv=true`; every negative delta carries `shrink_kv_before_restore=true`.
- No policy decision uses paper evaluation outcomes for tuning.

## Gate

1. Monitor retains raw and smoothed values for every paper signal.
2. One/two pressure spikes do not trigger; the third persistent sample does.
3. Five low-pressure samples recover only after the fifth.
4. All three modes obey action size/maxima.
5. Oscillation around thresholds does not repeatedly morph.
6. Commands explicitly coordinate KV expand and shrink-before-restore.
7. Invalid/non-finite metrics and unknown modes fail closed.

## Boundary

CPU policy-core evidence only. Passing tests do not establish integrated serving behavior, mode quality/latency, or author configuration fidelity.
