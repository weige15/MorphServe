# Reconstructed monitor/controller core result

Seven CPU tests pass for the frozen reconstruction:

- the machine-readable frozen config exactly matches code constants and all three mode definitions;
- all seven paper signals are retained raw and EMA-smoothed;
- pressure triggers only on the third persistent sample;
- recovery triggers only on the fifth qualifying sample;
- accuracy/default/performance use 1/2/4 layers per action with maxima 4/8/16;
- alternating threshold noise does not accumulate pressure;
- positive commands coordinate KV expansion and negative commands require shrink-before-restore;
- invalid samples/modes fail closed.

## Classification

**Approximate/modified-condition policy-core implementation.** The paper and candidate artifacts do not specify these full mode configurations; they are frozen reconstruction choices, not recovered author settings. CPU policy tests do not establish integrated request scheduling, GPU behavior, or latency/quality improvements.

Next gate: rerun the repaired atomic coordinator/executor on a real GPU, then connect measured samples to a scheduled workload only after those invariants pass.
