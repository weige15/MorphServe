# Normalized candidate FP16 baseline result

## Result

Attempt 3 passed every predeclared no-morph gate on the available Llama 3.1 8B base snapshot:

- finite logits, identical vocabulary shape (`128256`);
- exact top-1 token `505` and identical top-5 `[505,11,304,627,323]`;
- relative L2 logit error `0.0020401 < 0.005`;
- maximum absolute error `0.0234375` (reported, not a scale-independent gate);
- 291/291 source tensors exact after FP16 conversion, including separately checked up/gate slices in contiguous storage;
- process exit 0 and immutable vendor manifests clean.

The normalized candidate path used `ignore_kvcache=True`, no quantized layer, no controller, and no morphing.

## Classification

**Approximate/modified-condition FP16 behavior reproduced.** This validates the reconstructed no-morph numerical base for one local checkpoint/prompt on RTX 3090. It is not the paper's exact Llama 3 checkpoint/hardware, does not validate serving/KV behavior, and cannot establish headline latency or quality.

## Resource/timing caveat

Another user's process occupied about 6.9 GiB and 100% utilization on physical GPU 0 both before and after the run. The parity gate is numerical, but all wall timings are contaminated and must not be used as performance evidence. Process-local peaks were 16.07 GB (Transformers) and 16.17 GB (candidate). Total experiment runner wall time was 61.2 s, with candidate load+single forward 17.2 s; these include loading and are not steady-state.

## Prior failed attempts

- Attempt 1: inherited generic `MODEL_PATH` collision; no model loaded.
- Attempt 2: eager optional `evaluate` import; Transformers ran, candidate model did not load.
- Attempt 3: changed hypotheses fixed; no unchanged failure was retried.

## Evidence

- `results-attempt-3/metrics.json`
- `results-attempt-3/run.{stdout,stderr}.log`, `run.exitcode`
- `results-attempt-3/nvidia-{before,after}.csv`
- `runtime/candidate-python/PROVENANCE.md`
