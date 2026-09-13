# Completion audit: packed INT4 backend v6

Audit basis: the user objective, pre-registered v6 protocols, installed package
source, exact external checkpoint hashes, source diff, CPU/GPU tests, 5 matrix
classes, 672 checkpoint-component checks, FP16 parity, 0/8/16/32 smokes, 8
full-shape profiles, 4 model microbenchmarks, 16 frozen serving runs, 2
exploratory high-load repeats, raw request/telemetry files, regenerated
CSV/JSON/PNG outputs, and the final report. A green test or manifest is not
used as a substitute for missing experimental coverage.

## Objective restated as concrete deliverables

1. Preserve v2/v4/v5 and the valid NF4 NO-GO; make no controller, dynamic
   adaptation, runtime layer-swapping, KVResizer, scheduler, or attention-math
   optimization.
2. Verify installed vLLM 0.11.2 AWQ, AWQ-Marlin, and AWQ-Triton paths plus RTX
   3090 support for every exact Llama-3.1-8B decoder shape.
3. Freeze and record one defensible AWQ proxy configuration; quantize the exact
   base checkpoint offline once with public tooling and hash the result.
4. Reject full integration unless representative q/o, k/v, up/gate, down and
   decode/multi-row tests prove sane numerics, genuine packed execution,
   bounded workspace, storage relief, and credible performance versus FP16 and
   NF4.
5. Add the smallest selected-layer SwiftLLM backend/loading seam, keep
   unselected layers FP16, retain no hidden dense copy, use one packed
   representation for decode/prefill, and resolve measured MLP fusion overhead.
6. Prove mapping, selection, FP16 invariance, finite 0/8/16/32 generations,
   hot-path behavior, and run metadata with targeted tests.
7. Run two full-shape resource profiles and at least seven isolated
   prefill/decode samples per state; record persistent/packed/temporary memory,
   safe blocks/slots, latency, repeatability, and non-monotonicity.
8. Advance only viable candidates, then use the unchanged v5 FP16-calibrated
   64-request near-knee workload with two matched repeats; retain already-fixed
   adjacent loads when capacity makes them informative.
9. Regenerate all required latency, SLO, queue, throughput, stage timing,
   waiting/running/swapped, KV, block, swap/preemption, HBM, count, and
   variability metrics from raw evidence.
10. Classify eligibility without moving the load/metric after seeing W4. Run
    the 106-request paired quality/distortion study only if at least one state
    is eligible.
11. End with the required evidence sections and an explicit GO/NO-GO naming
    exact state/backend/config or the smallest measured blocker.

## Prompt-to-artifact checklist

| Explicit requirement / gate | Concrete evidence inspected | Result |
|---|---|---|
| Continue after static-frontier-v5 | Git history begins v6 after commit `3efe426`; v5 report/protocol read and inherited | PASS |
| Preserve v2, v4, v5 | `git diff --quiet 3efe426 --` all three prior namespaces; v6 machine audit check | PASS |
| Existing NF4 result remains NO-GO | V5 artifacts unchanged; v6 report treats NF4 only as comparator/diagnosis | PASS |
| No controller/dynamic swapping/KVResizer/scheduler change | Protocol exclusions; `git diff --quiet 3efe426 -- swiftLLM/swiftllm/server/scheduler.py`; source/file audit | PASS |
| Same checkpoint/tokenizer | Base hashes in v6 protocol and AWQ manifest equal v5 hashes | PASS |
| Inspect exact installed AWQ paths | `phase-a/backend-inventory.json` stores source paths and SHA-256 for `awq.py`, `awq_marlin.py`, `awq_triton.py`, `marlin_utils.py`, `_custom_ops.py` | PASS |
| Inspect GPU capability | Inventory and kernel result: RTX 3090, SM 8.6, 82 SMs, 24 GiB | PASS |
| Verify every real decoder shape | Installed checks pass q/o `(4096,4096)`, k/v `(4096,1024)`, up/gate `(4096,14336)`, down `(14336,4096)`, fused up+gate `(4096,28672)` | PASS |
| Prefer AWQ-Marlin | AWQ-Marlin tested first and passed | PASS |
| At most one fallback | AWQ-Triton path recorded but not measured because Marlin passed; ordinary AWQ dequant branch rejected | PASS |
| No custom kernel project | Integration calls installed `awq_marlin_repack` and `gptq_marlin_gemm`; no CUDA/Triton source added | PASS |
| Paper details checked | Paper text exposes AWQ INT4 but not group/ZP/calibration; protocol labels proxy explicitly | PASS |
| Freeze standard AWQ config | Pre-run commits `dca6a5a`, `4ff35ad`, `cdc1658`; W4, G128, asymmetric ZP, FP16, duo scaling, clipping, pinned Pile calibration | PASS |
| Public AWQ tooling | AutoAWQ 0.2.9 environment and emitted config in checkpoint manifest | PASS |
| Quantize offline once | One completed 32-layer AutoAWQ log/artifact; exporter refuses overwrite; post-export manifest-only recovery did not quantize again | PASS |
| Hash external artifact | 8 files / 5,745,260,850 bytes independently rehashed by completion audit | PASS |
| Representative q/o, k/v, MLP, down | Five shape classes in `phase-a/awq-marlin-kernel-feasibility.json` | PASS |
| Decode and multi-row counts | `M={1,16,128,1024}`, 10 warmups, 20 samples each | PASS |
| Numerical error vs same FP16 op | All finite; worst relative L2 0.1293; minimum cosine 0.9917 | PASS |
| Packed/scales/ZP/workspace bytes | Per-matrix values in `storage-results.csv`; <=25.98% FP16; workspace 328 bytes | PASS |
| One-row and prefill latency | Median/P95 for every shape/M/backend in `kernel-results.csv` | PASS |
| Compare FP16 and NF4 | Same-input timing, error, storage, and allocation rows for all three backends | PASS |
| Genuine packed hot path | Static call path is `apply_awq_marlin_linear -> gptq_marlin_gemm`; no dequant/matmul branch | PASS |
| No multi-GiB matrix workspace | Max unexplained kernel call allocation 5.13 MiB; Phase C workspace matches FP16 | PASS |
| Cheap gate before integration | Phase A result commit `329e8ee` precedes integration commit `5ebbb66` | PASS |
| Minimal backend abstraction | Two engine fields, one packed dataclass/loader, and common `linear()` dispatch; default remains NF4-compatible | PASS |
| Selected AWQ / unselected FP16 | Every smoke records per-layer type/source; 0/8/16/32 front-to-back boundaries exact | PASS |
| AWQ norms mapped correctly | Selected layer norms explicitly load from AWQ checkpoint; unselected norms load base; smoke layout records source | PASS |
| qweight/scales/ZP mapping | 224 matrices × 3 components = 672 exact key/shape/dtype checks | PASS |
| Mapping numerics | Seven matrices plus fused up+gate agree with public AutoAWQ dequantization at worst 0.000334 relative L2 | PASS |
| No hidden dense selected weights | Packed object fields and 5.374-GiB W4-32 resident allocation; source/AST audit finds no retained FP16 matrix | PASS |
| One serving representation | Marlin object has one qweight/scales/ZP set used for all row counts | PASS |
| MLP overhead measured | Separate vs fused full-shape profiles retained: 6.387 -> 4.145 GiB temporary, 1711 -> 3034 blocks | PASS |
| Correct fused `[up,gate]` | Source-key order and numerical fused check in checkpoint sanity | PASS |
| FP16 unchanged | Original-environment tests/NF4 smoke plus torch-2.9 HF parity: exact 8-token match | PASS |
| 0/8/16/32 sane generation | Four smoke JSONs, 8 finite integer tokens each | PASS |
| Metadata records backend/config | Every AWQ serving metadata has bits/group/ZP, config/index hashes, packed bytes, implementation, no-dequant flag | PASS |
| Scheduler/attention/KV/timing semantics unchanged | Scheduler source unchanged; attention arguments changed only to ABI-neutral keywords/bundled same FA2; FP16 parity passed; metric code reused | PASS |
| Two full-shape profiles/state | Eight committed Phase C profile JSONs | PASS |
| Profile shape identical to v5 | 32 sequences, 49,152 tokens, block 16, utilization .99 | PASS |
| Persistent/packed/temporary memory | `resource-performance-gate.csv` and raw profiles | PASS |
| Safe blocks/token slots | Exact 1768/3034/4286/6766 blocks and 28,288/48,544/68,576/108,256 slots | PASS |
| Seven post-warmup prefill/decode samples | Four micro JSONs, each seven values | PASS |
| Non-monotonic capacity investigated | Capacity is monotonic after MLP fusion; failed separate MLP/QKV alternatives retained and explained | PASS |
| Meaningful resource relief | +71.6%, +142.4%, +282.7%; 0% profile CV; all states pass Phase C | PASS |
| W4-32 always tested | Kernel, profile, microbenchmark, and full serving matrix present | PASS |
| Reuse FP16-only calibrated load | Exact v5 workload files/hashes; no new load selected | PASS |
| Frozen near-knee 64 requests | Scale 6, 0.6667 nominal RPS, 64 exact requests | PASS |
| Two repeats per state | 8 scale-6 runs: FP16 and every AWQ state rep0/rep1 | PASS |
| Neighboring low/high loads | Pre-frozen scale 8/4 run for all four states | PASS |
| Every serving request complete | 16 × 64 = 1024/1024 completed at exact 1024/512 | PASS |
| TTFT P50/P95/P99 and strict SLO | 16 complete rows in `analysis/serving_runs.csv` and final report | PASS |
| Queueing, throughput, prefill/decode | Raw-derived distributions and completed/token throughput in same table/report | PASS |
| Waiting/running/swapped | P50/P95/max per run in CSV/report | PASS |
| KV, blocks, swaps/preemptions, HBM | Every telemetry stream and derived row covers all fields | PASS |
| Variability/noise | `serving_variability.csv`; matched eligibility rows; 0.111-s frozen P95 noise floor | PASS |
| Telemetry supports mechanism | AWQ near-knee has much lower KV and zero preemptions, but slower first-prefill/P95/SLO | PASS |
| Eligibility criteria applied | `state_eligibility.csv`: all resource/mechanism gates pass, repeated P95/SLO advantage fails | PASS |
| No favorable adjacent-load substitution | W4-16 high-load repeats explicitly exploratory; decision still uses scale 6 | PASS |
| Bounded backend tuning only | Reduction and QKV protocols/results; neither advanced; default source restored | PASS |
| Quality only if eligible | No eligible state; no AWQ 106-request quality run; report labels quality blocked rather than borrowing NF4 | PASS / not applicable |
| Explicit final decision | `analysis/backend_decision.json` and final report both say NO-GO, empty runtime state set | PASS |
| Smallest blocker named | Multi-request prefill speed at synchronized 2-second boundary; not workspace/decode/controller | PASS |
| Required final sections | Confirmed, supported-but-uncertain, blocked, remaining uncertainty, backend decision | PASS |
| Exact regeneration | Regenerated serving CSV/JSON byte-for-byte from raw requests/telemetry; `regenerate.sh` provided | PASS |
| Machine audit covers objective | 183 checks include external hashes, raw protocol, telemetry, profiles, mapping, parity, preservation, sections, and decision | PASS |

## Key negative and positive evidence

The backend mechanism itself is substantially stronger than NF4: W4-8 already
has 71.6% more safe blocks, profile workspace equals FP16, and isolated decode
is non-regressive. W4-16 also beats saturated FP16 P95 in two exploratory
scale-4 repeats (2.343/3.556 s versus 15.779/13.207 s) with lower KV and no
preemption.

The fixed eligibility point is nevertheless negative. At scale 6, matched P95
reductions are -0.120/-0.074 s (W4-8), -0.157/-0.109 s (W4-16), and
-0.285/-0.245 s (W4-32); SLO reductions are also negative in every repeat.
Moving the decision to the favorable scale-4 result would violate the prompt.

## Machine audit and validation

Command:

```bash
PYTHONPATH="$PWD/swiftLLM" /nfs/home/s314511048/.venv/bin/python \
  -m benchmark.audit_packed_backend \
  --root benchmark-results/packed-int4-backend-v6 \
  --output benchmark-results/packed-int4-backend-v6/completion_audit.json
```

Observed: **PASS**, 183 checks, 0 failures, decision **NO-GO**.

Final validation also observed:

- legacy torch-2.5 suite: 16 tests passed;
- torch-2.9/vLLM suite: 16 tests passed;
- Python compilation: passed;
- v5 regeneration/audit: PASS, 157 checks, NO-GO (unchanged);
- v6 regeneration/audit: PASS, 183 checks, NO-GO;
- `git diff --check`: passed;
- old artifact and scheduler diffs from `3efe426`: empty.

## Audit conclusion

All explicit work and evidence gates are covered. The scientifically defensible
answer is not that AWQ failed like NF4: AWQ-Marlin fixes packed storage,
workspace, capacity, and decode. It still fails the exact predeclared runtime
eligibility gate because multi-request prefill shifts P95/SLO the wrong way at
the frozen near-knee workload. The goal therefore completes with **BACKEND
DECISION: NO-GO** and no controller implementation.
