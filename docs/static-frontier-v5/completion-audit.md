# Completion audit: static frontier v5

Audit basis: the actual repository state, frozen v5 protocol and calibration
decision, 4 quality runs, 16 final serving runs, 8 resource profiles, 4
microbenchmarks, raw JSONL, derived CSV/JSON/PNG, regeneration commands, test
output, and Git diff/status. A passing test or manifest is not treated as a
substitute for experimental coverage.

## Objective restated as concrete success criteria

1. Preserve v2/v4 and create a separate v5 namespace; make no dynamic,
   layer-swapping, controller, or scheduler implementation.
2. Characterize quality independently on at least 100 fixed DuReader requests
   for FP16/W4-8/W4-16/W4-32 under sequential/low-load 1024/512 greedy
   generation, with all request-level paired F1 deltas and paired uncertainty.
3. Diagnose non-monotonic F1 without data selection, using a predeclared
   secondary quantization-distortion measure.
4. Calibrate load with FP16 only, freeze a small set around the local knee, and
   run every state on the exact same >=50-request workloads. Repeat every state
   at near-knee and add a third repeat only if ordering changes.
5. Report every run's TTFT P50/P95/P99, strict >2-second SLO rate, queueing,
   prefill/decode timing, completed throughput, queue depths, KV capacity and
   occupancy, swap/preemption, HBM, request count, and variability.
6. Measure each state's persistent model allocation, peak temporary prefill
   workspace, safe KV blocks/token slots, isolated prefill/decode latency, and
   serving saturation point.
7. Classify all four states ELIGIBLE/INELIGIBLE using stable resource and
   same-workload performance evidence—not quantized-layer count alone.
8. Make an explicit GO/NO-GO decision. GO requires a repeated W4 latency/SLO
   advantage exceeding noise with an identified relief mechanism plus quality
   distinction/distortion evidence.
9. Save raw-derived tables/plots, executable regeneration commands, a final
   report with all required evidence sections, and this completion audit.

## Prompt-to-artifact checklist

| Requirement / gate | Evidence inspected | Result |
|---|---|---|
| Current validated substrate after `9b3814a7` | Git history; v5 run metadata; v4 `fp16-hf-parity.json` | PASS |
| Do not reopen passed FP16 parity absent falsification | v5 protocol inherits v4 parity; no parity rerun or parity-source change | PASS |
| Preserve v2 and v4 | Both namespaces exist; `git diff 9b3814a7 -- benchmark-results/static-quantization-quality-latency benchmark-results/table5-substrate-v4` has no tracked changes | PASS |
| Separate new namespace | All new experiment artifacts are under `benchmark-results/static-frontier-v5/` and `docs/static-frontier-v5/` | PASS |
| No dynamic adaptation, swapping controller, or scheduler implementation | Source diff contains only workload preparation, runner modes/warmup, analysis, audit, tests, and docs; scheduler diff is empty | PASS |
| Four fixed precision states | Protocol and raw metadata cover `fp16_0`, `w4_8`, `w4_16`, `w4_32` | PASS |
| Same checkpoint/tokenizer/quantizer/layer order | `protocol_manifest.json`; matching model/workload/manifest hashes in every final run | PASS |
| 1024 prompt / 512 output / greedy policy | Every one of 424 quality and 1024 serving request rows reports 1024/512; run metadata records greedy fixed-step generation | PASS |
| Sequential low-load quality | Four quality metadata files report `arrival_mode=sequential`; P95 queueing is about 0.016 s | PASS |
| Substantially more than 16 quality requests | 106/106 complete per state | PASS |
| Primary DuReader character F1 | `analyze_static_frontier.py`; `quality_table.csv`; 106-row paired table | PASS |
| Every per-request paired W4-FP16 difference | `analysis/quality_paired_differences.csv` has 106 rows and three delta columns | PASS |
| Paired uncertainty | 10,000-resample paired percentile-bootstrap 95% CIs in `quality_table.csv` | PASS |
| Do not force monotonic quality | Means remain 14.471/14.230/14.825/14.977; no records removed; all paired CIs include zero | PASS |
| Secondary metric predeclared before W4 | Protocol commits `41e38b8`, `f1913ea`, `914ef8e`; output-token agreement definition in manifest | PASS |
| Secondary distortion evidence kept separate | `quality_output_agreement.csv` and report subsection explicitly label mechanism-only role | PASS |
| FP16-only calibration first | Six calibration run metadata files all have `condition=fp16_0`; calibration/freeze commit `914ef8e` precedes v5 W4 runs | PASS |
| Cheap diagnosis before extension | Initial invariant SLO/low-queue anomaly documented; only FP16 scales 6/4 added before W4 | PASS |
| Freeze intensities before W4 | `calibration_decision.json` and manifest fix scales 8/6/4; all W4 metadata hashes that frozen manifest | PASS |
| Knee supported by pressure, not TTFT alone | Scale 6 first crosses peak KV 0.85 (0.998); scale 4 has P95 queue 10.413 s | PASS |
| Same final loads for all states | Required 4-state × scale `{8,6,4}` matrix present; workload hash is identical across states at each scale | PASS |
| >=50 final requests; <=~2 pp/request | 64/64 complete per run; one request = 1.5625 percentage points | PASS |
| Preserve final 1024/512 near-knee protocol | Both scale-6 repeats for all states have 64 complete 1024/512 rows | PASS |
| Repeat near-knee twice | 8 scale-6 runs: rep0/rep1 for each state | PASS |
| Third repeat rule | P95 ordering `FP16 < W4-32 < W4-16 < W4-8` identical twice; third repeat correctly not triggered | PASS |
| TTFT P50/P95/P99 per run | 16 rows in `analysis/serving_runs.csv`; full table in final report | PASS |
| Strict TTFT >2 s SLO | 16 per-run counts/rates regenerated from raw TTFT; denominator 64 | PASS |
| Queueing distribution | Per-run P50/P95/P99 and telemetry distributions in `serving_runs.csv`/report | PASS |
| Prefill/decode timing | Serving first-prefill-to-output and TPOT distributions plus isolated microbenchmarks | PASS |
| Completed throughput | Request and generated-token throughput in `serving_runs.csv` | PASS |
| Waiting/running/swapped queue depths | P50/P95/P99/max from every telemetry stream; report includes full table | PASS |
| KV utilization and available blocks | Every run reports distribution/peak plus fixed blocks/token slots | PASS |
| Swap/preemption events | Per-run swap-in/out/preemption counts in raw telemetry and derived table | PASS |
| Physical HBM | Peak used/minimum free GiB derived for every run | PASS |
| Request count and variability | 64/64 every run; `serving_variability.csv`; near-knee repeat ranges | PASS |
| Persistent model allocation | Two profiles/state; `resource_mechanism_table.csv` | PASS |
| Peak temporary prefill workspace | Raw profile peak minus persistent allocation, twice/state | PASS |
| Safe KV blocks/token slots | Fresh profiles reproduce 1768/312/568/2040 blocks and derived slots | PASS |
| Isolated 1024 prefill / one-token decode | Seven samples/state in `micro-*.json`; median/P95 in resource table | PASS |
| Saturation point from serving sweep | FP16 observed at 0.667 RPS; all W4 at/below lowest tested 0.5 RPS | PASS |
| W4-8/W4-16 pathology explicitly checked | 312/568 blocks versus FP16 1768; full KV, long queues, repeated swaps | PASS |
| Every state classified | `state_eligibility.csv`: FP16 ELIGIBLE; W4-8/16/32 INELIGIBLE | PASS |
| W4-32 not admitted by layer count/resource alone | +15.4% blocks but worse P95/SLO twice and slower microbench; INELIGIBLE | PASS |
| Advantage exceeds noise gate | W4-32 mean near-knee P95 penalty 13.423 s vs 2.186-s run range; no favorable advantage | PASS |
| Explicit GO/NO-GO | `dynamic_decision.json` and report both say `NO-GO` | PASS |
| Smallest blocker named | Low-workspace packed W4 prefill plus non-regressive decode backend required before controller work | PASS |
| Required quality table | `analysis/quality_table.csv` | PASS |
| Required latency/SLO load curves | `analysis/latency_slo_vs_load.png`; image opened and inspected | PASS |
| Required resource/mechanism table | `analysis/resource_mechanism_table.csv` | PASS |
| Required state-eligibility table | `analysis/state_eligibility.csv` | PASS |
| Recommended state set if GO | Decision is NO-GO; machine artifact correctly emits an empty recommended runtime set | PASS / not applicable |
| Runnable regeneration commands | executable `regenerate.sh`, `execution_commands.json`, and final report commands | PASS |
| Required final sections | Final report has CONFIRMED, SUPPORTED BUT UNCERTAIN, BLOCKED, REMAINING UNCERTAINTY, and DYNAMIC ADAPTATION DECISION | PASS |
| Raw-artifact derivation | Analyzer reads quality/serving requests and telemetry plus profile/micro JSON; no result constants | PASS |
| Raw summaries independently reproducible | Machine audit exactly regenerates all 16 serving summaries from raw files | PASS |
| Completion audit covers objective rather than proxy status | `audit_static_frontier.py` checks hashes, full run matrix, raw request protocol, telemetry fields, derived table semantics, report sections, and decision | PASS |

## Machine audit coverage and result

Command:

```bash
PYTHONPATH="$PWD/swiftLLM" /nfs/home/s314511048/.venv/bin/python \
  -m benchmark.audit_static_frontier \
  --root benchmark-results/static-frontier-v5 \
  --output benchmark-results/static-frontier-v5/completion_audit.json
```

Observed result: **PASS**, 157 checks, decision **NO-GO**.

The audit checks actual raw request/telemetry rows and exact hashes. It does not
infer completion from a green manifest. It verifies the 4 × 106 quality matrix,
16-run serving matrix, every 64-request completion, final workload identity,
1024/512 protocol, warmup exclusion, required telemetry, exact re-summarization,
near-knee ordering, 8-profile/4-microbenchmark mechanism matrix, table row
coverage, plots/files, eligibility classifications, report sections, commands,
and explicit decision.

## Validation commands and observed results

```bash
PYTHONPATH="$PWD/swiftLLM:$PWD/swiftLLM/csrc" \
  /nfs/home/s314511048/.venv/bin/python -m unittest \
  benchmark.test_benchmark benchmark.test_inference_substrate -v
# Ran 13 tests ... OK

/nfs/home/s314511048/.venv/bin/python -m py_compile \
  swiftLLM/benchmark/*.py \
  swiftLLM/swiftllm/model_config.py \
  swiftLLM/swiftllm/worker/model.py \
  swiftLLM/swiftllm/worker/weight.py \
  swiftLLM/swiftllm/worker/kernels/linear.py \
  swiftLLM/swiftllm/worker/layers/post_layer.py
# PASS

git diff --check
# PASS
```

The final Git/status audit is performed after committing result artifacts. Raw
`requests.jsonl`, `telemetry.jsonl`, and logs remain intentionally ignored by
Git but are present in the v5 namespace and were consumed by the machine audit.

## Audit conclusion

Every explicit deliverable is covered by current raw or source evidence. The
static frontier is stable enough to answer the scientific gate: the present
NF4 substrate has quantization distortion but no eligible pressure-relief
state. The completed objective therefore ends with **NO-GO**, not with a
controller implementation.
