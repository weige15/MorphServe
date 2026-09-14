# Onboarding

## What This Project Does

MorphServe is a research workspace built around a vendored SwiftLLM inference engine. It measures open-loop arrivals, queueing, TTFT, TPOT, KV pressure, and runtime precision behavior. V7 validates the static `{FP16, AWQ-W4-16}` crossover, v8 implements state-preserving one-process FP16↔AWQ morphing, and v9 confirms causal high-pressure entry but fails useful release. Release-side v10 demonstrates repeated useful restoration and exact two-cycle operation on held-out workloads; its formal decision remains **NO-GO** only because the preregistered 0.98 throughput non-inferiority margin narrowly failed.

## Quickstart

Validated assumptions:

- Legacy/FP16-NF4 Python: `/nfs/home/s314511048/.venv/bin/python` (3.12.3, torch 2.5.1).
- AWQ-Marlin Python: `/nfs/home/s314511048/.venvs/morphserve-vllm0112/bin/python` (torch 2.9.0, vLLM 0.11.2).
- Model: local Meta Llama 3.1 8B snapshot documented in `docs/phase-1/baseline-configuration.md`.
- Offline AWQ artifact: `/nfs/home/s314511048/.cache/morphserve/llama31-8b-autoawq-w4-g128-zp` (hash manifest in the v6 results).
- GPU: an available RTX 3090; v6 used physical CUDA device 5.
- Torch-2.9 SwiftLLM extension: `/nfs/home/s314511048/.cache/morphserve/swiftllm-c-torch29/lib`.

Run a small streaming benchmark from `swiftLLM/`:

```bash
cd swiftLLM
MODEL=/nfs/home/s314511048/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
VENV=/nfs/home/s314511048/.venv
PYTHONPATH="$PWD:$PWD/csrc" CUDA_VISIBLE_DEVICES=3 \
  "$VENV/bin/python" -m benchmark.run \
  --model-path "$MODEL" --target-rps 0.5 --arrival-mode fixed \
  --request-count 4 --prompt-token-count 8 --output-token-count 4 \
  --seed 2025 --telemetry-interval-s 0.1 \
  --output-dir ../benchmark-results/phase-1/baseline-low-load/runs --run-id fixed-arrivals
```

Run the tests from the repository root:

```bash
VENV=/nfs/home/s314511048/.venv
PYTHONPATH="$PWD/swiftLLM:$PWD/swiftLLM/csrc" "$VENV/bin/python" -m unittest benchmark.test_benchmark benchmark.test_inference_substrate benchmark.test_crossover benchmark.test_runtime_morphing benchmark.test_closed_loop_controller -v

AWQ_ENV=/nfs/home/s314511048/.venvs/morphserve-vllm0112
AWQ_EXT=/nfs/home/s314511048/.cache/morphserve/swiftllm-c-torch29/lib
PYTHONPATH="$AWQ_EXT:$PWD/swiftLLM" "$AWQ_ENV/bin/python" -m unittest benchmark.test_benchmark benchmark.test_inference_substrate benchmark.test_crossover benchmark.test_runtime_morphing benchmark.test_closed_loop_controller -v
```

Regenerate v10 analysis/tests and independently audit all saved raw evidence:

```bash
benchmark-results/release-side-runtime-v10/regenerate.sh
```

A successful artifact audit ends with `status: PASS`; its scientific decision remains `NO-GO` by design. The preregistered analyzer had a disclosed percentile-API correction; see `analysis_correction.json`. Reproducing raw v10 runs requires checkout of preregistration commit `8e7675f`, while routine regeneration uses the current result commit. V9, v8, v7, and v6 remain independently reproducible with their existing scripts. Do not rerun broad historical serving matrices for routine checks.

The complete final sweep command is preserved in [`docs/phase-1/kv-admission-sweep-report.md`](../docs/phase-1/kv-admission-sweep-report.md); it requires a CUDA GPU and is substantially more expensive than the smoke run.

## Important Files

| Path | Role |
|---|---|
| `swiftLLM/swiftllm/server/engine.py` | Engine observations plus serialized explicit runtime transition requests. |
| `swiftLLM/swiftllm/server/scheduler.py` | Strict-FCFS scheduler plus the restoration-only admission/swap-in hold. |
| `swiftLLM/swiftllm/worker/model.py` | Prepared weight transitions, segmented KV resize/remap, swapping, and transition traces. |
| `swiftLLM/benchmark/run.py` | Open-loop benchmark entry point. |
| `swiftLLM/benchmark/workload.py` | Fixed and Poisson arrival schedules. |
| `swiftLLM/benchmark/summarize.py` | Regenerates one run summary from raw files. |
| `swiftLLM/benchmark/analyze.py` | Historical short-workload aggregation. |
| `swiftLLM/benchmark/final_analyze.py` | Final KV-pressure aggregation and report generation. |
| `swiftLLM/benchmark/prepare_static_quantization_workload.py` | Builds the frozen workload for the static quantization quality–latency benchmark. |
| `swiftLLM/benchmark/run_static_quantization_condition.py` | Runs one static quantization condition and records raw evidence. |
| `swiftLLM/benchmark/analyze_static_quantization_quality_latency.py` | Derives historical v2/v4 quality/latency tables and plots. |
| `swiftLLM/benchmark/prepare_static_frontier_workloads.py` | Re-derives v5 quality/calibration/serving inputs from the frozen 106-row workload. |
| `swiftLLM/benchmark/analyze_static_frontier.py` | Regenerates v5 paired quality, serving curves, mechanism, eligibility, and decision artifacts. |
| `swiftLLM/benchmark/audit_static_frontier.py` | Verifies preserved v5 raw files, hashes, run matrix, summaries, and deliverables. |
| `swiftLLM/benchmark/awq_marlin_feasibility.py` | Shape-complete packed-kernel numerical, memory, and timing gate. |
| `swiftLLM/benchmark/quantize_awq_checkpoint.py` | One-time public AutoAWQ checkpoint exporter and hash manifest writer. |
| `swiftLLM/benchmark/awq_checkpoint_sanity.py` | Audits all packed components and actual-checkpoint Marlin numerics. |
| `swiftLLM/benchmark/analyze_packed_backend.py` | Regenerates v6 serving tables, plots, eligibility, and decision. |
| `swiftLLM/benchmark/audit_packed_backend.py` | Performs the v6 183-check completion audit. |
| `swiftLLM/benchmark/prepare_crossover_workloads.py` | Deterministically creates the frozen v7 nine-load grid. |
| `swiftLLM/benchmark/crossover_prefill_microbenchmark.py` | Measures integrated prefill over multi-request GEMM M. |
| `swiftLLM/benchmark/run_crossover_plan.py` | Executes/resumes the immutable 36-run matrix serially. |
| `swiftLLM/benchmark/analyze_crossover.py` | Regenerates per-run metrics, mechanism tables, causal signals, and the frozen crossover decision. |
| `swiftLLM/benchmark/audit_crossover.py` | Verifies preserved v7 raw coverage, analysis, and decision gates. |
| `swiftLLM/benchmark/closed_loop_controller.py` | Frozen v7/v9 sustained-pressure entry state machine using exact time-weighted helpers. |
| `swiftLLM/benchmark/release_intent_controller.py` | V10 causal pressure-drop release intent; entry delegates to the frozen controller. |
| `swiftLLM/benchmark/analyze_release_latency.py` | Replays v9 release barriers as development evidence. |
| `swiftLLM/benchmark/prepare_release_workloads.py` | Freezes alternate-content low-only, one-cycle, and two-cycle v10 inputs. |
| `swiftLLM/benchmark/run_release_condition.py` | V10 same-envelope runner with drain and NVML telemetry. |
| `swiftLLM/benchmark/run_release_plan.py` | Executes/resumes the immutable 16-cell v10 matrix. |
| `swiftLLM/benchmark/analyze_release_side.py` | Regenerates phase, transition, recovery, throughput, environment, and decision artifacts. |
| `swiftLLM/benchmark/audit_release_side.py` | Independently replays intent and audits raw request/KV/provenance evidence. |
| `swiftLLM/benchmark/run_closed_loop_condition.py` | Same-envelope static/Dynamic runner with controller, transition, and per-token precision traces. |
| `swiftLLM/benchmark/run_closed_loop_plan.py` | Executes/resumes the immutable 18-run matrix with preserved attempt lineage. |
| `swiftLLM/benchmark/analyze_closed_loop.py` | Regenerates v9 performance, quality, fidelity, timelines, gates, and decisions. |
| `swiftLLM/benchmark/audit_closed_loop.py` | Checks frozen hashes, raw structure/state, named outputs, and decision coverage. |
| `swiftLLM/benchmark/verify_closed_loop_results.py` | Post-result independent recomputation of raw metrics, gates, quality CI, fidelity, and decisions. |
| `swiftLLM/benchmark/test_closed_loop_controller.py` | Boundary/state-machine checks and exact v7 telemetry replay. |
| `swiftLLM/benchmark/run_runtime_morphing_validation.py` | Generates v8 static, mid-request, capacity, cost, and cycle evidence. |
| `swiftLLM/benchmark/test_runtime_morphing.py` | CPU contract and opt-in CUDA segmented-KV/allocator/swap tests. |
| `benchmark-results/release-side-runtime-v10/` | Frozen inputs/manifest, 16 held-out raw runs, exact release/drain timelines, GPU telemetry, analysis, commands, and audit. |
| `docs/release-side-runtime-v10/` | V9 delay diagnosis, v10 protocol, final report, and completion audit. |
| `benchmark-results/closed-loop-runtime-v9/` | Preserved frozen inputs/manifest, 18 raw runs, analysis, timelines, commands, logs, and audit. |
| `docs/closed-loop-runtime-v9/` | V9 protocol, final report, and exhaustive completion audit. |
| `benchmark-results/runtime-morphing-v8/` | Preserved five raw runs, nine transitions, diagnostics, tables, logs, commands, and audit. |
| `docs/runtime-morphing-v8/` | Preserved transition contract, ownership, trace format, final report, and audit. |
| `benchmark-results/fp16-awq-crossover-v7/` | Preserved protocol, raw batch/request/pressure streams, multi-M evidence, analysis, and 28-check audit. |
| `docs/fp16-awq-crossover-v7/` | Preserved protocol, diagnosis, final report, and completion audit. |
| `benchmark-results/packed-int4-backend-v6/` | Preserved backend validation, checkpoint manifest, profiles, raw serving, analysis, and audit. |
| `docs/packed-int4-backend-v6/` | Preserved v6 phase reports, bounded follow-ups, final report, and completion audit. |
| `benchmark-results/static-frontier-v5/` | Preserved NF4 raw quality/serving runs, mechanism probes, and derived outputs. |
| `docs/static-frontier-v5/` | Preserved v5 protocol, final report, and completion audit. |
| `benchmark-results/phase-1/` | Phase 1 raw runs and derived tables/figures, grouped by experiment. |
| `docs/phase-1/` | Current Phase 1 documentation. |
| `docs/static-quantization-quality-latency/` | Static quantization benchmark protocol, report, and completion audit. |
| `references/` | Source reference material. |

## Architecture Map

Serving benchmarks launch requests on saved absolute offsets without waiting for prior completion. SwiftLLM's control plane (`Engine` and `Scheduler`) admits and batches requests, while the data plane executes model work and maintains KV. Batch observation records pre-schedule pressure, admitted prefill/decode shape, effective M, duration, state, safe/used blocks, and swaps; request and periodic streams are analyzed only after the run.

The v8 control plane accepts explicit `morph_to_awq_w4_16()` and `restore_to_fp16()` calls between complete forwards. Selected layers load from pinned host layouts; only one representation is active in HBM. The base K/V segment has 1,759 blocks, AWQ adds 2,411 physical blocks for a safe total of 4,170, and restoration drains/remaps before publishing the base capacity.

V9 keeps policy outside the scheduler. A 0.25-s benchmark coroutine observes causal pressure, evaluates exact v7 time-weighted windows against a separate 1,768-block policy reference, and latches Engine transitions. V10 preserves entry but releases after a 15-second AWQ-only history shows a five-second quiet queue and at least 20% scheduler-used pressure drop versus the prior ten seconds. Release intent does not inspect physical safety. The Engine pauses prefill/swap-in, continues active decode until allocation fits 1,759, then uses the unchanged v8 shrink/restore path. Exact lifecycle and one-second NVML streams separate detection, drain, hot restore, resume, catch-up, and environment.

The safe dynamic total is 4,170 blocks. A 4,286-block diagnostic allocation completed but exceeded the 0.99 maximum-shape memory envelope, so it is not advertised as safe. See `docs/runtime-morphing-v8/state-memory-ownership.md` before changing capacity.

## Development Workflow

1. Read `docs/release-side-runtime-v10/final-report.md`, its completion audit, then the v10 protocol/diagnosis and preserved v9/v8/v7 evidence.
2. Treat v9 and v10 as completed frozen NO-GO experiments. Do not add selective repeats or alter frozen thresholds/workloads/gates.
3. Preserve `{FP16, AWQ-W4-16}`, the 1,768 policy reference, 1,759/4,170 physical capacities, serialization, and physical-before-scheduler resize ordering.
4. Preserve all v2–v10 raw evidence; any follow-up needs a separately preregistered namespace.
5. Do not waive v10's narrow throughput miss, resolve v7 ambiguous scales, or explain v9 static-AWQ variance with post-hoc runs.
6. Raw v10 provenance belongs to commit `8e7675f`; the result commit contains the disclosed analyzer correction. Never rewrite raw metadata.
7. Run CPU checks, opt-in CUDA tests when changing runtime code, compilation, the relevant analyzer/audit, and `git diff --check` before handoff.

## Testing

The validated CPU-only checks are:

```bash
VENV=/nfs/home/s314511048/.venv
PYTHONPATH="$PWD/swiftLLM:$PWD/swiftLLM/csrc" "$VENV/bin/python" -m unittest benchmark.test_benchmark benchmark.test_inference_substrate benchmark.test_crossover benchmark.test_runtime_morphing benchmark.test_closed_loop_controller -v
PYTHONPATH="$PWD/swiftLLM:$PWD/swiftLLM/csrc" "$VENV/bin/python" -m py_compile swiftLLM/benchmark/*.py swiftLLM/swiftllm/server/*.py swiftLLM/swiftllm/worker/*.py swiftLLM/swiftllm/worker/layers/*.py swiftLLM/swiftllm/worker/kernels/*.py
git diff --check
```

A successful v10 artifact validation has 16 selected runs, `analysis/run_validation.json` and `completion_audit.json` reporting `PASS`, six independently replayed release intents, and `analysis/decision.json` reporting the scientific `NO-GO`. The CUDA suite must run with `MORPHSERVE_RUN_CUDA_TESTS=1`; otherwise hardware cases intentionally skip. V9, v8, v7, v6, and v5 remain independently reproducible with their existing audits.

## Troubleshooting

- **Missing `vllm_flash_attn`:** install the validated wheel with `pip install --no-deps 'vllm-flash-attn==2.6.2'` in the legacy environment. The torch-2.9 AWQ environment uses vLLM's bundled FA2 fallback and has a fresh FP16 parity artifact.
- **`vllm._C` undefined c10 symbol:** vLLM 0.11.2 is ABI-incompatible with the legacy torch-2.5 environment. Use `/nfs/home/s314511048/.venvs/morphserve-vllm0112`; do not upgrade the legacy environment in place.
- **AWQ checkpoint missing:** restore the external path from `phase-b/awq-checkpoint-manifest.json`. Do not casually rerun the 27-minute quantization or overwrite the hashed artifact.
- **AWQ partial state loses KV capacity:** verify fused packed up+gate is active. Separate up/gate recreates a 6.39-GiB activation peak even without weight dequantization.
- **C++ extension build failure with editable install:** use `cd swiftLLM/csrc && $VENV/bin/python setup.py build_ext --inplace`, then return to the repository root. The upstream isolated `pip install -e csrc` path is known to fail in the validated environment.
- **No model or CUDA device:** use the local model snapshot and `CUDA_VISIBLE_DEVICES=3`, or stop; CPU execution is not a substitute for the Phase 1 result.
- **Need to inspect a failure:** check the run-specific `.log` file under the experiment's `logs/` directory and the raw files under its `runs/<run-id>/` directory. Do not treat incomplete runs as experiment evidence.
- **Unexpected v7 strict-SLO ordering:** synchronized multi-request service itself is near two seconds. Inspect P95 TTFT, queue area, KV dwell, preemption, and throughput; the strict threshold is reference-only.
- **V7 trigger fires from preemption alone:** this is the rejected candidate. Use persistent compound queue/KV or capacity-margin/queue-integral signals from the final report.
- **W4-8/W4-16 capacity collapse:** this is the measured bitsandbytes multi-row prefill workspace effect, not larger resident weights. See `docs/static-frontier-v5/final-report.md` before changing memory settings.
- **Dynamic target set to 4,286:** do not treat static capacity as runtime-safe. V8's diagnostic exceeded the 0.99 maximum-shape envelope; retain 4,170 unless a full probe establishes a new maximum.
- **Direct restore says it cannot drain:** start the engine loop so existing requests can complete while admissions are held, or wait until allocated blocks fit base. Never force-release extension tensors.
- **CUDA morphing tests skip:** set `MORPHSERVE_RUN_CUDA_TESTS=1` and use the torch-2.9 AWQ environment plus rebuilt extension.
- **Raw provenance mismatch:** frozen v8/v9 evidence intentionally fails when source changes. Do not rewrite completed evidence; use the recorded commit or create a new pre-registered namespace.
- **V9 release appears too late:** this is the measured result. Repeat 0 never released; repeat 1 restored after the final arrival. Do not tweak release or extend the frozen workload in v9.
- **V10 analyzer source differs from its manifest:** this is the bounded post-result percentile-API correction, not controller drift. Verify `analysis_correction.json`; only the analyzer and audit hashes may differ.
- **V10 raw plan refuses to resume on the result commit:** expected fail-fast behavior because analyzer/audit source hashes changed after the disclosed correction. Checkout `8e7675f` for raw reproduction; use `regenerate.sh` on the result commit.
- **V10 throughput looks nearly equal but decision is NO-GO:** the frozen 0.98 margin fails at 0.97778/0.97774 and one-cycle aggregate 0.97944. Do not waive or selectively repeat it.
- **V9 static AWQ headline variance:** repeat P95 is 21.395/2.253 s. Both raw runs are valid and must remain; no selective third repeat belongs in v9.

## Documentation Freshness Checklist

- [ ] README quickstart still works.
- [ ] Run commands match the current code.
- [ ] Test commands match the current code.
- [ ] Important files list is still accurate.
- [ ] Architecture map matches the current implementation.
- [ ] Troubleshooting section includes recent known failures.
