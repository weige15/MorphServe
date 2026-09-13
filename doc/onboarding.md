# Onboarding

## What This Project Does

MorphServe is a research workspace built around a vendored SwiftLLM inference engine. It measures open-loop request arrivals, queueing, TTFT, TPOT, KV occupancy, admission pressure, and mixed-precision behavior. V6 validates offline AutoAWQ plus vLLM AWQ-Marlin, v7 validates the static `{FP16, AWQ-W4-16}` crossover, and v8 implements a manual one-process FP16↔AWQ substrate with pinned weight staging and segmented physical KV resizing. Active requests survive both directions without re-prefill. The repository still does not implement a replacement scheduler or workload-pressure controller.

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
PYTHONPATH="$PWD/swiftLLM:$PWD/swiftLLM/csrc" "$VENV/bin/python" -m unittest benchmark.test_benchmark benchmark.test_inference_substrate benchmark.test_crossover benchmark.test_runtime_morphing -v

AWQ_ENV=/nfs/home/s314511048/.venvs/morphserve-vllm0112
AWQ_EXT=/nfs/home/s314511048/.cache/morphserve/swiftllm-c-torch29/lib
PYTHONPATH="$AWQ_EXT:$PWD/swiftLLM" "$AWQ_ENV/bin/python" -m unittest benchmark.test_benchmark benchmark.test_inference_substrate benchmark.test_crossover benchmark.test_runtime_morphing -v
```

Regenerate v8 analysis, both CPU suites, segmented-KV CUDA tests, and its completion audit:

```bash
CUDA_VISIBLE_DEVICES=5 benchmark-results/runtime-morphing-v8/regenerate.sh
```

A pass ends with `status: PASS`, all completion checks passing, and decision `GO`. V7 and v6 remain independently reproducible with their existing `regenerate.sh` scripts; do not rerun their broad serving matrices for routine v8 work.

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
| `swiftLLM/benchmark/run_runtime_morphing_validation.py` | Generates static, mid-request, live-extension, capacity, cost, and cycle evidence. |
| `swiftLLM/benchmark/analyze_runtime_morphing.py` | Regenerates v8 state, capacity, cost, memory, provenance, and amortization checks. |
| `swiftLLM/benchmark/audit_runtime_morphing.py` | Performs the independent v8 completion audit. |
| `swiftLLM/benchmark/test_runtime_morphing.py` | CPU contract and opt-in CUDA segmented-KV/allocator/swap tests. |
| `benchmark-results/runtime-morphing-v8/` | Current five raw runs, nine transitions, diagnostics, tables, logs, commands, and audit. |
| `docs/runtime-morphing-v8/` | Current transition contract, ownership, trace format, final report, and audit. |
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

Serving benchmarks launch requests on saved wall-clock offsets without waiting for prior completion; v5 quality runs use an explicit sequential mode. SwiftLLM's control plane (`Engine` and `Scheduler`) admits and batches requests, while the data plane executes model work and maintains the KV cache. V7 opt-in observation also records every forward batch in memory—pre-schedule pressure, admitted prefill/decode shape, effective M, duration, safe/used blocks, and swaps—then writes `batches.jsonl` after execution. Request, periodic, and batch streams are analyzed only after the run.

The v8 control plane accepts only explicit `morph_to_awq_w4_16()` and `restore_to_fp16()` calls. The main loop services them between complete forwards after CUDA synchronization. Selected layers load from pinned-host final layouts; only one representation is active in HBM. The original K/V tensors remain a 1,759-block base, while AWQ adds a 2,411-block physical extension. Block tables use virtual IDs, and store/attention/swap resolve base versus extension. Restoration holds new admission if needed, remaps live extension blocks to free base IDs, releases the extension, restores dense FP16, then publishes base capacity. Strict-FCFS ordering and v7 crossover logic are not changed.

The safe dynamic total is 4,170 blocks. A 4,286-block diagnostic allocation completed but exceeded the 0.99 maximum-shape memory envelope, so it is not advertised as safe. See `docs/runtime-morphing-v8/state-memory-ownership.md` before changing capacity.

## Development Workflow

1. Read `docs/runtime-morphing-v8/final-report.md`, then v7 for crossover evidence and v6 for backend details.
2. Treat v8 GO as validation of a **manual substrate**, not a controller or deployable policy.
3. Preserve `{FP16, AWQ-W4-16}`, the 4,170 safe target, serialization, byte-exact ownership, and physical-before-scheduler resize ordering.
4. Do not retrofit controller triggers into v7/v8. A controller needs a separately pre-registered namespace and workload experiment.
5. Preserve v2/v4/v5/v6/v7 and raw v8 evidence; put new experiments in a new namespace.
6. Do not resolve v7's ambiguous scales 5.25/4.75 with selective repeats or changed thresholds.
7. If runtime source changes, regenerate all five v8 raw files because each is bound to per-file source hashes.
8. Run both CPU suites, opt-in CUDA tests, compilation, the v8 analyzer/audit, and `git diff --check` before handoff.

## Testing

The validated CPU-only checks are:

```bash
VENV=/nfs/home/s314511048/.venv
PYTHONPATH="$PWD/swiftLLM:$PWD/swiftLLM/csrc" "$VENV/bin/python" -m unittest benchmark.test_benchmark benchmark.test_inference_substrate benchmark.test_crossover benchmark.test_runtime_morphing -v
PYTHONPATH="$PWD/swiftLLM:$PWD/swiftLLM/csrc" "$VENV/bin/python" -m py_compile swiftLLM/benchmark/*.py swiftLLM/swiftllm/server/*.py swiftLLM/swiftllm/worker/*.py swiftLLM/swiftllm/worker/layers/*.py swiftLLM/swiftllm/worker/kernels/*.py
git diff --check
```

A successful v8 validation has five raw JSON files, a nine-line `raw/transition-traces.jsonl`, `analysis/state_preservation_summary.json` with `all_pass: true`, and `completion_audit.json` with `status: PASS` and decision `GO`. The CUDA suite must run with `MORPHSERVE_RUN_CUDA_TESTS=1`; otherwise hardware cases intentionally skip. V7, v6, and v5 remain independently reproducible with their existing audits.

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
- **Raw provenance mismatch:** runtime source changed after evidence generation. Rerun all five commands in `benchmark-results/runtime-morphing-v8/execution_commands.json`.

## Documentation Freshness Checklist

- [ ] README quickstart still works.
- [ ] Run commands match the current code.
- [ ] Test commands match the current code.
- [ ] Important files list is still accurate.
- [ ] Architecture map matches the current implementation.
- [ ] Troubleshooting section includes recent known failures.
