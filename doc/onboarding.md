# Onboarding

## What This Project Does

MorphServe is a research workspace built around a vendored SwiftLLM inference engine. The repository measures open-loop request arrivals, queueing, TTFT, TPOT, logical KV occupancy, admission pressure, and static mixed-precision quality–latency behavior. V6 adds offline AutoAWQ plus vLLM AWQ-Marlin selected-layer execution. It solves v5's NF4 workspace/decode failure but still rejects dynamic adaptation under the frozen near-knee P95/SLO gate. The repository does not implement a replacement scheduler or controller.

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
PYTHONPATH="$PWD/swiftLLM:$PWD/swiftLLM/csrc" "$VENV/bin/python" -m unittest benchmark.test_benchmark benchmark.test_inference_substrate -v

AWQ_ENV=/nfs/home/s314511048/.venvs/morphserve-vllm0112
AWQ_EXT=/nfs/home/s314511048/.cache/morphserve/swiftllm-c-torch29/lib
PYTHONPATH="$AWQ_EXT:$PWD/swiftLLM" "$AWQ_ENV/bin/python" -m unittest benchmark.test_benchmark benchmark.test_inference_substrate -v
```

Regenerate v6 analysis and run its 183-check audit:

```bash
benchmark-results/packed-int4-backend-v6/regenerate.sh
```

The complete final sweep command is preserved in [`docs/phase-1/kv-admission-sweep-report.md`](../docs/phase-1/kv-admission-sweep-report.md); it requires a CUDA GPU and is substantially more expensive than the smoke run.

## Important Files

| Path | Role |
|---|---|
| `swiftLLM/swiftllm/server/engine.py` | Engine and benchmark observations. |
| `swiftLLM/swiftllm/server/scheduler.py` | Upstream scheduler; keep unchanged for baseline experiments. |
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
| `benchmark-results/packed-int4-backend-v6/` | Current protocol, kernel evidence, checkpoint manifest, profiles, raw serving, analysis, and audit. |
| `docs/packed-int4-backend-v6/` | Current phase reports, bounded follow-ups, final report, and completion audit. |
| `benchmark-results/static-frontier-v5/` | Preserved NF4 raw quality/serving runs, mechanism probes, and derived outputs. |
| `docs/static-frontier-v5/` | Preserved v5 protocol, final report, and completion audit. |
| `benchmark-results/phase-1/` | Phase 1 raw runs and derived tables/figures, grouped by experiment. |
| `docs/phase-1/` | Current Phase 1 documentation. |
| `docs/static-quantization-quality-latency/` | Static quantization benchmark protocol, report, and completion audit. |
| `references/` | Source reference material. |

## Architecture Map

Serving benchmarks launch requests on saved wall-clock offsets without waiting for prior completion; v5 quality runs use an explicit sequential mode. SwiftLLM's control plane (`Engine` and `Scheduler`) admits and batches requests, while the data plane executes model work and maintains the KV cache. The benchmark records request timestamps and periodic queue/KV/memory snapshots, then derives summaries and plots from those saved files.

The core research constraint is baseline fidelity: benchmark instrumentation may observe state, but `swiftLLM/swiftllm/server/scheduler.py` and its admission semantics remain unchanged. Selected v6 layers load AutoAWQ-scaled norms and packed matrix components, repack once into one Marlin representation, and use the same packed GEMM for decode and prefill. Unselected layers stay on the base FP16 checkpoint. Fused packed up+gate is required to avoid a multi-GiB activation-concatenation peak.

## Development Workflow

1. Read `docs/phase-1/README.md` first.
2. For the current backend decision, read `docs/packed-int4-backend-v6/final-report.md`; read v5 only for the preserved NF4 diagnosis.
3. Do not start controller work: v6 remains NO-GO at the frozen near-knee gate even though W4-16 is promising at the pre-existing high-load neighbor.
4. Change the smallest relevant source file. Do not add a new phase report at repository root.
5. Preserve v2/v4/v5. Keep new raw/derived work in a new namespace and regenerate conclusions from raw timestamps.
6. If revisiting the backend, pre-register a different kernel/backend or workload question before measurement; do not choose a favorable load from the observed v6 curve.
7. Update the current report when the conclusion or reproduction recipe changes.
8. Run both environment test suites, compilation checks, the applicable artifact audit, and `git diff --check` before handing off.

## Testing

The validated CPU-only checks are:

```bash
VENV=/nfs/home/s314511048/.venv
PYTHONPATH="$PWD/swiftLLM:$PWD/swiftLLM/csrc" "$VENV/bin/python" -m unittest benchmark.test_benchmark benchmark.test_inference_substrate -v
PYTHONPATH="$PWD/swiftLLM" "$VENV/bin/python" -m py_compile swiftLLM/benchmark/*.py swiftLLM/swiftllm/server/engine.py swiftLLM/swiftllm/server/structs.py
git diff --check
```

A successful benchmark run creates `metadata.json`, `requests.jsonl`, `telemetry.jsonl`, and `summary.json` in its run directory. Analysis additionally creates CSV/JSON/PNG artifacts from raw timestamps. For v6, use `benchmark-results/packed-int4-backend-v6/regenerate.sh`; success reports `PASS`, 183 checks, and decision `NO-GO`. V5 remains independently reproducible with its 157-check audit.

## Troubleshooting

- **Missing `vllm_flash_attn`:** install the validated wheel with `pip install --no-deps 'vllm-flash-attn==2.6.2'` in the legacy environment. The torch-2.9 AWQ environment uses vLLM's bundled FA2 fallback and has a fresh FP16 parity artifact.
- **`vllm._C` undefined c10 symbol:** vLLM 0.11.2 is ABI-incompatible with the legacy torch-2.5 environment. Use `/nfs/home/s314511048/.venvs/morphserve-vllm0112`; do not upgrade the legacy environment in place.
- **AWQ checkpoint missing:** restore the external path from `phase-b/awq-checkpoint-manifest.json`. Do not casually rerun the 27-minute quantization or overwrite the hashed artifact.
- **AWQ partial state loses KV capacity:** verify fused packed up+gate is active. Separate up/gate recreates a 6.39-GiB activation peak even without weight dequantization.
- **C++ extension build failure with editable install:** use `cd swiftLLM/csrc && $VENV/bin/python setup.py build_ext --inplace`, then return to the repository root. The upstream isolated `pip install -e csrc` path is known to fail in the validated environment.
- **No model or CUDA device:** use the local model snapshot and `CUDA_VISIBLE_DEVICES=3`, or stop; CPU execution is not a substitute for the Phase 1 result.
- **Need to inspect a failure:** check the run-specific `.log` file under the experiment's `logs/` directory and the raw files under its `runs/<run-id>/` directory. Do not treat incomplete runs as experiment evidence.
- **Unexpected v5 SLO ordering:** the frozen trace has synchronized arrival groups; inspect queueing and logical KV together with SLO. The 2-second threshold alone is not a saturation locator.
- **W4-8/W4-16 capacity collapse:** this is the measured bitsandbytes multi-row prefill workspace effect, not larger resident weights. See `docs/static-frontier-v5/final-report.md` before changing memory settings.

## Documentation Freshness Checklist

- [ ] README quickstart still works.
- [ ] Run commands match the current code.
- [ ] Test commands match the current code.
- [ ] Important files list is still accurate.
- [ ] Architecture map matches the current implementation.
- [ ] Troubleshooting section includes recent known failures.
