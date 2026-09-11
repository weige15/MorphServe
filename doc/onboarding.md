# Onboarding

## What This Project Does

MorphServe is a research workspace built around a vendored SwiftLLM inference engine. The repository measures open-loop request arrivals, queueing, TTFT, TPOT, logical KV occupancy, and admission pressure. Phase 1 produces reproducible raw telemetry and derived reports; it does not implement a replacement scheduler.

## Quickstart

Validated assumptions:

- Python: `/nfs/home/s314511048/.venv/bin/python` (3.12.3).
- Model: local Meta Llama 3.1 8B snapshot documented in `docs/phase-1/baseline-configuration.md`.
- GPU: CUDA device 3, an RTX 3090.
- Existing SwiftLLM C++ extension and required runtime dependencies are available in the validated environment.

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
PYTHONPATH="$PWD/swiftLLM" "$VENV/bin/python" -m unittest benchmark.test_benchmark -v
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
| `swiftLLM/benchmark/analyze_static_quantization_quality_latency.py` | Derives quality/latency tables and plots from raw condition runs. |
| `benchmark-results/phase-1/` | Phase 1 raw runs and derived tables/figures, grouped by experiment. |
| `docs/phase-1/` | Current Phase 1 documentation. |
| `docs/static-quantization-quality-latency/` | Static quantization benchmark protocol, report, and completion audit. |
| `references/` | Source reference material. |

## Architecture Map

The benchmark launches requests on wall-clock offsets without waiting for prior completion. SwiftLLM's control plane (`Engine` and `Scheduler`) admits and batches requests, while the data plane executes model work and maintains the KV cache. The benchmark records request timestamps and periodic queue/KV/memory snapshots, then derives summaries and plots from those saved files.

The core research constraint is baseline fidelity: benchmark instrumentation may observe state, but `swiftLLM/swiftllm/server/scheduler.py` and its admission semantics remain unchanged.

## Development Workflow

1. Read `docs/phase-1/README.md` first.
2. For Phase 1 results, read `docs/phase-1/kv-admission-sweep-report.md`; for static quantization quality–latency results, read `docs/static-quantization-quality-latency/static-quantization-benchmark-report.md`.
3. Change the smallest relevant source file. Do not add a new phase report at repository root.
4. Keep Phase 1 raw runs under `benchmark-results/phase-1/<experiment>/runs/` and regenerate derived artifacts from raw data.
5. Update the phase context card only when the current conclusion, reproduction recipe, or document map changes.
6. Run tests, compilation checks, and `git diff --check` before handing off.

## Testing

The validated CPU-only checks are:

```bash
VENV=/nfs/home/s314511048/.venv
PYTHONPATH="$PWD/swiftLLM" "$VENV/bin/python" -m unittest benchmark.test_benchmark -v
PYTHONPATH="$PWD/swiftLLM" "$VENV/bin/python" -m py_compile swiftLLM/benchmark/*.py swiftLLM/swiftllm/server/engine.py swiftLLM/swiftllm/server/structs.py
 git diff --check
```

A successful benchmark run creates `metadata.json`, `requests.jsonl`, `telemetry.jsonl`, and `summary.json` in its run directory. Analysis additionally creates CSV/JSON/PNG artifacts and verifies percentiles from raw timestamps.

## Troubleshooting

- **Missing `vllm_flash_attn`:** install the validated wheel with `pip install --no-deps 'vllm-flash-attn==2.6.2'` in the project environment, as recorded in `docs/phase-1/baseline-configuration.md`.
- **C++ extension build failure with editable install:** use `cd swiftLLM/csrc && $VENV/bin/python setup.py build_ext --inplace`, then return to the repository root. The upstream isolated `pip install -e csrc` path is known to fail in the validated environment.
- **No model or CUDA device:** use the local model snapshot and `CUDA_VISIBLE_DEVICES=3`, or stop; CPU execution is not a substitute for the Phase 1 result.
- **Need to inspect a failure:** check the run-specific `.log` file under the experiment's `logs/` directory and the raw files under its `runs/<run-id>/` directory. Do not treat incomplete runs as experiment evidence.

## Documentation Freshness Checklist

- [ ] README quickstart still works.
- [ ] Run commands match the current code.
- [ ] Test commands match the current code.
- [ ] Important files list is still accurate.
- [ ] Architecture map matches the current implementation.
- [ ] Troubleshooting section includes recent known failures.
