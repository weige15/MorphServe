# MorphServe

MorphServe is a research repository for measuring SwiftLLM serving behavior under load. Phase 1 reproduced a mixed GPU-KV-admission and compute transition without changing the scheduler. The latest packed-backend experiment replaced the rejected NF4 path with offline AWQ + vLLM Marlin. It achieved monotonic 71.6–282.7% safe-KV relief and non-regressive isolated decode, but remains **NO-GO** for dynamic adaptation because every W4 state had worse P95/SLO at the frozen near-knee workload.

## Start here

- [Phase 1 context card](docs/phase-1/README.md) — the only document normally needed for current work.
- [Final Phase 1 report](docs/phase-1/kv-admission-sweep-report.md) — results and reproduction commands.
- [Final audit](docs/phase-1/kv-admission-sweep-audit.md) — detailed verification evidence.
- [Onboarding guide](doc/onboarding.md) — setup, workflow, tests, and troubleshooting.
- [Packed INT4 backend v6 final report](docs/packed-int4-backend-v6/final-report.md) — AWQ-Marlin kernel, integration, resource, repeated serving, and NO-GO evidence.
- [Packed backend v6 completion audit](docs/packed-int4-backend-v6/completion-audit.md) — prompt-to-artifact verification.
- [Static frontier v5 final report](docs/static-frontier-v5/final-report.md) — preserved NF4 diagnosis and prior NO-GO decision.
- [Static quantization quality–latency benchmark](docs/static-quantization-quality-latency/static-quantization-benchmark-report.md) — historical v2 protocol and results.

Historical reports are preserved in [`docs/phase-1/archive/`](docs/phase-1/archive/) but are not required for normal context.

## Quickstart

The validated environment uses Python 3.12, the project virtualenv at `/nfs/home/s314511048/.venv`, a local Llama 3.1 8B snapshot, and CUDA device 3. A low-load benchmark smoke run is:

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

Run the CPU-only checks from the repository root:

```bash
VENV=/nfs/home/s314511048/.venv
PYTHONPATH="$PWD/swiftLLM:$PWD/swiftLLM/csrc" "$VENV/bin/python" -m unittest \
  benchmark.test_benchmark benchmark.test_inference_substrate -v
```

Regenerate and audit the saved v6 evidence (the audit also verifies the external AWQ checkpoint hashes):

```bash
benchmark-results/packed-int4-backend-v6/regenerate.sh
```

## Project structure

- `swiftLLM/swiftllm/` — vendored SwiftLLM implementation.
- `swiftLLM/benchmark/` — open-loop runner, analysis, metrics, and tests.
- `benchmark-results/phase-1/` — Phase 1 raw runs and derived artifacts, grouped by experiment.
- `docs/phase-1/` — curated current documentation; `archive/` contains superseded reports.
- `docs/packed-int4-backend-v6/` and `benchmark-results/packed-int4-backend-v6/` — current AWQ-Marlin protocol, implementation evidence, profiles, serving results, reports, and audit.
- `docs/static-frontier-v5/` and `benchmark-results/static-frontier-v5/` — preserved NF4 static-frontier evidence.
- `docs/static-quantization-quality-latency/` and `benchmark-results/static-quantization-quality-latency/` — preserved historical v2 benchmark.
- `references/` — experimental reference material and source PDF.

For the full workflow and known environment issues, see [`doc/onboarding.md`](doc/onboarding.md).
