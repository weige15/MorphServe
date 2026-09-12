# MorphServe

MorphServe is a research repository for measuring SwiftLLM serving behavior under load. Phase 1 reproduced a mixed GPU-KV-admission and compute transition without changing the scheduler. The latest static-frontier experiment found **NO-GO** for dynamic adaptation on the current NF4 backend: none of the tested W4 states provides repeatable latency/SLO relief.

## Start here

- [Phase 1 context card](docs/phase-1/README.md) — the only document normally needed for current work.
- [Final Phase 1 report](docs/phase-1/kv-admission-sweep-report.md) — results and reproduction commands.
- [Final audit](docs/phase-1/kv-admission-sweep-audit.md) — detailed verification evidence.
- [Onboarding guide](doc/onboarding.md) — setup, workflow, tests, and troubleshooting.
- [Static frontier v5 final report](docs/static-frontier-v5/final-report.md) — 106-request paired quality, repeated serving sweep, state eligibility, and NO-GO decision.
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
PYTHONPATH="$PWD/swiftLLM" "$VENV/bin/python" -m unittest benchmark.test_benchmark -v
```

## Project structure

- `swiftLLM/swiftllm/` — vendored SwiftLLM implementation.
- `swiftLLM/benchmark/` — open-loop runner, analysis, metrics, and tests.
- `benchmark-results/phase-1/` — Phase 1 raw runs and derived artifacts, grouped by experiment.
- `docs/phase-1/` — curated current documentation; `archive/` contains superseded reports.
- `docs/static-frontier-v5/` — current static-frontier protocol, report, and completion audit.
- `benchmark-results/static-frontier-v5/` — current frozen inputs, raw runs, mechanism probes, and derived evidence.
- `docs/static-quantization-quality-latency/` and `benchmark-results/static-quantization-quality-latency/` — preserved historical v2 benchmark.
- `references/` — experimental reference material and source PDF.

For the full workflow and known environment issues, see [`doc/onboarding.md`](doc/onboarding.md).
