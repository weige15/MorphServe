# MorphServe

MorphServe is a research repository for measuring SwiftLLM serving behavior under load. Phase 1 is complete: the final experiment reproduced a mixed GPU-KV-admission and compute transition without changing the scheduler.

## Start here

- [Phase 1 context card](docs/phase1/README.md) — the only document normally needed for current work.
- [Final Phase 1 report](docs/phase1/final-report.md) — results and reproduction commands.
- [Final audit](docs/phase1/final-audit.md) — detailed verification evidence.
- [Onboarding guide](doc/onboarding.md) — setup, workflow, tests, and troubleshooting.
- [Static quantization quality–latency benchmark](docs/static-quantization-quality-latency/static-quantization-benchmark-report.md) — protocol, results, and regeneration commands.

Historical reports are preserved in [`docs/phase1/archive/`](docs/phase1/archive/) but are not required for normal context.

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
  --output-dir ../benchmark-results --run-id phase1-low-load-fixed
```

Run the CPU-only checks from the repository root:

```bash
VENV=/nfs/home/s314511048/.venv
PYTHONPATH="$PWD/swiftLLM" "$VENV/bin/python" -m unittest benchmark.test_benchmark -v
```

## Project structure

- `swiftLLM/swiftllm/` — vendored SwiftLLM implementation.
- `swiftLLM/benchmark/` — open-loop runner, analysis, metrics, and tests.
- `benchmark-results/` — raw JSONL runs and derived Phase 1 artifacts.
- `docs/phase1/` — curated current documentation; `archive/` contains superseded reports.
- `docs/static-quantization-quality-latency/` — static quantization benchmark protocol, report, and audit.
- `benchmark-results/static-quantization-quality-latency/` — raw and derived static quantization benchmark artifacts.
- `references/` — experimental reference material and source PDF.

For the full workflow and known environment issues, see [`doc/onboarding.md`](doc/onboarding.md).
