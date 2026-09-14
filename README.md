# MorphServe

MorphServe is a research repository for measuring and changing SwiftLLM serving behavior under load. V7 validated the static `{FP16, AWQ-W4-16}` crossover, v8 established state-preserving one-process morphing, and v9 confirmed high-pressure entry but failed useful release. Release-side v10 now demonstrates repeated useful AWQ→FP16 restoration on held-out one- and two-cycle workloads, while retaining a formal **NO-GO** because its preregistered 0.98 throughput non-inferiority margin narrowly failed.

## Start here

- [Release-side runtime v10 final report](docs/release-side-runtime-v10/final-report.md) — current 16-run held-out reversibility result and formal decision.
- [Release-side v10 completion audit](docs/release-side-runtime-v10/completion-audit.md) — prompt-to-artifact verification and bounded analysis-correction disclosure.
- [V9 release-delay diagnosis](docs/release-side-runtime-v10/release-delay-diagnosis.md) — development replay and frozen release-intent rationale.
- [Closed-loop runtime v9 final report](docs/closed-loop-runtime-v9/final-report.md) — preserved 18-run NO-GO baseline.
- [Closed-loop runtime v9 completion audit](docs/closed-loop-runtime-v9/completion-audit.md) — exhaustive prompt-to-artifact verification.
- [Runtime morphing v8 final report](docs/runtime-morphing-v8/final-report.md) — one-process state/KV transition GO evidence.
- [Runtime morphing v8 completion audit](docs/runtime-morphing-v8/completion-audit.md) — prompt-to-artifact verification.
- [FP16/AWQ crossover v7 final report](docs/fp16-awq-crossover-v7/final-report.md) — preserved nine-load, 36-run crossover evidence.
- [Crossover v7 completion audit](docs/fp16-awq-crossover-v7/completion-audit.md) — prompt-to-artifact verification.
- [Phase 1 context card](docs/phase-1/README.md) — original KV-admission context.
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
  benchmark.test_benchmark benchmark.test_inference_substrate benchmark.test_crossover \
  benchmark.test_runtime_morphing benchmark.test_closed_loop_controller -v
```

Regenerate the saved v10 diagnosis, held-out tables, decision, tests, and raw-evidence audit:

```bash
benchmark-results/release-side-runtime-v10/regenerate.sh
```

The audit ends `status: PASS`; the frozen scientific decision remains `NO-GO`. V9 remains reproducible with `benchmark-results/closed-loop-runtime-v9/regenerate.sh`. V8 (including CUDA segmented-KV tests), v7, and v6 remain independently reproducible with
`CUDA_VISIBLE_DEVICES=5 benchmark-results/runtime-morphing-v8/regenerate.sh`,
`benchmark-results/fp16-awq-crossover-v7/regenerate.sh` and
`benchmark-results/packed-int4-backend-v6/regenerate.sh`.

## Project structure

- `swiftLLM/swiftllm/` — vendored SwiftLLM plus explicit manual morphing and segmented KV support.
- `swiftLLM/benchmark/` — open-loop runners, runtime validation, analysis, metrics, and tests.
- `benchmark-results/phase-1/` — Phase 1 raw runs and derived artifacts, grouped by experiment.
- `docs/phase-1/` — curated current documentation; `archive/` contains superseded reports.
- `docs/release-side-runtime-v10/` and `benchmark-results/release-side-runtime-v10/` — release diagnosis, preregistered policy/workloads, 16 held-out runs, exact drain timelines, GPU telemetry, report, and audit.
- `docs/closed-loop-runtime-v9/` and `benchmark-results/closed-loop-runtime-v9/` — preserved frozen controller protocol, 18 runs, raw traces, tables, timelines, final report, and audit.
- `docs/runtime-morphing-v8/` and `benchmark-results/runtime-morphing-v8/` — architecture/ownership contract, raw transitions, capacity/cost/state evidence, report, and audit.
- `docs/fp16-awq-crossover-v7/` and `benchmark-results/fp16-awq-crossover-v7/` — preserved crossover protocol, multi-M diagnosis, 36 raw serving runs, analysis, report, and audit.
- `docs/packed-int4-backend-v6/` and `benchmark-results/packed-int4-backend-v6/` — preserved AWQ-Marlin backend validation and prior static-gate NO-GO.
- `docs/static-frontier-v5/` and `benchmark-results/static-frontier-v5/` — preserved NF4 static-frontier evidence.
- `docs/static-quantization-quality-latency/` and `benchmark-results/static-quantization-quality-latency/` — preserved historical v2 benchmark.
- `references/` — experimental reference material and source PDF.

For the full workflow and known environment issues, see [`doc/onboarding.md`](doc/onboarding.md).
