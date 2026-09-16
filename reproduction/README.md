# MorphServe reproduction workspace

## What This Project Does

Auditable reproduction of the supplied 19-page MorphServe paper. It preserves reported values separately from measurements, vendors immutable public snapshots, tests a repaired independent reconstruction, and records negative results/blockers. No headline paper claim is currently reproduced.

## Quickstart

Requirements: Linux, Python 3.12, `uv`, CUDA 12.x, and an NVIDIA GPU for GPU runners. Scripts create `reproduction/.venv`; model/data paths default to the locally audited assets.

```bash
# Setup-free CPU checks
PYTHONPATH="$PWD/reproduction/runtime:$PWD/reproduction/runtime/candidate-python" \
  python3 -m unittest reproduction.tests.test_profiling \
  reproduction.tests.test_candidate_runtime reproduction.tests.test_controller \
  reproduction.tests.test_controller_integration -v

# Current state and evidence
cat reproduction/research-state.yaml
```

## Common Commands

```bash
# Synthetic C++/KV/event GPU gates
CUDA_VISIBLE_DEVICES=0 reproduction/scripts/run_candidate_memory_manager_test.sh
CUDA_VISIBLE_DEVICES=0 reproduction/scripts/run_candidate_kv_mapping_test.sh
CUDA_VISIBLE_DEVICES=0 reproduction/scripts/run_candidate_lifetime_test.sh

# Full-model modified-condition gates
CUDA_VISIBLE_DEVICES=1 reproduction/scripts/run_candidate_fp16_baseline.sh
CUDA_VISIBLE_DEVICES=1 reproduction/scripts/run_static_autoawq_baseline.sh       # expected exit 1: exact-repeat gate
CUDA_VISIBLE_DEVICES=1 reproduction/scripts/run_autoawq_layer_switch.sh
CUDA_VISIBLE_DEVICES=1 reproduction/scripts/run_active_kv_switch.sh
CUDA_VISIBLE_DEVICES=1 reproduction/scripts/run_lis_real_pilot.sh
```

## Project Structure

- `vendor/`: immutable candidate MorphServe and SwiftLLM snapshots.
- `runtime/`: independent repaired C++/Python reconstruction and LIS/AutoAWQ adapters.
- `experiments/`: locked protocols, raw logs, metrics, analyses, and negative results.
- `profiles/`: saved offline layer orders.
- `configs/paper-reference-values.json`: paper values only, never measurements.
- `docs/`: source map, claim register, paper audit, and artifact audit.
- `doc/`: onboarding and debug reports.

## Testing

Passing tests/runners must do real work and save raw evidence. The main verified surfaces are FP16 parity, real W4 switching, physical KV reclaim/mapping, event lifetime, active-KV same-history continuity, and a real three-layer conditioned-MDS profile. Check each runner's `run.exitcode`/`test.exitcode` and `verification.log`; do not rely only on console `OK`.

## Documentation Map

- [Current reproduction report](REPORT.md)
- [Prompt-to-artifact completion audit](docs/completion-audit.md)
- [Onboarding and handoff](doc/onboarding.md)
- [Research state](research-state.yaml)
- [Paper evidence brief](docs/paper-evidence-brief.md)
- [Source-to-implementation map](docs/source-map.md)
- [Claim register](docs/claim-register.md)
- [Public artifact audit](docs/author-artifact-audit.md)
- [Public trace audit](docs/trace-audit.md)
- [Task/quality artifact audit](docs/task-artifact-audit.md)
- [Research findings](findings.md)
- [Research log](research-log.md)

## Troubleshooting

- Official `ds2-lab/MorphServe` is README-only; do not treat it as released code.
- Candidate source is not runnable unmodified; see `experiments/candidate-artifact-smoke/`.
- Candidate C++ originally segfaulted at shutdown; see `doc/debug-report.md`.
- AutoAWQ split-K logits are not bit deterministic; see `doc/debug-report-autoawq-repeat.md`.
- First-call multi-second attention logs are Triton JIT, not steady-state timings.
- Verify a GPU is actually free with `nvidia-smi`; saved before/after snapshots identify contention.
