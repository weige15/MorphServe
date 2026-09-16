# Onboarding

## What This Project Does

This workspace investigates whether MorphServe's paper mechanisms and results can be reproduced from the supplied PDF, public sources, local models/data, and RTX 3090 hardware. The main output will be an audited `REPORT.md`; the investigation is not complete. Current evidence supports several modified-condition mechanisms, not the paper's headline aggregate.

## Quickstart

```bash
cd /nfs/home/s314511048/MorphServe
cat reproduction/research-state.yaml

PYTHONPATH="$PWD/reproduction/runtime:$PWD/reproduction/runtime/candidate-python" \
  python3 -m unittest reproduction.tests.test_profiling \
  reproduction.tests.test_candidate_runtime reproduction.tests.test_controller \
  reproduction.tests.test_controller_integration -v
```

Expected CPU result: sixteen tests pass. GPU runners create/clear `reproduction/.venv`, install pinned cached dependencies, save commands/logs/metrics, and return a nonzero code when a predeclared gate fails.

Representative GPU gates:

```bash
CUDA_VISIBLE_DEVICES=0 reproduction/scripts/run_candidate_kv_mapping_test.sh
CUDA_VISIBLE_DEVICES=1 reproduction/scripts/run_autoawq_layer_switch.sh
CUDA_VISIBLE_DEVICES=1 reproduction/scripts/run_active_kv_switch.sh
CUDA_VISIBLE_DEVICES=1 reproduction/scripts/run_lis_real_pilot.sh
```

## Important Files

| Path | Role |
|---|---|
| `research-state.yaml` | Current checkpoint, resources, hypothesis, next actions |
| `research-log.md`, `findings.md` | Attempt timeline and synthesized memory |
| `docs/paper-evidence-brief.md` | Visual audit of all 19 pages, Tables 1–8, Figures 1–7, equations |
| `docs/source-map.md` | Paper requirement → source/artifact mapping |
| `docs/claim-register.md` | Claim status and required evidence |
| `configs/paper-reference-values.json` | Paper-reported references only |
| `vendor/author-morphserve/` | Immutable candidate artifact at `85c4fbf...` |
| `runtime/candidate-csrc/` | Native-metadata/event-safe repaired C++ reconstruction |
| `runtime/candidate-python/` | Normalized candidate Python fork |
| `runtime/morphserve/` | Tested LIS core and independent AutoAWQ adapter |
| `experiments/*/protocol.md` | Pre-registered experiment contract |
| `experiments/*/results*/` | Raw logs, metrics, commands, verification |
| `profiles/` | Frozen offline layer order artifacts |
| `doc/debug-report*.md` | Root-cause records and applied repairs |

Local heavyweight assets remain outside Git under `/nfs/home/s314511048/.cache/`; their hashes are in `results/raw/environment.json`.

## Architecture Map

Paper flow: Serving Monitor → Morphing Controller → per-worker Executor → layer swap + KV resize. The reconstruction currently validates lower layers of that stack:

1. normalized SwiftLLM FP16 model;
2. real AutoAWQ per-layer packed variants in pinned CPU memory;
3. repaired C++ same-address copy and reclaimed-tail storage;
4. candidate Triton KV store/PagedAttention virtual block mapping;
5. explicit per-region CUDA lifetime events;
6. independent Algorithm 1 conditioned-MDS profiler.

Controller modes, workload replay, multi-request pressure, and paper baselines remain incomplete. Exact author code provenance is unresolved.

## Development Workflow

1. Read `research-state.yaml`, `findings.md`, and the relevant protocol/result.
2. Never edit `vendor/`; verify `MANIFEST.sha256` before/after GPU work.
3. Lock a protocol in Git before an experiment; commit results separately.
4. Use a separate `runtime/` repair and document provenance.
5. Fix correctness before trusting timing. Save negative results.
6. Keep paper reference values separate from observed metrics.
7. Update `research-log.md`, `findings.md`, claim/source maps, and this onboarding file after material changes.

## Testing

| Surface | Command | Passing evidence |
|---|---|---|
| LIS logic | CPU unittest in Quickstart | 4 profiling tests pass |
| Candidate config | CPU unittest in Quickstart | config test passes |
| C++ physical reclaim | `run_candidate_memory_manager_test.sh` | exit 0, bounded addresses, exact restore |
| Two-tail mapping | `run_candidate_kv_mapping_test.sh` | dense attention oracle max error 0 |
| Event lifetime | `run_candidate_lifetime_test.sh` | recorded race 0 corruption; unrecorded sensitivity corrupts |
| FP16 baseline | `run_candidate_fp16_baseline.sh` | top-k match, relative L2 <0.005, exact weights |
| Static W4 | `run_static_autoawq_baseline.sh` | expected exit 1 only because bit-exact repeat gate fails; packed mechanism evidence remains valid |
| W4 switch | `run_autoawq_layer_switch.sh` | real W4, same base, exact FP16 restore |
| Active KV | `run_active_kv_switch.sh` | same-history gates and migration pass |
| LIS pilot | `run_lis_real_pilot.sh` | 6 conditioned calls, saved `[29,30,31]` profile |

## Troubleshooting

| Symptom | Likely cause / action |
|---|---|
| `package directory 'morphserve' does not exist` | Expected unmodified candidate packaging defect; use normalized runtime |
| Shutdown SIGSEGV after tests | Do not use vendor extension; use `runtime/candidate-csrc` native metadata repair |
| `MODEL_PATH` selects a remote ID | Use `MORPHSERVE_MODEL_PATH`; runner validates `config.json` |
| `evaluate` missing during core import | Normalized runtime lazy-loads it only for ROUGE |
| W4 repeats differ slightly | Expected AutoAWQ split-K atomic variance; use frozen envelope, not exact equality |
| Multi-second first decode | Triton JIT; warm/precompile before timing |
| CUDA OOM/unexpected latency | Inspect saved `nvidia-before.csv`; another process may occupy the physical GPU |
| Full 32-layer pinned variants fail | Audit the 16.45-GB memlock limit before retrying; do not bypass limits |

## Documentation Freshness Checklist

- [ ] README quickstart still works.
- [ ] Run commands match the current code.
- [ ] Test commands match the current code.
- [ ] Important files list is still accurate.
- [ ] Architecture map matches the current implementation.
- [ ] Troubleshooting section includes recent known failures.
