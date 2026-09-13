# Runtime morphing v8 completion audit

## Objective restated as checkable deliverables

One SwiftLLM process must manually transition the validated selected layers and real KV capacity from FP16 to AWQ-Marlin W4-16 and back, at a legal forward boundary, while live requests keep their request/token/position/block/KV state and are never re-prefilled. The AWQ state must physically expose meaningful additional KV storage; restoration must compact or drain live extension references before releasing that storage. Both weight layouts must match their validated static controls. Transition cost, memory ownership/peaks, capacity, integrity, and repeated stability must be measured. No pressure controller, threshold tuning, backend change, scheduler-policy change, broad-frontier rerun, or prior-namespace overwrite is allowed. A GO/NO-GO must follow the six explicit runtime gates.

## Prompt-to-artifact checklist

### Prohibitions and preservation

| Requirement | Evidence | Result |
|---|---|---|
| No workload-pressure controller | `Engine` exposes explicit methods only; audit searches runtime source for v7 pressure/threshold terms | PASS |
| No threshold/backend/scheduler-policy change | State pair is fixed; Marlin path unchanged; scheduler diff is only a temporary explicit restore admission/swap-in hold | PASS |
| Preserve v2/v4/v5/v6/v7 namespaces | `git diff --name-only HEAD` has no path in the five historical result/doc namespaces; audit check | PASS |
| Do not use two engines/request routing as morphing | `state-morph.json` and `state-roundtrip.json` each contain one initialized `Engine`, one model, and in-process transition traces | PASS |

### Phase A — frozen transition contract

| Requirement | Evidence | Result |
|---|---|---|
| Document static FP16/AWQ layouts for layers 0–15 | `state-memory-ownership.md` lists both norms, q/k/v/o, fused up+gate, down, flags, all shapes/dtypes/bytes | PASS |
| Include Marlin workspace/state | Same table lists `[82]` workspace and empty g-index tensors per matrix | PASS |
| Preserve AutoAWQ-rescaled norms and packed matrices | Host staging code in `weight.py`; byte-exact static/runtime hashes in `capacity-cycles.json` | PASS |
| Restore original FP16 norms/dense matrices | 6,979,584,000 compared bytes and matching static SHA-256 | PASS |
| Record 1,768/4,286 static references | Static controls reproduce both exactly; architecture/ownership docs | PASS |
| Define legal boundary | `transition-architecture.md`; `_service_pending_transition()` before scheduling; explicit CUDA synchronize | PASS |

### Phase B — runtime layer morphing

| Requirement | Evidence | Result |
|---|---|---|
| Manual two-state API | `Engine.morph_to_awq_w4_16()` / `restore_to_fp16()` | PASS |
| No checkpoint reload on each transition | `runtime_preparation_trace`: one 10.800-s preparation; hot traces contain H2D only | PASS |
| Alternate representations prepared off hot path | 8,792,834,048 bytes staged in pinned host before KV allocation | PASS |
| Inactive dense/AWQ GPU copy released | Per-layer allocated HBM falls/rises with replacement; only host prepared lists remain; physical extension consumes reclaimed HBM | PASS |
| Serial transition/no tensor race | One async transition lock; main-loop ownership; direct over-capacity restore rejection; tests | PASS |
| Instrument all requested fields | Nine raw traces include timestamps, 16 layer timings, H2D/D2H/D2D, sync, start/intermediate/coexist/end HBM, active requests, used blocks, state | PASS |
| Runtime AWQ byte/shape/backend equivalent to static | 1,813,250,048 immutable bytes and SHA-256 equal to fresh static AWQ; all 16 layouts recorded | PASS |

### Phase C — real dynamic KV

| Requirement | Evidence | Result |
|---|---|---|
| Capacity growth is physical, not scheduler-only | 2,411-block K/V tensors allocated; manager and scheduler publish 4,170 only after allocation | PASS |
| Retain base and active blocks on expansion | Base tensors/IDs unchanged; active logical hashes identical | PASS |
| Virtual IDs span base and extension | `BlockManager.extend`; full model uses `[0,1759]` | PASS |
| KV store/paged attention resolve segments | Triton implementation plus prefill/decode/cross-segment CUDA parity tests | PASS |
| Swap resolves virtual IDs | `_swap` partitions IDs; GPU→CPU→extension GPU test | PASS |
| Safe restoration/drain | Engine holds new admissions/swap-ins while over base; direct unsafe restore rejected; allocator negative test | PASS |
| Remap before extension release | Live `[0,1759]→[0,1]` copy/hash evidence and two additional synthetic repeats | PASS |
| Scheduler updates after physical shrink | Engine context after each restore reports physical=scheduler=1,759 and extension=0 | PASS |
| Do not discard active requests | Two real engine requests survive round-trip; the live extension request remains allocated through remap and supplies the next token | PASS |

### Phase D — deterministic state preservation

| Requirement | Evidence | Result |
|---|---|---|
| All-FP16 token IDs | `raw/state-fp16.json`, two prompts ×24 steps | PASS |
| All-static-AWQ token IDs | `raw/state-static-awq.json`, same prompts ×24 steps | PASS |
| FP16 prefix→AWQ | `raw/state-morph.json`: FP16 ×5 then AWQ ×19 per request | PASS |
| FP16→AWQ→FP16 | `raw/state-roundtrip.json`: 5/7/12 steps per request | PASS |
| Save boundaries | `transition_boundaries` has exact request/output lengths and monotonic timestamps | PASS |
| No retokenize/re-prefill | One prefill batch per benchmark request; all later step rows are decode | PASS |
| Valid output/positions/no skipped steps | 24 contiguous indices, positions increment exactly one, all IDs in vocabulary | PASS |
| Preserve request/block identity | IDs 0/1 stable; base block IDs unchanged on expansion; live extension remap preserves sequence-relative slots | PASS |
| Internal KV integrity | Sampled 2-MiB block SHA-256 before/after expansion and shrink | PASS |
| Concurrent requests | Both transition directions occur with two active engine requests | PASS |
| Real extension state consumed after restore | Full model crosses segment, remaps live KV, then generates a valid FP16 token at position 17 | PASS |

### Phase E — capacity and cost

| Requirement | Evidence | Result |
|---|---|---|
| Runtime FP16 near validated regime/no hidden AWQ HBM | 1,759 blocks, nine-block/0.51% documented preparation-runtime gap | PASS |
| Dense GPU memory released/AWQ active | Coexistence and after-weight HBM, active types/flags, byte hash | PASS |
| Additional blocks physically allocatable | 4,170 total, +2,411/+137.1%; real store/read/free tests | PASS |
| Safe maximum measured | 4,286 diagnostic rejected; 4,170 passes exact 32×49,152 maximum-shape envelope with <one-block budget remainder | PASS |
| Restoration returns FP16 memory/capacity | All cycles end at 1,759 physical/scheduler blocks and static FP16 byte hash | PASS |
| Multiple latency repetitions/decomposition | Three controlled repetitions/direction in `transition_costs.csv` and summary | PASS |
| Leak test | Three cycles: allocated final drift 1,024 B, reserved final drift 2 MiB; 222-MiB non-monotonic middle-cycle variability disclosed | PASS (bounded to 3 cycles) |
| v7 break-even | 0.661-s one-way / 2.171-s round trip versus 2.436–13.018-s v7 P95 advantage; request-equivalent caveat stated | PASS |

### Phase F — decision gates

| Gate | Evidence | Result |
|---|---|---|
| 1. One process transitions both ways | Round-trip raw trace | PASS |
| 2. Active KV/decode state, no re-prefill | Two-request batch/step/KV evidence | PASS |
| 3. Meaningful physical KV growth | 1,759→4,170 (+137.1%) | PASS |
| 4. Safe return to FP16 | live compaction then 1,759 physical blocks/static byte hash | PASS |
| 5. Measured cost does not obviously erase v7 benefit | medians and amortization table | PASS |
| 6. Repeated transitions do not materially leak/corrupt | three-cycle memory and allocator/hash checks | PASS within measured horizon |

### Required outputs

| Output | Artifact | Result |
|---|---|---|
| Architecture note | `transition-architecture.md` | PASS |
| State/memory ownership table | `state-memory-ownership.md` | PASS |
| Timeline/log format | `transition-trace-format.md` | PASS |
| Mid-request tests | `test_runtime_morphing.py`, `state-morph.json`, `state-roundtrip.json` | PASS |
| KV resize/integrity tests | CUDA suite and `capacity-cycles.json` | PASS |
| Repeated leak test | `memory_cycles.csv` | PASS |
| Cost table | `transition_costs.csv`, `transition_cost_summary.json` | PASS |
| Actual capacities | `capacity.csv` | PASS |
| Reproduction commands | `execution_commands.json`, `regenerate.sh`, final report | PASS |
| Raw traces | `raw/transition-traces.jsonl` and five raw JSON files | PASS |
| Completion audit | this document and `completion_audit.json` | PASS |
| Final required report ending | `final-report.md` ends with all five required headings and GO | PASS |

## Verification commands and coverage

- Legacy CPU suite: 27 tests, with five runtime CUDA cases skipped by contract.
- AWQ CPU suite: same 27 tests.
- Opt-in runtime suite: 9 tests, including five real segmented-KV/allocator/swap checks.
- Full raw analyzer: all prompt/state/capacity/provenance/hash/leak/cost checks pass.
- Completion audit: verifies required artifacts, source provenance, historical namespace preservation, external AWQ checkpoint hashes, runtime layouts, physical safety, raw transition coverage, state preservation, costs, and decision consistency.
- `py_compile` covers all benchmark/server/worker Python files.
- `git diff --check` passes.

The green analyzer is not accepted as a proxy by itself: the audit directly reopens the five raw run files, the rejected-capacity diagnostic, source files, static checkpoint, transition JSONL, report, and command manifest.

## Residual qualification

GO is limited to the manual substrate and exact validated environment. The three-cycle test is not a long soak, the controller remains absent, and 4,170—not static 4,286—is the safe one-process capacity. These limitations are carried into the final report rather than treated as completed controller evidence.
