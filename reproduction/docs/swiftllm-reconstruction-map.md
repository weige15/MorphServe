# Code Context

## Repository identity/history

- Public repository: `https://github.com/interestingLSY/swiftLLM` (case-sensitive display name; GitHub clone URL works as shown below).
- Remote inspection on 2026-09-16 found **one branch only**, `master` at `682cf9a28f97f7490409981a2f181528f377eb5d`, and **no tags**. GitHub reports the last push on 2025-06-10.
- Relevant history anchors:
  - `682cf9a28f97f7490409981a2f181528f377eb5d` (2025-06-10), current master, merge of model auto-detection/stream work.
  - `81c26fbfb91a49ae90ee226835430c4c89b4b3af` (2025-06-03), model auto-detection and stream optimization.
  - `af7a5589fdac7b2d8b080ed34f2be706f20724a0` (2024-07-05), installation/examples; last commit in the original 2024 development period.
  - `0db592af6ab17bbbee76ee5602ca302b5a77b723` (2024-06-13), online serving/API complete.
  - `f6b80f888cd4fd4dd80d89b53ef427ab3859a2f9` (2024-06-13), data-plane block swapping.
  - `1940e591850c3a9dc1befa16fcaa5566d2bd60b9` (2024-05-16), GPU block manager.
- There is no upstream tag or manifest tying SwiftLLM to the MorphServe paper. If the paper implementation dates to 2024, `af7a558...` is the defensible contemporaneous pin; if reconstructing against currently published code, use `682cf9a...`. This choice remains a provenance risk and must be checked against the paper artifact/supplement.
- Existing local copies were found under `/nfs/home/s314511048/precision-batching/vendor/swiftLLM-upstream` and `.../vendor/swiftLLM`. They are directories inside the enclosing `precision-batching` Git worktree, not standalone upstream clones. `swiftLLM-upstream/UPSTREAM_COMMIT` records exactly `682cf9a...`; `swiftLLM` is a locally modified research variant (extra precision/KV files and built `.so`). A compiled cache also exists at `/nfs/home/s314511048/.cache/morphserve/swiftllm-c-torch29`. Do not treat the modified vendor directory as pristine.

Clone/pin commands:
```bash
git clone https://github.com/interestingLSY/swiftLLM.git
git -C swiftLLM checkout 682cf9a28f97f7490409981a2f181528f377eb5d
# Alternative paper-era candidate:
git -C swiftLLM checkout af7a5589fdac7b2d8b080ed34f2be706f20724a0
```

## Files Retrieved

1. `/nfs/home/s314511048/precision-batching/vendor/swiftLLM-upstream/README.md` (lines 1-119) - stated control/data-plane architecture, feature exclusions, installation.
2. `.../swiftllm/server/engine.py` (lines 16-180) - initialization and the sole async scheduling/forward loop.
3. `.../swiftllm/server/scheduler.py` (lines 8-144) - request IDs, strict-FCFS queues, admission and whole-sequence swap decisions.
4. `.../swiftllm/server/structs.py` (lines 5-64) - request/step state suitable for monitor annotations.
5. `.../swiftllm/worker/model.py` (lines 18-408) - worker, layer loop, memory profiling, monolithic KV tensors, allocation, swaps.
6. `.../swiftllm/worker/block_manager.py` (lines 5-109) - GPU-resident block table/free bitmap and append-only allocation.
7. `.../swiftllm/worker/infer_state.py` (lines 1-31) - per-forward attention metadata.
8. `.../swiftllm/worker/weight.py` (lines 10-269) - per-layer weight objects, eager FP16 loading and fused `up_gate_proj`.
9. `.../swiftllm/worker/layers/transformer_layer.py` (lines 16-121) - layer execution, FP16 linear calls, KV store, flash-prefill and paged decode.
10. `.../swiftllm/worker/kernels/kvcache_mgmt.py` (lines 8-139) - Triton block-table mapping for KV writes.
11. `.../swiftllm/worker/kernels/paged_attn.py` (lines 8-238) - Triton decode reads block IDs; assumes uniform contiguous cache tensor layout.
12. `.../swiftllm/worker/kernels/block_mgmt.py` (lines 6-121) - Triton updates/gathers rows of the block table.
13. `.../swiftllm/worker/kernels/linear.py` (lines 1-4) - `torch.nn.functional.linear`, the AWQ replacement seam.
14. `.../csrc/src/block_swapping.cpp` (lines 1-100) - current-stream `cudaMemcpyAsync`, grouped only for adjacent source+destination blocks.
15. `.../swiftllm/engine_config.py` (lines 5-91) - static block/memory/batch limits.
16. `.../requirements.txt` (lines 1-7), `setup.py` (lines 1-12), `csrc/setup.py` (lines 1-34) - dependency/build surface.

## Key Code

- `Engine._main_event_loop()` is the integration spine: `Scheduler.get_next_batch()` -> synchronous swap-out -> synchronous swap-in -> `LlamaModel.forward()` -> completion/free. Instrument this for a **Serving Monitor** (arrival rate, queue sizes, TTFT/TPOT, GPU block pressure) and insert controller decisions immediately before scheduling and after batch completion.
- `Scheduler.{waiting_q,running_q,swapped_q,num_decoding_gpu_blocks}`, `_get_block_needed()`, `get_next_batch()`, and `on_batch_finish()` are the **Morphing Controller** seams. Today policy is strict FCFS and only counts uniform KV blocks; it has no weight residency, bandwidth, layer state, deadlines, or hysteresis.
- `LlamaModel._forward()` loops over `self.transformer_layers`; this is the **Morphing Executor** seam. Add a per-layer residency/state machine, transfer stream/events, safe handoff, and controller command queue here. `load_weights()` currently eagerly materializes all layers on CUDA.
- `LlamaTransformerLayerWeight`/`RegisteredWeightItem` and `load_weights()` plus `kernels/linear.linear()` are the **AWQ INT4** seams. AWQ requires packed-qweight/qzeros/scales metadata and a GEMM/dequant kernel; the current loader asserts dense exact shapes/CUDA and converts every item to FP16. Preserve FP16 norms/embeddings/head and decide how the fused gate/up projection is packed.
- KV is allocated as `[physical_block, layer, kv_head, token, head_dim]` in `LlamaModel.init_kvcache_and_swap()`. `BlockManager.block_table[seq_id, logical_block] -> physical_block` already supports non-contiguous physical blocks. Both `store_kvcache` and `paged_attention` dereference this mapping. Thus **non-contiguous placement already exists**, but dynamic resizing means growing/shrinking pools or changing per-layer capacity, which the single monolithic tensor and uniform block IDs do not support.
- For dynamic/per-layer KV sizing, change `LlamaModel.init_kvcache_and_swap`, `BlockManager` allocation APIs, `LlamaInferState`, `store_kvcache`, and `paged_attention`. Likely replace monolithic K/V with layer-specific slabs/pointers and layer block tables, or preserve virtual IDs behind an indirection table. Add explicit shrink/migrate/compact operations; current allocation only grows a sequence to `ceil(len/block_size)` and frees whole sequences.
- `LlamaModel._swap()` gathers all blocks of whole sequences and calls `swiftllm_c.swap_blocks`. `block_swapping.cpp::swap_blocks()` issues async copies on the current stream but Python awaits the whole operation before forward and CPU tensors are ordinary (not explicitly pinned). **Asynchronous per-layer swapping** needs pinned host buffers, dedicated transfer streams, per-layer copy APIs, CUDA events consumed by each layer, lifetime protection, and overlap-aware scheduling.

## Architecture

Control plane is a single Python `Engine` plus Ray tokenization actor. Data plane is one in-process `LlamaModel` on one NVIDIA GPU. A forward prepares GPU metadata, allocates pages, then executes pre-layer -> all transformer layers -> post-layer. Prefill uses `vllm_flash_attn`; decode and KV/block management use custom Triton. A tiny CUDA extension only handles CPU/GPU block copies. There is no tensor/pipeline parallelism, distributed worker RPC, quantization, multi-model support, or non-greedy sampling.

Dependencies are weakly pinned: Python >=3.9, `fastapi>=0.111`, `flash_attn>=2.5.8`, `ray[default]>=2.21`, `safetensors>=0.4.3`, `transformers>=4.40`, `uvicorn>=0.29`, and `vllm_flash_attn>=2.5.8`; PyTorch/Triton/CUDA have no lock. The CUDA extension uses PyTorch `CUDAExtension`. Reproducibility therefore requires a tested torch/CUDA/Triton/flash-attention matrix and a lockfile/container.

## Findings and risks

- **blocker** — `swiftllm/server/engine.py:57` calls `Scheduler(self.model, ...)`, while `scheduler.py:39-42` expects a `LlamaModelConfig` and later reads no model-specific field today. It is an API/type defect and could fail once model config is used; correct to `self.model_config` in a reconstruction.
- **high** — AWQ is explicitly unsupported upstream. Implementing correct INT4 loading plus performant fused GEMM across supported GPUs is a major kernel project; estimate 4-8 engineer-weeks plus numerical/performance validation.
- **high** — True layer morphing conflicts with eager all-GPU dense weights and a strictly sequential layer loop. Async eviction can race kernels unless event/lifetime ownership is rigorous; estimate 4-7 weeks.
- **high** — Per-layer/dynamic KV capacities invalidate the uniform physical-block stride embedded in Triton pointer arithmetic and C++ whole-block byte sizing. Estimate 3-6 weeks; test fragmented tables, resize under decode, swap during resize, OOM rollback, and long contexts.
- **high** — CPU swap buffers are not explicitly pinned (`torch.zeros(..., device="cpu")`), so promised async H2D/D2H overlap is doubtful and must be measured.
- **medium** — Monitor timings around executor-thread CUDA calls require events/synchronization discipline; naive wall-clock metrics measure enqueue time, not kernel completion.
- **medium** — one worker/one GPU only; any MorphServe design assuming multi-GPU placement needs a new worker/RPC architecture rather than extension alone.
- **medium** — no upstream tests were found. Build unit tests for allocator invariants and mocked controller first, then GPU kernel parity against dense PyTorch, concurrency stress, memory-leak tests, and end-to-end serving SLO tests.
- **medium** — loose dependencies and private API sensitivity (`vllm_flash_attn`) make current-master behavior environment-dependent.

## Start Here

Open `swiftllm/worker/model.py` first: it owns eager layer storage, the layer execution boundary, KV slabs/block managers, and swap entry points. In parallel, use `swiftllm/server/engine.py::_main_event_loop` as the monitor/controller command boundary.

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "Concrete source paths/symbols, upstream commit hashes, severity-ranked findings, extension seams, commands, and residual implementation risks are documented above."
    }
  ],
  "changedFiles": [],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "find /nfs/home/s314511048 -maxdepth 5 -iname '*swiftllm*'",
      "result": "passed",
      "summary": "Found vendor copies, modified research copy, scripts/results, and compiled cache."
    },
    {
      "command": "git ls-remote --heads --tags https://github.com/interestingLSY/swiftLLM.git",
      "result": "passed",
      "summary": "Found only master at 682cf9a28f97f7490409981a2f181528f377eb5d and no tags."
    },
    {
      "command": "GitHub commits API inspection",
      "result": "passed",
      "summary": "Enumerated history from initial 2024 commit through 2025 master."
    },
    {
      "command": "targeted source inspection (find/grep/read/nl)",
      "result": "passed",
      "summary": "Mapped engine, scheduler, worker, weights, allocator, Triton kernels, CUDA swap extension, and dependencies."
    }
  ],
  "validationOutput": [
    "UPSTREAM_COMMIT in the pristine vendor snapshot equals remote master: 682cf9a28f97f7490409981a2f181528f377eb5d.",
    "No upstream test directory or pinned dependency lock was present."
  ],
  "residualRisks": [
    "The exact commit cited/used by the MorphServe paper cannot be attested from SwiftLLM metadata alone; af7a558 is only a date-based paper-era candidate.",
    "No GPU build or runtime tests were run because this was a no-modification scouting task.",
    "Local vendor directories belong to another enclosing worktree and one is research-modified; neither should silently become the clean reconstruction base."
  ],
  "noStagedFiles": true,
  "diffSummary": "No repository files modified; only the required external scouting artifact was written.",
  "reviewFindings": [
    "blocker: swiftllm/server/engine.py:57 - Scheduler receives self.model although its constructor declares LlamaModelConfig.",
    "high: swiftllm/worker/model.py:134-159 and kernels/kvcache_mgmt.py - monolithic uniform FP16 KV layout blocks per-layer dynamic capacity without kernel/layout redesign.",
    "high: swiftllm/worker/weight.py:30-49,189-269 - eager dense CUDA/FP16 assumptions are incompatible with AWQ packed INT4.",
    "high: csrc/src/block_swapping.cpp:20-99 - copies are enqueued asynchronously but orchestration serializes swaps and host tensors are not explicitly pinned."
  ],
  "manualNotes": "Source was inspected from the local pristine snapshot whose UPSTREAM_COMMIT matches remote master; remote refs/history were independently queried."
}
```
