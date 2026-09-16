# Active-KV same-history switching result

## Result

All predeclared gates passed for one active Llama 3.1 8B request:

- one FP16 prefill (9 tokens), four layer-31 W4 decode steps, then one FP16 decode;
- no re-prefill, model reload, scheduler restart, KV quantization, or KV eviction;
- prompt KV cache unchanged at switch;
- the fourth W4 decode allocated virtual block 3 from the reclaimed layer tail, which exposed 615 block slots at block size 4;
- each in-place W4 step matched the same-history separately allocated W4 reference top-1, with relative logit L2 0.00058–0.00088;
- reclaimed K/V block matched the reference original block within relative L2 0.000095/0.000207;
- occupied reclaimed K/V migrated byte-exactly to physical original block 3 before FP16 restore; block-table IDs remained `[0,1,2,3]`;
- all 23 packed and 8 restored layer tensors were exact;
- final FP16 top-1 matched the same-history reference, relative logit L2 0.000354 (<0.01).

The forced token/precision history was `[505,279,3363,11,304]` with schedule FP16 prefill → W4×4 → FP16×1.

## Classification

**Approximate/modified-condition state-preservation reproduction.** This is the strongest current evidence for the joint mechanism: real W4, physical KV attachment, active sequence continuity, same-history comparison, event-safe restore, and exact migration without flush/eviction.

Important limits:

- occupied-block migration is an independent reconstruction; candidate code only waits for reclaimed blocks to become unused;
- one request, one W4 layer, one migrated block, forced shared tokens, block size 4;
- no concurrent arrivals, queueing, controller, ordinary preemption, or overlap timing;
- local AutoAWQ/checkpoint/RTX 3090, not paper artifact/hardware.

## Timing caveat

First candidate decode emitted multi-second per-layer logs from Triton JIT compilation for the new full-model/block-size shape. Runner wall time was 43.8 s including model load and JIT. The paper requires warmup/precompilation, so these are initialization artifacts and excluded from steady-state claims.

## Evidence

- `results-attempt-2/metrics.json`, `verification.log`
- `results-attempt-2/run.{stdout,stderr}.log`, `run.exitcode`
- `scripts/active_kv_switch.py`
