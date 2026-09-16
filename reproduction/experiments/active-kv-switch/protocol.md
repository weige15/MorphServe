# Protocol: active-KV same-history FP16/W4/FP16 switching

Status: pre-registered before implementation/execution.

## Question

Can one active Llama 3.1 8B request continue through FP16 prefill → four one-layer-W4 decode steps → FP16 decode without re-prefill, while a newly needed logical KV block is physically stored in reclaimed layer memory and safely migrated before FP16 restore?

## Same-history reference

Use one loaded normalized candidate model and fork from a byte-for-byte snapshot immediately after FP16 prefill:

- Use PagedAttention block size 4, so the 9-token prompt occupies original blocks 0–2 and the fourth W4 decode allocates block 3.
- Reference branch: layer-31 AutoAWQ tensors live in separate GPU allocations; the fourth W4 decode allocates original KV block 3.
- In-place branch: reset to the same prefill snapshot, install layer 31 in-place, reduce original capacity to the three occupied prompt blocks, attach the reclaimed tail, and force the fourth W4 decode to allocate virtual block 3 in reclaimed group 0.
- Feed the reference branch's greedy token at every step to both branches, so precision/token history is identical even if atomic W4 logits vary.

Before FP16 restoration, copy the occupied reclaimed block to the otherwise unused physical original block 3, switch the mapping boundary back to four original blocks, release reclaimed views, wait on recorded lifetime events, restore FP16, and continue one decode step. This migration is a labeled reconstruction choice because candidate code only waits for reclaimed blocks to become unused.

## Gate

1. No second full model, scheduler restart, re-prefill, KV quantization, or KV eviction.
2. In-place branch allocates virtual block 3 from reclaimed memory; preexisting prompt KV bytes are unchanged at switch.
3. At each W4 step, reference/in-place top-1 matches and relative logit L2 `<0.005` (independent AutoAWQ repeat envelope).
4. Reclaimed block K/V after four W4 steps matches the same-history reference original block with relative L2 `<0.005` per K and V.
5. Migration is byte-exact from reclaimed source to original block 3; block table still maps the active request's logical block to physical ID 3.
6. After FP16 restore, final reference/in-place top-1 matches and relative logit L2 `<0.01` (predeclared allowance for propagated W4 atomic variance).
7. FP16 layer bytes restore exactly; process exits 0; vendor manifests unchanged.

## Boundaries

One request, one quantized layer, one migrated block, greedy forced-token history, no concurrent arrivals or overlap timing. This is modified-condition state-preservation evidence, not the paper controller or headline quality/latency result.
