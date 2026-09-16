# Protocol: two-region Triton KV mapping

Status: pre-registered before execution.

## Question

Do the candidate custom Triton KV-store and PagedAttention kernels correctly remap virtual block IDs across the original cache plus two disjoint reclaimed layer tails, while preserving prefill/decode content and attention output?

## Frozen layout

- Repaired native-metadata extension; candidate Python kernel files remain byte-identical to commit `85c4fbf...`.
- One contiguous 8,192-byte owner represents two adjacent 4,096-byte FP16 layer regions.
- Each layer is replaced by 1,024 packed bytes; each 3,072-byte tail exposes 12 K/V blocks (`num_layers=2`, `num_kv_heads=1`, `block_size=4`, `head_dim=8`).
- Swap/register the higher-address layer first, matching candidate back-to-front runtime order. Group 0 points into its tail; group 1 must be reached by subtracting one 4,096-byte layer stride from the group-0 pointer.
- Original cache has 2 blocks. Virtual IDs are original block `0`, group-0 block `2`, and group-1 blocks `14` and `15`; block table is `[0,2,14,15]` for a 13-token sequence.

## Required checks

1. Candidate prefill store puts tokens 0–3 in original block 0, 4–7 in reclaimed group 0, and 8–11 in reclaimed group 1 exactly.
2. Candidate decode store puts token 12 in group-1 block 1 exactly.
3. Candidate combined PagedAttention reads all 13 tokens and matches an independent dense PyTorch softmax-attention oracle (`atol=rtol=0.02`, FP16 output).
4. Sentinel bytes in packed prefixes remain unchanged after store/attention.
5. All K/V view bounds remain inside their corresponding layer regions; allocator delta remains zero.
6. Process exits 0 and vendor manifests remain unchanged.

## Boundaries

This tests the candidate's fixed-stride, back-to-front two-region mapping only. It does not establish arbitrary allocator fragmentation, GQA/MHA coverage beyond one head, occupied shrink, in-flight event safety, repeated oscillation, or full-model quality/latency.
