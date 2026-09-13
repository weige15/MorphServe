# Packed INT4 backend v6 — Phase B integration

Status: **complete; proceed to the full-shape resource/performance gate**.

## Offline artifact

Public AutoAWQ 0.2.9 quantized the exact local Llama-3.1-8B base checkpoint
once using the frozen W4/group-128/asymmetric-zero-point configuration. The
calibration input was the pinned Pile validation revision, AutoAWQ shuffle seed
42, 128 accepted samples (30,511 concatenated tokens; 59 complete 512-token
blocks), duo scaling, and clipping. Quantization completed all 32 layers before
a manifest-finalization bug looked for a standalone `quant_config.json` that
AutoAWQ does not emit; the script was corrected to read
`config.json.quantization_config` and finalized hashes without repeating
quantization.

The external checkpoint is 5,745,260,850 bytes at
`/nfs/home/s314511048/.cache/morphserve/llama31-8b-autoawq-w4-g128-zp`.
Per-file SHA-256 hashes, emitted config, calibration hash, and tool versions are
in `phase-b/awq-checkpoint-manifest.json`. The exact accepted calibration token
IDs are retained in `phase-b/calibration-selected.jsonl`.

## Minimal integration

The implementation adds two engine fields: `quantization_backend` and
`quantized_model_path`. Defaults preserve the existing NF4 behavior. For an
AWQ-selected front-to-back layer, the loader:

1. loads the layer's AutoAWQ-scaled norms plus each matrix's `qweight`,
   `qzeros`, and FP16 `scales` from the offline artifact;
2. applies vLLM 0.11.2's one-time AWQ-to-Marlin repack at model load;
3. discards source-layout temporaries and retains only Marlin qweight, scales,
   zero points, empty g-index tensors, and a 328-byte workspace;
4. dispatches every activation row count through `gptq_marlin_gemm` with no
   decode/prefill representation duplication and no full-weight dequantization.

Unselected decoder layers, embeddings, final norm, and LM head continue to load
from the exact base FP16 checkpoint. Scheduler, KV allocation, attention math,
request timing, and the legacy FP16/NF4 paths are unchanged. The only attention
compatibility edit passes existing FlashAttention arguments by keyword and
falls back to vLLM's bundled FA2 module when the old external extension cannot
load against torch 2.9; matched torch-2.9 FP16/Hugging-Face parity passed all 8
greedy tokens, the first token, and 9/10 first-step top-k overlap.

## Packed mapping and numerical checks

`phase-b/awq-checkpoint-sanity.json` checks all 224 decoder matrices and all 672
component tensors. Every component key, dtype, and shape matches the expected
AutoAWQ GEMM layout. For all seven layer-0 projections, plus the fused up+gate
representation, Marlin output was finite and agreed with public AutoAWQ dense
dequantization at worst 0.000334 relative L2 error.

The four 8-token short-generation smokes (`phase-b/smoke/`) all completed and
produced the same finite/sane continuation:

`[20437, 13, 578, 6864, 315, 15704, 374, 22463]` —
“ Berlin. The capital of Italy is Rome”.

The smokes record every layer's selected flag, runtime type, packed source
keys, norm source, and fused MLP source keys. In W4-8, layers 0–7 are packed
AWQ-Marlin with AWQ norms and layers 8–31 are base FP16 tensors. The same rule
holds at 16 and 32 layers. Packed representation bytes are 906,509,696,
1,813,019,392, and 3,626,038,784 for W4-8/16/32.

## MLP fusion finding

The initial correct but separate AWQ up/gate execution exposed a hidden
full-shape activation cost: W4-8's 49,152-token profile used 6.387 GiB above
persistent allocation and left 1,711 KV blocks. This reproduced the same
capacity pathology as v5 even though Marlin itself did not dequantize weights.
The cause was two 1.31-GiB projection outputs plus the concatenated 2.63-GiB
`[up,gate]` buffer.

The loader now concatenates the two AutoAWQ packed output dimensions before the
one-time Marlin repack, preserving exact `[up,gate]` ordering and using one
Marlin launch/representation. The numerical check above covers this fused
mapping. Under the identical W4-8 full-shape probe, temporary allocation fell
to 4.145 GiB and safe blocks rose from 1,711 to 3,034. Persistent allocation
and packed bytes were effectively unchanged. Both exploratory profiles are
retained under `phase-b/`; only the fused path advances to Phase C.

## Validation

- Original torch-2.5 environment: 16 unit tests passed, including unchanged
  NF4 dispatch and a fresh two-token NF4 smoke.
- Compatible torch-2.9/vLLM environment: the same 16 unit tests passed.
- FP16/Hugging-Face parity in the execution environment: exact 8-token greedy
  match.
- Offline checkpoint audit: 672/672 component checks passed; all packed
  numerical checks finite.
- 0/8/16/32 fused-path short generations: all completed with 8 finite IDs.
- Python compilation and `git diff --check`: passed.

No controller, scheduler, runtime swapping, or KV semantics were added.
