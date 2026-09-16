#!/usr/bin/env python3
"""Audit pinned bytes required for all Llama 3.1 8B FP16+W4 decoder variants."""

import json
import resource
from pathlib import Path

LAYERS = 32
FP16_LAYER_BYTES = 436_224_000
W4_LAYER_BYTES = 113_328_128

fp16 = LAYERS * FP16_LAYER_BYTES
w4 = LAYERS * W4_LAYER_BYTES
soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
payload = {
    "schema_version": 1,
    "model": "local Llama 3.1 8B",
    "uniform_decoder_layers": LAYERS,
    "measured_layer_31": {
        "fp16_region_bytes": FP16_LAYER_BYTES,
        "w4_bytes_including_norms_qweight_qzeros_scales": W4_LAYER_BYTES,
    },
    "all_decoder_variants": {
        "fp16_bytes": fp16,
        "w4_bytes": w4,
        "combined_bytes": fp16 + w4,
        "combined_gib": (fp16 + w4) / 2**30,
    },
    "rlimit_memlock": {"soft_bytes": soft, "hard_bytes": hard, "soft_gib": soft / 2**30},
    "excess_over_soft_bytes": fp16 + w4 - soft,
    "feasible_simultaneously_pinned": fp16 + w4 <= soft,
    "classification": "resource blocker for paper-style simultaneous pinned full+W4 decoder variants" if fp16 + w4 > soft else "feasible",
    "limitations": [
        "Does not include allocator alignment beyond measured per-layer regions.",
        "Does not include embeddings, lm_head, KV cache, staging buffers, or runtime overhead.",
        "Changing RLIMIT_MEMLOCK is not authorized by this project.",
    ],
}
out = Path("reproduction/results/raw/full-profile-memory-feasibility.json")
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
