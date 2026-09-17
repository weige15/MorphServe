#!/usr/bin/env python3
"""Create the frozen deterministic 1024-token system-benchmark payload."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=20260917)
    p.add_argument("--tokens", type=int, default=1024)
    args = p.parse_args()
    if args.tokens != 1024:
        raise SystemExit("this protocol requires exactly 1024 prompt tokens")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    ids = torch.randint(0, int(tokenizer.vocab_size), (args.tokens,), generator=generator, dtype=torch.int64).tolist()
    if len(ids) != 1024 or any(not 0 <= int(x) < tokenizer.vocab_size for x in ids):
        raise RuntimeError("generated token payload is not valid")
    encoded = json.dumps(ids, separators=(",", ":")).encode()
    result = {
        "schema_version": 1,
        "classification": "frozen synthetic system-workload input payload",
        "seed": args.seed,
        "token_count": len(ids),
        "token_ids": ids,
        "payload_sha256": hashlib.sha256(encoded).hexdigest(),
        "tokenizer_path": str(Path(args.model).resolve()),
        "tokenizer_vocab_size": int(tokenizer.vocab_size),
        "tokenizer_revision": getattr(tokenizer, "name_or_path", str(args.model)),
        "eos_dependent": False,
        "generation_policy": {"tokens_per_request": 512, "stop_on_eos": False, "forced_length": True},
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("token_count", "payload_sha256", "seed")}, indent=2))


if __name__ == "__main__":
    main()
