"""Quantize the fixed Llama checkpoint once with public AutoAWQ 0.2.9."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import torch
from awq import AutoAWQForCausalLM
from datasets import load_dataset
from transformers import AutoTokenizer


DATASET = "mit-han-lab/pile-val-backup"
DATASET_REVISION = "2f5e46ae6a69cf0dce4b12f78241c408936ca0e4"
DATASET_SPLIT = "validation"
SHUFFLE_SEED = 42
MAX_SAMPLES = 128
MAX_SEQUENCE_LENGTH = 512
MAX_CHUNK_MEMORY = 512 * 1024 * 1024
QUANT_CONFIG = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="finalize hashes after a completed export without quantizing again",
    )
    args = parser.parse_args()

    started = time.time()
    if args.manifest_only:
        if not args.output_checkpoint.is_dir() or not args.calibration_output.is_file():
            raise FileNotFoundError(
                "manifest-only mode requires completed checkpoint and calibration files"
            )
        selected_rows = [
            json.loads(line)
            for line in args.calibration_output.read_text(encoding="utf-8").splitlines()
        ]
        selected = [row["token_ids"] for row in selected_rows]
    else:
        if args.output_checkpoint.exists():
            raise FileExistsError(
                f"refusing to repeat or overwrite offline quantization: {args.output_checkpoint}"
            )
        temporary_output = args.output_checkpoint.with_name(
            args.output_checkpoint.name + ".incomplete"
        )
        if temporary_output.exists():
            shutil.rmtree(temporary_output)

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        dataset = load_dataset(
            DATASET,
            split=DATASET_SPLIT,
            revision=DATASET_REVISION,
        ).shuffle(seed=SHUFFLE_SEED)
        selected: list[list[int]] = []
        selected_rows: list[dict] = []
        for shuffled_index, row in enumerate(dataset):
            text = row["text"].strip()
            token_ids = tokenizer.encode(text)
            if not token_ids or len(token_ids) > MAX_SEQUENCE_LENGTH:
                continue
            selected.append(token_ids)
            selected_rows.append(
                {
                    "accepted_index": len(selected_rows),
                    "shuffled_index": shuffled_index,
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "token_ids": token_ids,
                }
            )
            if len(selected) == MAX_SAMPLES:
                break
        if len(selected) != MAX_SAMPLES:
            raise RuntimeError(
                f"only found {len(selected)} accepted calibration samples"
            )

        args.calibration_output.parent.mkdir(parents=True, exist_ok=True)
        with args.calibration_output.open("w", encoding="utf-8") as handle:
            for row in selected_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

        model = AutoAWQForCausalLM.from_pretrained(
            str(args.model_path),
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            use_cache=False,
        )
        model.quantize(
            tokenizer,
            quant_config=QUANT_CONFIG,
            calib_data=selected,
            duo_scaling=True,
            apply_clip=True,
            n_parallel_calib_samples=1,
            max_calib_samples=MAX_SAMPLES,
            max_calib_seq_len=MAX_SEQUENCE_LENGTH,
            max_chunk_memory=MAX_CHUNK_MEMORY,
        )
        model.save_quantized(str(temporary_output), safetensors=True, shard_size="4GB")
        tokenizer.save_pretrained(temporary_output)
        os.replace(temporary_output, args.output_checkpoint)
        torch.cuda.synchronize()

    import awq
    import datasets
    import transformers

    files = []
    for path in sorted(args.output_checkpoint.iterdir()):
        if path.is_file():
            files.append(
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    emitted_config = json.loads(
        (args.output_checkpoint / "config.json").read_text(encoding="utf-8")
    )
    emitted_quant_config = emitted_config["quantization_config"]
    manifest = {
        "schema_version": 1,
        "protocol_manifest": "benchmark-results/packed-int4-backend-v6/protocol_manifest.json",
        "base_checkpoint": str(args.model_path),
        "base_config_sha256": sha256(args.model_path / "config.json"),
        "base_weight_index_sha256": sha256(
            args.model_path / "model.safetensors.index.json"
        ),
        "output_checkpoint": str(args.output_checkpoint),
        "quantization": QUANT_CONFIG,
        "emitted_quantization_config": emitted_quant_config,
        "calibration": {
            "dataset": DATASET,
            "revision": DATASET_REVISION,
            "split": DATASET_SPLIT,
            "shuffle_seed": SHUFFLE_SEED,
            "accepted_samples": MAX_SAMPLES,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "concatenated_tokens": sum(len(row) for row in selected),
            "full_512_token_blocks": sum(len(row) for row in selected)
            // MAX_SEQUENCE_LENGTH,
            "selected_jsonl": str(args.calibration_output),
            "selected_jsonl_sha256": sha256(args.calibration_output),
            "duo_scaling": True,
            "apply_clip": True,
            "n_parallel_calib_samples": 1,
            "max_chunk_memory_bytes": MAX_CHUNK_MEMORY,
        },
        "software": {
            "python": os.sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "autoawq": getattr(awq, "__version__", "0.2.9"),
        },
        "device": {
            "name": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "elapsed_seconds": None if args.manifest_only else time.time() - started,
        "manifest_finalized_from_existing_export": args.manifest_only,
        "files": files,
        "total_checkpoint_bytes": sum(row["bytes"] for row in files),
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
