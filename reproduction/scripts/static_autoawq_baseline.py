#!/usr/bin/env python3
"""Run a real packed AutoAWQ W4 numerical baseline."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(reference, candidate):
    reference = reference.float()
    candidate = candidate.float()
    delta = candidate - reference
    return {
        "shape_reference": list(reference.shape),
        "shape_candidate": list(candidate.shape),
        "finite_reference": bool(torch.isfinite(reference).all()),
        "finite_candidate": bool(torch.isfinite(candidate).all()),
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "relative_l2": float(delta.norm() / reference.norm().clamp_min(1e-12)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16-model", required=True)
    parser.add_argument("--w4-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="Life blooms like a flower, far away")
    args = parser.parse_args()
    fp16_path = Path(args.fp16_model).resolve()
    w4_path = Path(args.w4_model).resolve()

    tokenizer = AutoTokenizer.from_pretrained(fp16_path, local_files_only=True)
    token_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.cuda()

    fp16 = AutoModelForCausalLM.from_pretrained(
        fp16_path,
        local_files_only=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).cuda().eval()
    with torch.inference_mode():
        fp16_logits = fp16(token_ids, use_cache=False, return_dict=True).logits[0, -1].detach().cpu()
    del fp16
    gc.collect()
    torch.cuda.empty_cache()

    from awq import AutoAWQForCausalLM, __version__ as autoawq_version
    from awq.modules.linear.gemm import WQLinear_GEMM

    w4_wrapper = AutoAWQForCausalLM.from_quantized(
        str(w4_path),
        max_seq_len=2048,
        trust_remote_code=False,
        fuse_layers=False,
        use_exllama=False,
        use_exllama_v2=False,
        safetensors=True,
        device_map={"": 0},
    )
    w4 = w4_wrapper.model.eval()
    packed_modules = []
    storage = {"qweight": 0, "qzeros": 0, "scales": 0, "bias": 0}
    dtype_sets = {key: set() for key in storage}
    for name, module in w4.named_modules():
        if not isinstance(module, WQLinear_GEMM):
            continue
        packed_modules.append(name)
        for key in ("qweight", "qzeros", "scales"):
            tensor = getattr(module, key)
            storage[key] += tensor.numel() * tensor.element_size()
            dtype_sets[key].add(str(tensor.dtype))
        if module.bias is not None:
            storage["bias"] += module.bias.numel() * module.bias.element_size()
            dtype_sets["bias"].add(str(module.bias.dtype))

    unique_tensors = {}
    for tensor in w4.state_dict().values():
        unique_tensors.setdefault(tensor.data_ptr(), tensor.numel() * tensor.element_size())

    with torch.inference_mode():
        repeats = [
            w4(token_ids, use_cache=False, return_dict=True).logits[0, -1].detach().cpu()
            for _ in range(5)
        ]
    first, second = repeats[0], repeats[-1]
    repeat_diagnostics = [stats(first, value) for value in repeats]
    comparison = stats(fp16_logits, second)
    reference_top1 = int(fp16_logits.argmax())
    w4_top1 = int(second.argmax())
    gate = {
        "packed_modules_present": len(packed_modules) > 0,
        "qweight_int32": dtype_sets["qweight"] == {"torch.int32"},
        "qzeros_int32": dtype_sets["qzeros"] == {"torch.int32"},
        "scales_fp16": dtype_sets["scales"] == {"torch.float16"},
        "repeat_logits_exact": all(torch.equal(first, value) for value in repeats[1:]),
        "finite": comparison["finite_reference"] and comparison["finite_candidate"],
        "shape_match": comparison["shape_reference"] == comparison["shape_candidate"],
    }
    config = json.loads((w4_path / "config.json").read_text())
    payload = {
        "schema_version": 1,
        "classification": "modified-condition-static-autoawq-w4",
        "paths": {"fp16": str(fp16_path), "w4": str(w4_path)},
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(w4_path.glob("*.safetensors"))
        },
        "config": {
            "sha256": sha256(w4_path / "config.json"),
            "quantization_config": config["quantization_config"],
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "transformers": __import__("transformers").__version__,
            "autoawq": autoawq_version,
        },
        "prompt": args.prompt,
        "token_ids": token_ids[0].cpu().tolist(),
        "packed": {
            "module_class": f"{WQLinear_GEMM.__module__}.{WQLinear_GEMM.__name__}",
            "module_count": len(packed_modules),
            "module_examples": packed_modules[:10],
            "storage_bytes": storage,
            "decoder_packed_bytes_including_metadata": sum(storage.values()),
            "dtypes": {key: sorted(values) for key, values in dtype_sets.items()},
            "total_unique_state_bytes": sum(unique_tensors.values()),
        },
        "logits": {
            **comparison,
            "fp16_top1": reference_top1,
            "w4_top1": w4_top1,
            "top1_match": reference_top1 == w4_top1,
            "fp16_top5": [int(x) for x in fp16_logits.topk(5).indices],
            "w4_top5": [int(x) for x in second.topk(5).indices],
            "repeat_exact": gate["repeat_logits_exact"],
        },
        "repeat_diagnostics": {
            "comparisons_to_first": repeat_diagnostics,
            "top1": [int(value.argmax()) for value in repeats],
            "top5": [[int(x) for x in value.topk(5).indices] for value in repeats],
        },
        "gate": gate,
        "passed": all(gate.values()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gate": gate, "packed": payload["packed"], "logits": payload["logits"]}, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
