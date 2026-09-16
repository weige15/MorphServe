#!/usr/bin/env python3
"""Compare normalized candidate FP16 next-token logits with Transformers."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import time
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


def tensor_stats(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tensor_loader(model_path: Path):
    files = sorted(model_path.glob("*.safetensors"))
    index_path = model_path / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"] if index_path.exists() else {}
    handles = {}

    def get(key):
        filename = weight_map.get(key, files[0].name)
        path = model_path / filename
        if path not in handles:
            handles[path] = safe_open(str(path), framework="pt", device="cpu")
        return handles[path].get_tensor(key).to(torch.float16)

    return get, handles


def audit_weights(model, model_path: Path) -> dict:
    get, handles = source_tensor_loader(model_path)
    checks = []

    def check(name, actual, key):
        expected = get(key)
        actual_cpu = actual.detach().cpu()
        checks.append({
            "name": name,
            "source_key": key,
            "shape": list(actual_cpu.shape),
            "exact": bool(torch.equal(actual_cpu, expected)),
        })

    check("wte", model.weight.wte, "model.embed_tokens.weight")
    lm_key = "model.embed_tokens.weight" if model.model_config.tie_word_embeddings else "lm_head.weight"
    check("lm_head", model.weight.lm_head, lm_key)
    check("final_norm", model.weight.final_norm, "model.norm.weight")
    for layer_id, layer in enumerate(model.weight.layers):
        prefix = f"model.layers.{layer_id}"
        check(f"layers.{layer_id}.attn_norm", layer.attn_norm, f"{prefix}.input_layernorm.weight")
        check(f"layers.{layer_id}.q_proj", layer.q_proj, f"{prefix}.self_attn.q_proj.weight")
        check(f"layers.{layer_id}.k_proj", layer.k_proj, f"{prefix}.self_attn.k_proj.weight")
        check(f"layers.{layer_id}.v_proj", layer.v_proj, f"{prefix}.self_attn.v_proj.weight")
        check(f"layers.{layer_id}.o_proj", layer.o_proj, f"{prefix}.self_attn.o_proj.weight")
        check(f"layers.{layer_id}.ffn_norm", layer.ffn_norm, f"{prefix}.post_attention_layernorm.weight")
        split = model.model_config.ffn_inter_dim
        check(f"layers.{layer_id}.up_proj", layer.up_gate_proj[:split], f"{prefix}.mlp.up_proj.weight")
        check(f"layers.{layer_id}.gate_proj", layer.up_gate_proj[split:], f"{prefix}.mlp.gate_proj.weight")
        check(f"layers.{layer_id}.down_proj", layer.down_proj, f"{prefix}.mlp.down_proj.weight")
    for handle in handles.values():
        handle.__exit__(None, None, None)
    mismatches = [row for row in checks if not row["exact"]]
    return {"checked": len(checks), "mismatches": len(mismatches), "examples": mismatches[:10]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="Life blooms like a flower, far away")
    args = parser.parse_args()
    model_path = Path(args.model_path).resolve()

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    token_ids = tokenizer(args.prompt, return_tensors="pt").input_ids[0]

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    hf_start = time.perf_counter()
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).cuda().eval()
    with torch.inference_mode():
        hf_logits = hf_model(token_ids.cuda().unsqueeze(0), use_cache=False, return_dict=True).logits[0, -1].detach().cpu()
    torch.cuda.synchronize()
    hf_wall = time.perf_counter() - hf_start
    hf_peak = torch.cuda.max_memory_allocated()
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()

    from swiftllm.engine_config import EngineConfig
    from swiftllm.worker.model import LlamaModel

    config = EngineConfig(
        org_model_path=str(model_path),
        use_dummy=False,
        num_cpu_blocks=0,
        max_seqs_in_block_table=8,
        max_blocks_per_seq=2048,
        max_batch_size=8,
        max_tokens_in_batch=2048,
    )
    torch.cuda.reset_peak_memory_stats()
    candidate_start = time.perf_counter()
    candidate_model = LlamaModel(config, str(model_path))
    candidate_model.load_weights()
    candidate_logits_gpu = candidate_model.forward(
        [token_ids.tolist()], [0], [], ignore_kvcache=True, return_logits=True
    )[0]
    torch.cuda.synchronize()
    candidate_wall = time.perf_counter() - candidate_start
    candidate_peak = torch.cuda.max_memory_allocated()
    candidate_logits = candidate_logits_gpu.detach().cpu()
    weight_audit = audit_weights(candidate_model, model_path)

    stats = tensor_stats(hf_logits, candidate_logits)
    reference_top1 = int(hf_logits.argmax())
    candidate_top1 = int(candidate_logits.argmax())
    gate = {
        "finite": stats["finite_reference"] and stats["finite_candidate"],
        "shape_match": stats["shape_reference"] == stats["shape_candidate"],
        "top1_match": reference_top1 == candidate_top1,
        "relative_l2_below_0.005": stats["relative_l2"] < 0.005,
        "weights_exact": weight_audit["mismatches"] == 0,
    }

    runtime_root = Path(__file__).resolve().parents[1] / "runtime" / "candidate-python"
    source_files = sorted(runtime_root.rglob("*.py"))
    payload = {
        "schema_version": 1,
        "classification": "modified-condition-fp16-baseline",
        "model": {
            "path": str(model_path),
            "snapshot": model_path.name,
            "config_sha256": sha256(model_path / "config.json"),
        },
        "prompt": args.prompt,
        "token_ids": token_ids.tolist(),
        "token_text": tokenizer.convert_ids_to_tokens(token_ids.tolist()),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "transformers": __import__("transformers").__version__,
            "runtime_source_sha256": hashlib.sha256(b"".join(path.read_bytes() for path in source_files)).hexdigest(),
        },
        "transformers": {"attention": "eager", "wall_seconds_including_load": hf_wall, "peak_allocated_bytes": hf_peak},
        "candidate": {"ignore_kvcache": True, "wall_seconds_including_load": candidate_wall, "peak_allocated_bytes": candidate_peak},
        "logits": {
            **stats,
            "reference_top1": reference_top1,
            "candidate_top1": candidate_top1,
            "reference_top5": [int(x) for x in hf_logits.topk(5).indices],
            "candidate_top5": [int(x) for x in candidate_logits.topk(5).indices],
        },
        "weight_audit": weight_audit,
        "gate": gate,
        "passed": all(gate.values()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "gate": gate, "logits": payload["logits"], "weight_audit": weight_audit}, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
