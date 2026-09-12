"""Compare SwiftLLM FP16 greedy inference with Hugging Face Transformers.

This intentionally uses the same checkpoint and tokenizer, a short prompt, and
one GPU process.  It is a correctness gate, not a serving benchmark.  The
SwiftLLM side runs the data plane directly so scheduler timing cannot hide a
model mismatch; the same model.forward call also exposes first-step logits for
a top-k numerical comparison.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from swiftllm.engine_config import EngineConfig
from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.model import LlamaModel
from swiftllm.worker.weight import _checkpoint_weight_keys, resolve_lm_head_key


DEFAULT_PROMPT = (
    "The capital of France is Paris. The capital of Germany is"
)


def _topk(logits: torch.Tensor, k: int) -> list[dict[str, float | int]]:
    values, indices = torch.topk(logits.float(), k=k, dim=-1)
    return [
        {"token_id": int(index), "logit": float(value)}
        for index, value in zip(indices.tolist(), values.tolist())
    ]


def _tokenizer_identity(tokenizer: Any, model_path: Path) -> dict[str, Any]:
    files = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        path = model_path / name
        if path.exists():
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "name_or_path": tokenizer.name_or_path,
        "vocab_size": len(tokenizer),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "files_sha256": files,
    }


def _reference_generation(
    model_path: Path,
    tokenizer: Any,
    prompt_ids: list[int],
    max_new_tokens: int,
    top_k: int,
) -> dict[str, Any]:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).cuda()
    model.eval()
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
    generated: list[int] = []
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True)
        first_logits = output.logits[0, -1].detach()
        next_token = int(torch.argmax(first_logits).item())
        past = output.past_key_values
        eos_ids = tokenizer.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = {eos_ids}
        else:
            eos_ids = set(eos_ids or [])
        for _ in range(max_new_tokens):
            generated.append(next_token)
            if len(generated) >= max_new_tokens:
                break
            if next_token in eos_ids:
                break
            output = model(
                input_ids=torch.tensor([[next_token]], device="cuda"),
                past_key_values=past,
                use_cache=True,
            )
            next_token = int(torch.argmax(output.logits[0, -1]).item())
            past = output.past_key_values
    result = {
        "first_token_id": int(generated[0]),
        "output_token_ids": generated,
        "decoded": tokenizer.decode(generated, skip_special_tokens=True),
        "first_logits_topk": _topk(first_logits, top_k),
    }
    del model, output, past
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _swift_generation(
    model_path: Path,
    prompt_ids: list[int],
    max_new_tokens: int,
    top_k: int,
) -> dict[str, Any]:
    config = EngineConfig(
        model_path=str(model_path),
        use_dummy=False,
        block_size=16,
        gpu_mem_utilization=0.90,
        num_cpu_blocks=1,
        max_seqs_in_block_table=4,
        max_blocks_per_seq=256,
        max_batch_size=1,
        max_tokens_in_batch=max(len(prompt_ids), 16),
        quantized_layer_count=0,
    )
    model = LlamaModel(config)
    model.load_weights()
    num_blocks = model.profile_num_blocks()
    model.init_kvcache_and_swap(num_blocks)
    try:
        with torch.inference_mode():
            first_logits = model.forward(
                [prompt_ids], [0], [], return_logits=True
            )[0]
            generated: list[int] = [int(torch.argmax(first_logits).item())]
            for _ in range(1, max_new_tokens):
                if generated[-1] in (128001, 2):
                    break
                logits = model.forward(
                    [[generated[-1]]],
                    [0],
                    [len(prompt_ids) + len(generated)],
                    return_logits=True,
                )[0]
                generated.append(int(torch.argmax(logits).item()))
        return {
            "first_token_id": generated[0],
            "output_token_ids": generated,
            "first_logits_topk": _topk(first_logits, top_k),
            "num_gpu_blocks": num_blocks,
        }
    finally:
        model.free_seqs_resources([0])
        del model
        gc.collect()
        torch.cuda.empty_cache()


def run(model_path: Path, prompt: str, max_new_tokens: int, top_k: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("inference parity requires CUDA")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompt_ids = tokenizer(prompt, return_attention_mask=False)["input_ids"]
    model_config = LlamaModelConfig.load_from_model_path(str(model_path))
    keys = _checkpoint_weight_keys(str(model_path))
    lm_head_key = resolve_lm_head_key(str(model_path), model_config, keys)
    reference = _reference_generation(
        model_path, tokenizer, prompt_ids, max_new_tokens, top_k
    )
    swift = _swift_generation(model_path, prompt_ids, max_new_tokens, top_k)
    swift["decoded"] = tokenizer.decode(
        swift["output_token_ids"], skip_special_tokens=True
    )
    common = 0
    for left, right in zip(reference["output_token_ids"], swift["output_token_ids"]):
        if left != right:
            break
        common += 1
    reference_top = [item["token_id"] for item in reference["first_logits_topk"]]
    swift_top = [item["token_id"] for item in swift["first_logits_topk"]]
    return {
        "schema_version": 1,
        "checkpoint": str(model_path),
        "checkpoint_config": {
            "tie_word_embeddings": model_config.tie_word_embeddings,
            "rope_scaling": model_config.rope_scaling,
            "rope_theta": model_config.rope_theta,
            "num_hidden_layers": model_config.num_layers,
            "hidden_size": model_config.hidden_size,
            "vocab_size": model_config.vocab_size,
            "resolved_lm_head_key": lm_head_key,
            "weight_key_count": len(keys) if keys is not None else None,
        },
        "tokenizer": _tokenizer_identity(tokenizer, model_path),
        "prompt": prompt,
        "prompt_token_ids": prompt_ids,
        "reference": reference,
        "swiftllm_fp16": swift,
        "first_token_match": reference["first_token_id"] == swift["first_token_id"],
        "first_topk_overlap": len(set(reference_top) & set(swift_top)),
        "greedy_common_prefix_tokens": common,
        "greedy_exact_match": reference["output_token_ids"] == swift["output_token_ids"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.model_path, args.prompt, args.max_new_tokens, args.top_k)
    rendered = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
