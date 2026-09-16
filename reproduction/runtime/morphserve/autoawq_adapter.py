"""Independent AutoAWQ layer adapter for the normalized candidate runtime."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open


FP16_ATTRIBUTES = (
    "attn_norm",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "ffn_norm",
    "up_gate_proj",
    "down_proj",
)
QUANT_MODULES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


class AsyncLayerCopier:
    """Queue pinned layer copies and expose same-address typed GPU views."""

    def __init__(self, model, extension):
        self.model = model
        self.extension = extension
        self.stream = torch.cuda.Stream()

    def enqueue(self, layer_id: int, source: torch.Tensor, size: int, tensor_map: list[dict]):
        if source.device.type != "cpu" or not source.is_pinned():
            raise ValueError("layer source must be pinned CPU memory")
        region = self.extension.get_layer_memory_org_gpu(layer_id)
        if size > source.numel() or size > region.numel():
            raise ValueError("layer copy exceeds source or destination region")
        with torch.cuda.stream(self.stream):
            previous = self.model.layer_transfer_events.get(layer_id)
            if previous is not None:
                self.stream.wait_event(previous)
            if self.model.last_forward_event is not None:
                self.stream.wait_event(self.model.last_forward_event)
            region[:size].copy_(source[:size], non_blocking=True)
            ready = torch.cuda.Event(enable_timing=True)
            ready.record()
        self.model.layer_transfer_events[layer_id] = ready
        return self.views(layer_id, tensor_map, region), ready

    def views(self, layer_id: int, tensor_map: list[dict], region=None):
        region = self.extension.get_layer_memory_org_gpu(layer_id) if region is None else region
        views = []
        for entry in tensor_map:
            _, info = next(iter(entry.items()))
            view = region[info["offset"]:info["offset"] + info["size"]]
            views.append(view.view(info["dtype"]).view(info["shape"]))
        return views

    def wait_current(self, layer_id: int) -> bool:
        ready = self.model.layer_transfer_events.pop(layer_id, None)
        if ready is None:
            return False
        torch.cuda.current_stream().wait_event(ready)
        return True


def _tensor_info(name: str, tensor: torch.Tensor, offset: int) -> dict:
    return {name: {
        "offset": offset,
        "shape": list(tensor.shape),
        "dtype": tensor.dtype,
        "size": tensor.numel() * tensor.element_size(),
        "is_param": True,
    }}


def _checkpoint_getter(model_path: Path):
    index = json.loads((model_path / "model.safetensors.index.json").read_text())["weight_map"]
    handles = {}

    def get(key: str) -> torch.Tensor:
        path = model_path / index[key]
        if path not in handles:
            handles[path] = safe_open(str(path), framework="pt", device="cpu")
        return handles[path].get_tensor(key)

    return get, handles


def backup_fp16_layer(model, layer_id: int, extension) -> dict:
    weight = model.transformer_layers[layer_id].weight
    tensors = [(name, getattr(weight, name)) for name in FP16_ATTRIBUTES]
    tensors.sort(key=lambda item: item[1].data_ptr())
    base = tensors[0][1].data_ptr()
    end = max(tensor.data_ptr() + tensor.numel() * tensor.element_size() for _, tensor in tensors)
    size = end - base
    buffer = torch.empty(size, dtype=torch.uint8, pin_memory=True)
    tensor_map = []
    for name, tensor in tensors:
        offset = tensor.data_ptr() - base
        byte_view = tensor.detach().view(torch.uint8).reshape(-1)
        buffer[offset:offset + byte_view.numel()].copy_(byte_view)
        tensor_map.append(_tensor_info(name, tensor, offset))
    extension.register_layer_memory_org_gpu(layer_id, base, size)
    extension.register_layer_memory_org_cpu(layer_id, buffer.data_ptr(), size)
    extension.register_layer_memory_tensor_map_org(layer_id, tensor_map)
    return {"buffer": buffer, "base": base, "size": size, "tensor_map": tensor_map}


def load_packed_layer(model_path: str | Path, layer_id: int, extension) -> dict:
    model_path = Path(model_path)
    get, handles = _checkpoint_getter(model_path)
    prefix = f"model.layers.{layer_id}"
    items = [
        ("input_layernorm.weight", get(f"{prefix}.input_layernorm.weight")),
        ("post_attention_layernorm.weight", get(f"{prefix}.post_attention_layernorm.weight")),
    ]
    for module_name in QUANT_MODULES:
        for suffix in ("qweight", "qzeros", "scales"):
            key = f"{prefix}.{module_name}.{suffix}"
            items.append((f"{module_name}.{suffix}", get(key)))
    total = sum(tensor.numel() * tensor.element_size() for _, tensor in items)
    buffer = torch.empty(total, dtype=torch.uint8, pin_memory=True)
    tensor_map = []
    offset = 0
    for name, tensor in items:
        tensor = tensor.contiguous()
        byte_view = tensor.view(torch.uint8).reshape(-1)
        buffer[offset:offset + byte_view.numel()].copy_(byte_view)
        tensor_map.append(_tensor_info(name, tensor, offset))
        offset += byte_view.numel()
    for handle in handles.values():
        handle.__exit__(None, None, None)
    extension.register_layer_memory_quant(layer_id, buffer.data_ptr(), total)
    extension.register_layer_memory_tensor_map_quant(layer_id, tensor_map)
    return {"buffer": buffer, "size": total, "tensor_map": tensor_map}


def materialize_packed_tensors(packed: dict, device: str = "cuda") -> list[torch.Tensor]:
    tensors = []
    for entry in packed["tensor_map"]:
        _, info = next(iter(entry.items()))
        byte_view = packed["buffer"][info["offset"]:info["offset"] + info["size"]]
        tensor = byte_view.view(info["dtype"]).view(info["shape"]).to(device)
        tensors.append(tensor)
    return tensors


def release_fp16_layer(model, layer_id: int) -> None:
    weight = model.transformer_layers[layer_id].weight
    for name in FP16_ATTRIBUTES:
        if hasattr(weight, name):
            delattr(weight, name)
    model.transformer_layers[layer_id] = None


def _dimensions(config, module_name: str) -> tuple[int, int]:
    hidden = config.hidden_size
    kv = config.num_kv_heads * config.head_dim
    ffn = config.ffn_inter_dim
    return {
        "self_attn.q_proj": (hidden, hidden),
        "self_attn.k_proj": (hidden, kv),
        "self_attn.v_proj": (hidden, kv),
        "self_attn.o_proj": (hidden, hidden),
        "mlp.gate_proj": (hidden, ffn),
        "mlp.up_proj": (hidden, ffn),
        "mlp.down_proj": (ffn, hidden),
    }[module_name]


def install_autoawq_layer(model, layer_id: int, tensors: list[torch.Tensor], tensor_map: list[dict]):
    from awq.modules.linear.gemm import WQLinear_GEMM
    from swiftllm.worker.layers.transformer_layer import AWQTransformerLayer

    names = [next(iter(entry)) for entry in tensor_map]
    values = dict(zip(names, tensors))
    self_attn = SimpleNamespace()
    mlp = SimpleNamespace()
    modules = []
    for module_name in QUANT_MODULES:
        in_features, out_features = _dimensions(model.model_config, module_name)
        module = WQLinear_GEMM(4, 128, in_features, out_features, False, "meta")
        module.qweight = values[f"{module_name}.qweight"]
        module.qzeros = values[f"{module_name}.qzeros"]
        module.scales = values[f"{module_name}.scales"]
        module.eval()
        parent, attribute = module_name.split(".")
        setattr(self_attn if parent == "self_attn" else mlp, attribute, module)
        modules.append(module)
    awq_layer = SimpleNamespace(
        input_layernorm=SimpleNamespace(weight=values["input_layernorm.weight"]),
        post_attention_layernorm=SimpleNamespace(weight=values["post_attention_layernorm.weight"]),
        self_attn=self_attn,
        mlp=mlp,
    )
    wrapper = AWQTransformerLayer(
        awq_layer,
        model.model_config,
        model.engine_config,
        layer_id,
        torch.cuda.Stream(),
    )
    model.transformer_layers[layer_id] = wrapper
    return wrapper, modules


def restore_fp16_layer(model, layer_id: int, tensors: list[torch.Tensor], tensor_map: list[dict]):
    from swiftllm.worker.layers.transformer_layer import LlamaTransformerLayer
    from swiftllm.worker.weight import LlamaTransformerLayerWeight

    weight = LlamaTransformerLayerWeight(layer_id, model.model_config, torch.float16)
    for entry, tensor in zip(tensor_map, tensors):
        setattr(weight, next(iter(entry)), tensor)
    wrapper = LlamaTransformerLayer(
        model.model_config,
        model.engine_config,
        weight,
        torch.cuda.Stream(),
        layer_id,
    )
    model.weight.layers[layer_id] = weight
    model.transformer_layers[layer_id] = wrapper
    return wrapper
