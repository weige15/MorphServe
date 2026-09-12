import dataclasses
import json
import os
from typing import Any

import safetensors
import torch

from swiftllm.model_config import LlamaModelConfig


@dataclasses.dataclass
class QuantizedMatrix:
    """NF4 matrices kept in the layouts required by bitsandbytes.

    ``gemv_*`` is source [out, in] layout for one-token decode. ``matmul_*``
    is transposed [in, out] layout for the bitsandbytes batched prefill
    fallback. Both remain packed W4; neither is an FP16 full-weight cache.
    """

    matmul_data: torch.Tensor
    matmul_quant_state: Any
    gemv_data: torch.Tensor
    gemv_quant_state: Any
    shape: tuple[int, int]


@dataclasses.dataclass
class RegisteredWeightItem:
    attr_name: str
    key: str
    shape: tuple
    dtype: torch.dtype
    quantizable: bool = False

class WeightBase:
    """
    The base class of all weight classes (i.e. LlamaTransformerLayerWeight or LlamaWeight)

    During weight initialization, each concrete weight class should first register
    all weight items. Each weight item has its own attribute name, key, shape, and dtype.

    During weight loading, RegisterWeightItem will be passed to the weight getter
    function, which should return the corresponding weight value (real/dummy).
    """

    def __init__(self):
        self.registered_weights = []

    def register_weight(self, item: RegisteredWeightItem):
        self.registered_weights.append(item)

    def _post_process_after_load(self, getter: callable):
        """
        This function is called after loading weights (real/dummy).
        Defined in each concrete weight class, called by load_weights().
        """
        raise NotImplementedError()
    
    def load_weights(self, getter: callable):
        """Load weights, optionally replacing matrix weights with the W4 proxy."""
        for item in self.registered_weights:
            weight_value = getter(item)
            assert weight_value is not None, f"getter() returned None for {item.key} ({item})"
            assert isinstance(weight_value, torch.Tensor), f"Weight {item.key} is not a tensor"
            assert weight_value.shape == item.shape, f"Shape of weight {item.key} does not match"
            assert weight_value.device.type == "cuda", f"Weight {item.key} is not on GPU"
            if getattr(self, "quantized", False) and item.quantizable:
                setattr(self, item.attr_name, quantize_matrix(weight_value, item.dtype))
            else:
                setattr(self, item.attr_name, weight_value.to(item.dtype))
        self._post_process_after_load(getter)


def quantize_matrix(weight: torch.Tensor, dtype: torch.dtype) -> QuantizedMatrix:
    """Quantize one [out, in] weight with the shared NF4 W4 proxy.

    This deliberately uses bitsandbytes NF4 weight-only quantization, not AWQ.
    The source-layout packed view feeds bitsandbytes' low-bit GEMV for
    one-token decode. The transposed packed view feeds the available
    bitsandbytes batched prefill fallback. Neither view is a full FP16 matrix.
    """
    try:
        import bitsandbytes.functional as bnb_functional
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("bitsandbytes is required for the W4 proxy") from exc

    source = weight.to(dtype=dtype).contiguous()
    gemv_data, gemv_quant_state = bnb_functional.quantize_4bit(
        source,
        absmax=None,
        blocksize=64,
        compress_statistics=False,
        quant_type="nf4",
    )
    matmul_data, matmul_quant_state = bnb_functional.quantize_4bit(
        source.transpose(0, 1).contiguous(),
        absmax=None,
        blocksize=64,
        compress_statistics=False,
        quant_type="nf4",
    )
    return QuantizedMatrix(
        matmul_data=matmul_data,
        matmul_quant_state=matmul_quant_state,
        gemv_data=gemv_data,
        gemv_quant_state=gemv_quant_state,
        shape=tuple(weight.shape),
    )


class LlamaTransformerLayerWeight(WeightBase):
    """
    Class stores the weights of one transformer layer (transformer block) in Llama model.
    """

    def __init__(
        self,
        layer_id: int,
        model_config: LlamaModelConfig,
        dtype: torch.dtype,
        model_version: str = "llama",
        quantized: bool = False,
    ):
        super().__init__()

        self.layer_id = layer_id
        self.model_config = model_config
        self.dtype = dtype
        self.model_version = model_version
        self.quantized = quantized

        self.register_weight(RegisteredWeightItem(
            "attn_norm",
            f"model.layers.{self.layer_id}.input_layernorm.weight",
            (self.model_config.hidden_size,),
            self.dtype
        ))
        self.register_weight(RegisteredWeightItem(
            "q_proj",
            f"model.layers.{self.layer_id}.self_attn.q_proj.weight",
            (self.model_config.hidden_size, self.model_config.hidden_size),
            self.dtype,
            quantizable=True,
        ))
        self.register_weight(RegisteredWeightItem(
            "k_proj",
            f"model.layers.{self.layer_id}.self_attn.k_proj.weight",
            (self.model_config.num_kv_heads*self.model_config.head_dim, self.model_config.hidden_size),
            self.dtype,
            quantizable=True,
        ))
        self.register_weight(RegisteredWeightItem(
            "v_proj",
            f"model.layers.{self.layer_id}.self_attn.v_proj.weight",
            (self.model_config.num_kv_heads*self.model_config.head_dim, self.model_config.hidden_size),
            self.dtype,
            quantizable=True,
        ))
        self.register_weight(RegisteredWeightItem(
            "o_proj",
            f"model.layers.{self.layer_id}.self_attn.o_proj.weight",
            (self.model_config.hidden_size, self.model_config.hidden_size),
            self.dtype,
            quantizable=True,
        ))

        self.register_weight(RegisteredWeightItem(
            "ffn_norm",
            f"model.layers.{self.layer_id}.post_attention_layernorm.weight",
            (self.model_config.hidden_size,),
            self.dtype
        ))
        self.register_weight(RegisteredWeightItem(
            "up_proj",
            f"model.layers.{self.layer_id}.mlp.up_proj.weight",
            (self.model_config.ffn_inter_dim, self.model_config.hidden_size),
            self.dtype,
            quantizable=True,
        ))
        self.register_weight(RegisteredWeightItem(
            "gate_proj",
            f"model.layers.{self.layer_id}.mlp.gate_proj.weight",
            (self.model_config.ffn_inter_dim, self.model_config.hidden_size),
            self.dtype,
            quantizable=True,
        ))
        self.register_weight(RegisteredWeightItem(
            "down_proj",
            f"model.layers.{self.layer_id}.mlp.down_proj.weight",
            (self.model_config.hidden_size, self.model_config.ffn_inter_dim),
            self.dtype,
            quantizable=True,
        ))

    def _post_process_after_load(self, getter: callable):
        # pylint: disable=no-member
        # Packed W4 matrices cannot be concatenated in source-weight layout.
        # Keep the two MLP projections separate in a quantized layer.
        if not self.quantized:
            self.up_gate_proj = torch.cat((self.up_proj, self.gate_proj), dim=0).contiguous()
            del self.up_proj, self.gate_proj


class LlamaWeight(WeightBase):
    def __init__(
        self,
        model_config: LlamaModelConfig,
        dtype: torch.dtype,
        model_version: str = "llama",
        quantized_layer_count: int = 0,
        lm_head_key: str | None = None,
    ):
        super().__init__()

        self.model_config = model_config
        self.dtype = dtype
        self.model_version = model_version
        self.lm_head_key = lm_head_key or (
            "model.embed_tokens.weight"
            if model_config.tie_word_embeddings
            else "lm_head.weight"
        )

        self.register_weight(RegisteredWeightItem(
            "wte",
            "model.embed_tokens.weight",
            (self.model_config.vocab_size, self.model_config.hidden_size),
            self.dtype
        ))

        self.register_weight(RegisteredWeightItem(
            "lm_head",
            self.lm_head_key,
            (self.model_config.vocab_size, self.model_config.hidden_size),
            self.dtype
        ))

        self.register_weight(RegisteredWeightItem(
            "final_norm",
            "model.norm.weight",
            (self.model_config.hidden_size,),
            self.dtype
        ))

        self.layers: list[LlamaTransformerLayerWeight] = []
        for i in range(self.model_config.num_layers):
            layer = LlamaTransformerLayerWeight(
                i,
                self.model_config,
                self.dtype,
                self.model_version,
                quantized=i < quantized_layer_count,
            )
            self.layers.append(layer)

    def _post_process_after_load(self, getter: callable):
        for layer in self.layers:
            layer.load_weights(getter)


def _checkpoint_weight_keys(model_path: str) -> set[str] | None:
    """Return checkpoint keys without loading model tensors into GPU memory."""
    safetensor_index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(safetensor_index_path):
        with open(safetensor_index_path, "r", encoding="utf-8") as handle:
            return set(json.load(handle)["weight_map"])

    safetensor_files = [
        name for name in os.listdir(model_path) if name.endswith(".safetensors")
    ]
    if safetensor_files:
        if len(safetensor_files) != 1:
            raise ValueError(
                "model.safetensors.index.json is required when a checkpoint has "
                "multiple safetensors files"
            )
        with safetensors.safe_open(
            os.path.join(model_path, safetensor_files[0]),
            framework="pt",
            device="cpu",
        ) as handle:
            return set(handle.keys())

    pytorch_index_path = os.path.join(model_path, "pytorch_model.bin.index.json")
    if os.path.exists(pytorch_index_path):
        with open(pytorch_index_path, "r", encoding="utf-8") as handle:
            return set(json.load(handle)["weight_map"])

    # A legacy single pytorch_model.bin has no cheap key index. The config's
    # tie_word_embeddings flag is the only safe fallback; the tensor getter
    # below still reports the exact missing key if the file disagrees.
    return None


def resolve_lm_head_key(
    model_path: str,
    model_config: LlamaModelConfig,
    available_keys: set[str] | None = None,
) -> str:
    """Resolve lm_head from actual checkpoint keys, never from rope_scaling."""
    keys = available_keys if available_keys is not None else _checkpoint_weight_keys(model_path)
    if keys is not None:
        if "lm_head.weight" in keys:
            return "lm_head.weight"
        if "model.embed_tokens.weight" in keys and model_config.tie_word_embeddings:
            return "model.embed_tokens.weight"
        interesting = sorted(
            key for key in keys if "embed_tokens" in key or "lm_head" in key
        )
        raise KeyError(
            "Checkpoint has no usable lm_head.weight; available embedding/head keys: "
            f"{interesting}"
        )
    return (
        "model.embed_tokens.weight"
        if model_config.tie_word_embeddings
        else "lm_head.weight"
    )


def load_weights(
    model_config: LlamaModelConfig,
    dtype: torch.dtype,
    model_path: str,
    use_dummy: bool = False,
    model_version: str = "auto",
    quantized_layer_count: int = 0,
) -> LlamaWeight:
    """
    Load weights from a given path
    """
    available_keys = None if use_dummy else _checkpoint_weight_keys(model_path)
    lm_head_key = resolve_lm_head_key(model_path, model_config, available_keys)

    # ``model_version`` is retained for API compatibility with older callers;
    # no weight layout decision is made from rope_scaling.
    if model_version == "auto":
        model_version = "llama"

    if use_dummy:
        def weight_getter_dummy(item: RegisteredWeightItem):
            return torch.empty(item.shape, dtype=item.dtype, device="cuda").uniform_(-0.001, 0.001)
        getter = weight_getter_dummy
    else:
        safetensor_files = [name for name in os.listdir(model_path) if name.endswith(".safetensors")]
        if len(safetensor_files) > 0:
            # Use Safetensors
            safetensor_index_path = os.path.join(model_path, "model.safetensors.index.json")
            if os.path.exists(safetensor_index_path):
                # The weight is stored in multiple files
                f = open(safetensor_index_path, "r", encoding="utf-8")
                safetensor_index = json.load(f)["weight_map"]
                safetensor_filename = None
            else:
                # The weight is stored in a single file
                assert len(safetensor_files) == 1, "model.safetensors.index.json not found, but there are multiple .safetensors files"
                safetensor_index = None
                safetensor_filename = safetensor_files[0]

            def weight_getter_real(item: RegisteredWeightItem):
                if safetensor_index is not None:
                    if item.key not in safetensor_index:
                        raise KeyError(f"Missing checkpoint key: {item.key}")
                    file_name = safetensor_index[item.key]
                else:
                    file_name = safetensor_filename
                file_path = os.path.join(model_path, file_name)
                # For safetensor files, since "opening" it is cheap, we open it every time.
                with safetensors.safe_open(file_path, framework="pt", device="cuda") as f:
                    if item.key not in f.keys():
                        raise KeyError(f"Missing checkpoint key: {item.key}")
                    tensor = f.get_tensor(item.key)
                return tensor.to(item.dtype)
            getter = weight_getter_real

        else:
            # Use PyTorch
            pytorch_index_path = os.path.join(model_path, "pytorch_model.bin.index.json")
            if os.path.exists(pytorch_index_path):
                # The weight is stored in multiple files
                f = open(pytorch_index_path, "r", encoding="utf-8")
                pytorch_index = json.load(f)["weight_map"]
                pytorch_filename = None
            else:
                # The weight is stored in a single file
                pytorch_index = None
                pytorch_filename = "pytorch_model.bin"
            
            # For PyTorch files, since "opening" it is slow (due to deserialization),
            # we open it only once and then store the opened files in a dictionary.
            # We add `mmap=True` to avoid loading the entire file into memory.
            opened_files = {}
            def weight_getter_real(item: RegisteredWeightItem):
                if pytorch_index is not None:
                    if item.key not in pytorch_index:
                        raise KeyError(f"Missing checkpoint key: {item.key}")
                    file_name = pytorch_index[item.key]
                else:
                    file_name = pytorch_filename
                file_path = os.path.join(model_path, file_name)
                if file_path not in opened_files:
                    opened_files[file_path] = torch.load(file_path, map_location="cuda", mmap=True)
                file = opened_files[file_path]
                if item.key not in file:
                    raise KeyError(f"Missing checkpoint key: {item.key}")
                return file[item.key].to(item.dtype)
            getter = weight_getter_real

    weight = LlamaWeight(
        model_config,
        dtype,
        model_version,
        quantized_layer_count=quantized_layer_count,
        lm_head_key=lm_head_key,
    )
    weight.load_weights(getter)
    return weight
