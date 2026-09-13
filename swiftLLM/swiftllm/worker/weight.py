import dataclasses
import json
import os
from typing import Any, Callable

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
class AWQMarlinMatrix:
    """One vLLM-Marlin packed matrix loaded from an offline AutoAWQ checkpoint."""

    qweight: torch.Tensor
    scales: torch.Tensor
    qzeros: torch.Tensor
    workspace: torch.Tensor
    g_idx: torch.Tensor
    g_idx_sort_indices: torch.Tensor
    shape: tuple[int, int]
    source_keys: tuple[str, ...]
    group_size: int = 128
    bits: int = 4
    zero_point: bool = True
    backend: str = "awq_marlin"


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

    def _post_process_after_load(self, getter: Callable, awq_getter: Callable | None):
        """
        This function is called after loading weights (real/dummy).
        Defined in each concrete weight class, called by load_weights().
        """
        raise NotImplementedError()
    
    def load_weights(self, getter: Callable, awq_getter: Callable | None = None):
        """Load dense weights or selected packed matrices from the fixed backend."""
        for item in self.registered_weights:
            backend = getattr(self, "quantization_backend", "nf4_bitsandbytes")
            selected = getattr(self, "quantized", False)
            if (
                selected
                and backend == "awq_marlin"
                and item.attr_name in ("up_proj", "gate_proj")
            ):
                continue
            if selected and item.quantizable and backend == "awq_marlin":
                if awq_getter is None:
                    raise ValueError("AWQ-Marlin selected without an AWQ checkpoint getter")
                value = load_awq_marlin_matrix(item, awq_getter)
            else:
                # AutoAWQ rescales selected layers' norms together with their
                # packed linears. Load those norms from the AWQ artifact too.
                source_getter = (
                    awq_getter
                    if selected and backend == "awq_marlin" and awq_getter is not None
                    else getter
                )
                value = source_getter(item)
                assert value is not None, f"getter() returned None for {item.key} ({item})"
                assert isinstance(value, torch.Tensor), f"Weight {item.key} is not a tensor"
                assert value.shape == item.shape, f"Shape of weight {item.key} does not match"
                assert value.device.type == "cuda", f"Weight {item.key} is not on GPU"
                value = (
                    quantize_matrix(value, item.dtype)
                    if selected and item.quantizable
                    else value.to(item.dtype)
                )
            setattr(self, item.attr_name, value)
        if selected and backend == "awq_marlin":
            up_item = next(item for item in self.registered_weights if item.attr_name == "up_proj")
            gate_item = next(item for item in self.registered_weights if item.attr_name == "gate_proj")
            self.up_gate_proj = load_awq_marlin_fused_matrix(
                up_item, gate_item, awq_getter
            )
        self._post_process_after_load(getter, awq_getter)


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


def awq_component_items(item: RegisteredWeightItem, group_size: int = 128):
    """Return the exact AutoAWQ GEMM component keys and shapes for a matrix."""
    out_features, in_features = item.shape
    if in_features % group_size or out_features % 8:
        raise ValueError(f"AWQ group/packing is incompatible with {item.key}: {item.shape}")
    prefix = item.key.removesuffix(".weight")
    return (
        RegisteredWeightItem(
            "qweight", f"{prefix}.qweight", (in_features, out_features // 8), torch.int32
        ),
        RegisteredWeightItem(
            "qzeros",
            f"{prefix}.qzeros",
            (in_features // group_size, out_features // 8),
            torch.int32,
        ),
        RegisteredWeightItem(
            "scales",
            f"{prefix}.scales",
            (in_features // group_size, out_features),
            torch.float16,
        ),
    )


def _convert_autoawq_to_marlin(
    item: RegisteredWeightItem,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    source_keys: tuple[str, ...],
    group_size: int = 128,
) -> AWQMarlinMatrix:
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        awq_to_marlin_zero_points,
        marlin_make_empty_g_idx,
        marlin_make_workspace_new,
        marlin_permute_scales,
        verify_marlin_supports_shape,
    )

    out_features, in_features = item.shape
    verify_marlin_supports_shape(
        output_size_per_partition=out_features,
        input_size_per_partition=in_features,
        input_size=in_features,
        group_size=group_size,
    )
    expected_shapes = (
        (in_features, out_features // 8),
        (in_features // group_size, out_features // 8),
        (in_features // group_size, out_features),
    )
    for name, value, shape, dtype in zip(
        ("qweight", "qzeros", "scales"),
        (qweight, qzeros, scales),
        expected_shapes,
        (torch.int32, torch.int32, torch.float16),
    ):
        if tuple(value.shape) != shape or value.dtype != dtype:
            raise ValueError(
                f"{item.key} {name} expected {shape}/{dtype}, "
                f"got {tuple(value.shape)}/{value.dtype}"
            )
        if value.device.type != "cuda":
            raise ValueError(f"{item.key} {name} must load directly onto CUDA")

    marlin_qweight = ops.awq_marlin_repack(
        qweight, size_k=in_features, size_n=out_features, num_bits=4
    )
    marlin_scales = marlin_permute_scales(
        scales,
        size_k=in_features,
        size_n=out_features,
        group_size=group_size,
    )
    marlin_qzeros = awq_to_marlin_zero_points(
        qzeros,
        size_k=in_features // group_size,
        size_n=out_features,
        num_bits=4,
    )
    device = marlin_qweight.device
    return AWQMarlinMatrix(
        qweight=marlin_qweight,
        scales=marlin_scales,
        qzeros=marlin_qzeros,
        workspace=marlin_make_workspace_new(device),
        g_idx=marlin_make_empty_g_idx(device),
        g_idx_sort_indices=marlin_make_empty_g_idx(device),
        shape=tuple(item.shape),
        source_keys=source_keys,
        group_size=group_size,
    )


def load_awq_marlin_matrix(
    item: RegisteredWeightItem, getter: Callable, group_size: int = 128
) -> AWQMarlinMatrix:
    """Load AutoAWQ components and retain only their vLLM Marlin transforms."""
    qweight_item, qzeros_item, scales_item = awq_component_items(item, group_size)
    qweight = getter(qweight_item)
    qzeros = getter(qzeros_item)
    scales = getter(scales_item)
    for component_item, value in (
        (qweight_item, qweight),
        (qzeros_item, qzeros),
        (scales_item, scales),
    ):
        if value.shape != component_item.shape or value.dtype != component_item.dtype:
            raise ValueError(
                f"{component_item.key} expected {component_item.shape}/{component_item.dtype}, "
                f"got {tuple(value.shape)}/{value.dtype}"
            )
        if value.device.type != "cuda":
            raise ValueError(f"{component_item.key} must load directly onto CUDA")

    return _convert_autoawq_to_marlin(
        item,
        qweight,
        qzeros,
        scales,
        (qweight_item.key, scales_item.key, qzeros_item.key),
        group_size,
    )


def load_awq_marlin_fused_matrix(
    first: RegisteredWeightItem,
    second: RegisteredWeightItem,
    getter: Callable,
    group_size: int = 128,
) -> AWQMarlinMatrix:
    """Fuse AutoAWQ output columns before repacking, preserving `[first,second]`."""
    if first.shape != second.shape:
        raise ValueError("fused AWQ matrices must have the same shape")
    component_sets = [awq_component_items(item, group_size) for item in (first, second)]
    loaded = [[getter(component) for component in components] for components in component_sets]
    qweight = torch.cat((loaded[0][0], loaded[1][0]), dim=1).contiguous()
    qzeros = torch.cat((loaded[0][1], loaded[1][1]), dim=1).contiguous()
    scales = torch.cat((loaded[0][2], loaded[1][2]), dim=1).contiguous()
    fused = RegisteredWeightItem(
        "up_gate_proj",
        first.key.replace("up_proj.weight", "up_gate_proj.weight"),
        (first.shape[0] + second.shape[0], first.shape[1]),
        first.dtype,
        quantizable=True,
    )
    source_keys = tuple(
        component.key for components in component_sets for component in components
    )
    return _convert_autoawq_to_marlin(
        fused, qweight, qzeros, scales, source_keys, group_size
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
        quantization_backend: str = "nf4_bitsandbytes",
    ):
        super().__init__()

        self.layer_id = layer_id
        self.model_config = model_config
        self.dtype = dtype
        self.model_version = model_version
        self.quantized = quantized
        self.quantization_backend = quantization_backend

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

    def _post_process_after_load(self, getter: Callable, awq_getter: Callable | None):
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
        quantization_backend: str = "nf4_bitsandbytes",
    ):
        super().__init__()

        self.model_config = model_config
        self.dtype = dtype
        self.model_version = model_version
        self.quantized_layer_count = quantized_layer_count
        self.quantization_backend = quantization_backend
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
                quantization_backend=quantization_backend,
            )
            self.layers.append(layer)

    def _post_process_after_load(self, getter: Callable, awq_getter: Callable | None):
        for layer in self.layers:
            layer.load_weights(getter, awq_getter)

    def quantized_representation_bytes(self) -> int:
        total = 0
        for layer in self.layers:
            if not layer.quantized:
                continue
            matrix_names = (
                ("q_proj", "k_proj", "v_proj", "o_proj", "up_gate_proj", "down_proj")
                if layer.quantization_backend == "awq_marlin"
                else ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj")
            )
            total += sum(
                quantized_matrix_bytes(getattr(layer, name)) for name in matrix_names
            )
        return total


def quantized_matrix_bytes(matrix: QuantizedMatrix | AWQMarlinMatrix) -> int:
    if isinstance(matrix, AWQMarlinMatrix):
        tensors = (
            matrix.qweight,
            matrix.scales,
            matrix.qzeros,
            matrix.workspace,
            matrix.g_idx,
            matrix.g_idx_sort_indices,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    tensors = [matrix.matmul_data, matrix.gemv_data]
    for state in (matrix.matmul_quant_state, matrix.gemv_quant_state):
        tensors.extend(value for value in vars(state).values() if isinstance(value, torch.Tensor))
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


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


def _make_weight_getter(model_path: str):
    safetensor_files = [
        name for name in os.listdir(model_path) if name.endswith(".safetensors")
    ]
    if safetensor_files:
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as handle:
                index = json.load(handle)["weight_map"]
            single_filename = None
        else:
            if len(safetensor_files) != 1:
                raise ValueError(
                    "model.safetensors.index.json is required for sharded checkpoints"
                )
            index = None
            single_filename = safetensor_files[0]

        def safetensor_getter(item: RegisteredWeightItem):
            if index is not None:
                if item.key not in index:
                    raise KeyError(f"Missing checkpoint key: {item.key}")
                filename = index[item.key]
            else:
                filename = single_filename
            with safetensors.safe_open(
                os.path.join(model_path, filename), framework="pt", device="cuda"
            ) as handle:
                if item.key not in handle.keys():
                    raise KeyError(f"Missing checkpoint key: {item.key}")
                tensor = handle.get_tensor(item.key)
            return tensor.to(item.dtype)

        return safetensor_getter

    index_path = os.path.join(model_path, "pytorch_model.bin.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as handle:
            index = json.load(handle)["weight_map"]
        single_filename = None
    else:
        index = None
        single_filename = "pytorch_model.bin"
    opened_files = {}

    def pytorch_getter(item: RegisteredWeightItem):
        if index is not None:
            if item.key not in index:
                raise KeyError(f"Missing checkpoint key: {item.key}")
            filename = index[item.key]
        else:
            filename = single_filename
        path = os.path.join(model_path, filename)
        if path not in opened_files:
            opened_files[path] = torch.load(path, map_location="cuda", mmap=True)
        if item.key not in opened_files[path]:
            raise KeyError(f"Missing checkpoint key: {item.key}")
        return opened_files[path][item.key].to(item.dtype)

    return pytorch_getter


def _validate_awq_checkpoint(model_path: str) -> None:
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r", encoding="utf-8") as handle:
        quantization = json.load(handle).get("quantization_config", {})
    observed = (
        quantization.get("quant_method"),
        quantization.get("bits"),
        quantization.get("group_size"),
        quantization.get("zero_point"),
        str(quantization.get("version", "")).lower(),
    )
    expected = ("awq", 4, 128, True, "gemm")
    if observed != expected:
        raise ValueError(f"AWQ checkpoint config must be {expected}, got {observed}")


def load_weights(
    model_config: LlamaModelConfig,
    dtype: torch.dtype,
    model_path: str,
    use_dummy: bool = False,
    model_version: str = "auto",
    quantized_layer_count: int = 0,
    quantization_backend: str = "nf4_bitsandbytes",
    quantized_model_path: str | None = None,
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

    if quantization_backend not in ("nf4_bitsandbytes", "awq_marlin"):
        raise ValueError(f"unsupported quantization backend: {quantization_backend}")
    if use_dummy and quantized_layer_count and quantization_backend == "awq_marlin":
        raise ValueError("dummy AWQ weights are unsupported; use the offline checkpoint")

    if use_dummy:
        def getter(item: RegisteredWeightItem):
            return torch.empty(item.shape, dtype=item.dtype, device="cuda").uniform_(
                -0.001, 0.001
            )
    else:
        getter = _make_weight_getter(model_path)

    awq_getter = None
    if quantized_layer_count and quantization_backend == "awq_marlin":
        if quantized_model_path is None:
            raise ValueError("AWQ-Marlin requires quantized_model_path")
        _validate_awq_checkpoint(quantized_model_path)
        awq_getter = _make_weight_getter(quantized_model_path)

    weight = LlamaWeight(
        model_config,
        dtype,
        model_version,
        quantized_layer_count=quantized_layer_count,
        lm_head_key=lm_head_key,
        quantization_backend=quantization_backend,
    )
    weight.load_weights(getter, awq_getter)
    return weight
