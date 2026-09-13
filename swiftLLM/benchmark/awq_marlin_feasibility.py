"""Cheap, shape-complete AWQ-Marlin feasibility benchmark for Llama-3.1-8B.

This does not quantize the full checkpoint.  It packs representative real
checkpoint matrices into the public AutoAWQ GEMM layout using the exact
asymmetric groupwise formula and column order, converts that layout with
vLLM's installed AWQ-Marlin utilities, and compares the packed hot path with
FP16 and the existing bitsandbytes NF4 path.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
from typing import Callable

import bitsandbytes as bnb
import bitsandbytes.functional as bnb_functional
import safetensors
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    apply_awq_marlin_linear,
    awq_to_marlin_zero_points,
    check_marlin_supported,
    marlin_make_empty_g_idx,
    marlin_make_workspace_new,
    marlin_permute_scales,
    verify_marlin_supports_shape,
)
from vllm.scalar_type import scalar_types


GROUP_SIZE = 128
BITS = 4
PACK_FACTOR = 8
AUTOAWQ_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)
REPRESENTATIVES = {
    "q_or_o_projection": (
        "model.layers.0.self_attn.q_proj.weight",
        (4096, 4096),
    ),
    "k_or_v_projection": (
        "model.layers.0.self_attn.k_proj.weight",
        (1024, 4096),
    ),
    "up_or_gate_projection": (
        "model.layers.0.mlp.up_proj.weight",
        (14336, 4096),
    ),
    "down_projection": (
        "model.layers.0.mlp.down_proj.weight",
        (4096, 14336),
    ),
}


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    rank = (len(ordered) - 1) * p / 100
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_tensor(model_path: Path, key: str) -> torch.Tensor:
    index_path = model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    filename = index[key]
    with safetensors.safe_open(
        model_path / filename, framework="pt", device="cuda"
    ) as handle:
        value = handle.get_tensor(key)
    return value.to(dtype=torch.float16).contiguous()


def pack_autoawq_columns(values: torch.Tensor) -> torch.Tensor:
    """Pack unsigned int4 columns in AutoAWQ GEMM's interleaved order."""
    if values.dtype != torch.int32 or values.ndim != 2:
        raise TypeError("values must be a 2-D int32 tensor")
    if values.shape[1] % PACK_FACTOR:
        raise ValueError("column count must be divisible by eight")
    order = torch.tensor(AUTOAWQ_ORDER, dtype=torch.long, device=values.device)
    arranged = values.reshape(values.shape[0], -1, PACK_FACTOR)[:, :, order]
    shifts = torch.arange(PACK_FACTOR, dtype=torch.int32, device=values.device) * BITS
    return torch.sum(arranged << shifts, dim=-1).to(torch.int32).contiguous()


def autoawq_rtn_layout(
    weight: torch.Tensor, group_size: int = GROUP_SIZE
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create the AutoAWQ GEMM layout after AWQ scaling/clipping.

    The feasibility benchmark has no activation calibration stage. It applies
    AutoAWQ's final asymmetric groupwise quantize/pack formula directly to a
    real matrix. Full-checkpoint Phase B uses the public AutoAWQ quantizer.
    """
    out_features, in_features = weight.shape
    if in_features % group_size:
        raise ValueError(f"input size {in_features} is not divisible by {group_size}")
    grouped = weight.float().reshape(out_features, in_features // group_size, group_size)
    minimum = grouped.amin(dim=-1)
    maximum = grouped.amax(dim=-1)
    scales_out_group = ((maximum - minimum).clamp_min(1e-5) / 15).to(torch.float16)
    zeros_out_group = (
        -torch.round(minimum / scales_out_group.float())
    ).clamp_(0, 15).to(torch.int32)
    quantized_out_in = torch.round(
        grouped / scales_out_group.float().unsqueeze(-1)
        + zeros_out_group.float().unsqueeze(-1)
    ).clamp_(0, 15).to(torch.int32).reshape(out_features, in_features)

    # AutoAWQ GEMM stores input-major qweight and group-major metadata.
    qweight = pack_autoawq_columns(quantized_out_in.t().contiguous())
    scales = scales_out_group.t().contiguous()
    qzeros = pack_autoawq_columns(zeros_out_group.t().contiguous())
    return qweight, qzeros, scales


class MarlinMatrix:
    def __init__(
        self,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        in_features: int,
        out_features: int,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.qweight = ops.awq_marlin_repack(
            qweight,
            size_k=in_features,
            size_n=out_features,
            num_bits=BITS,
        )
        self.scales = marlin_permute_scales(
            scales,
            size_k=in_features,
            size_n=out_features,
            group_size=GROUP_SIZE,
        )
        self.qzeros = awq_to_marlin_zero_points(
            qzeros,
            size_k=in_features // GROUP_SIZE,
            size_n=out_features,
            num_bits=BITS,
        )
        self.workspace = marlin_make_workspace_new(torch.device("cuda"))
        self.g_idx = marlin_make_empty_g_idx(torch.device("cuda"))
        self.g_idx_sort_indices = marlin_make_empty_g_idx(torch.device("cuda"))

    @property
    def component_bytes(self) -> dict[str, int]:
        return {
            "qweight": tensor_bytes(self.qweight),
            "scales": tensor_bytes(self.scales),
            "qzeros": tensor_bytes(self.qzeros),
            "workspace": tensor_bytes(self.workspace),
            "g_idx": tensor_bytes(self.g_idx),
            "g_idx_sort_indices": tensor_bytes(self.g_idx_sort_indices),
        }

    def apply(self, inputs: torch.Tensor) -> torch.Tensor:
        return apply_awq_marlin_linear(
            input=inputs,
            weight=self.qweight,
            weight_scale=self.scales,
            weight_zp=self.qzeros,
            g_idx=self.g_idx,
            g_idx_sort_indices=self.g_idx_sort_indices,
            workspace=self.workspace,
            quant_type=scalar_types.uint4,
            output_size_per_partition=self.out_features,
            input_size_per_partition=self.in_features,
        )


class NF4Matrix:
    def __init__(self, weight: torch.Tensor) -> None:
        self.gemv_data, self.gemv_state = bnb_functional.quantize_4bit(
            weight,
            blocksize=64,
            compress_statistics=False,
            quant_type="nf4",
        )
        self.matmul_data, self.matmul_state = bnb_functional.quantize_4bit(
            weight.t().contiguous(),
            blocksize=64,
            compress_statistics=False,
            quant_type="nf4",
        )

    @property
    def component_bytes(self) -> dict[str, int]:
        def state_bytes(state) -> int:
            total = 0
            for field in vars(state).values():
                if isinstance(field, torch.Tensor):
                    total += tensor_bytes(field)
            return total

        return {
            "gemv_packed": tensor_bytes(self.gemv_data),
            "gemv_state_tensors": state_bytes(self.gemv_state),
            "matmul_packed": tensor_bytes(self.matmul_data),
            "matmul_state_tensors": state_bytes(self.matmul_state),
        }

    def apply(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 2 and inputs.shape[0] == 1:
            return bnb_functional.gemv_4bit(
                inputs, self.gemv_data, state=self.gemv_state
            )
        return bnb.matmul_4bit(
            inputs,
            self.matmul_data,
            bias=None,
            quant_state=self.matmul_state,
        )


def latency_samples(
    function: Callable[[], torch.Tensor], warmup: int, repeats: int
) -> list[float]:
    for _ in range(warmup):
        output = function()
    del output
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = function()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
        del output
    return samples


def peak_call_bytes(function: Callable[[], torch.Tensor]) -> dict[str, int]:
    gc.collect()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = function()
    torch.cuda.synchronize()
    peak_delta = torch.cuda.max_memory_allocated() - before
    output_bytes = tensor_bytes(output)
    del output
    torch.cuda.synchronize()
    return {
        "peak_delta_bytes": int(peak_delta),
        "output_bytes": int(output_bytes),
        "unexplained_temporary_bytes": int(max(0, peak_delta - output_bytes)),
    }


def numerical(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float | bool]:
    reference_float = reference.float()
    actual_float = actual.float()
    reference_norm = reference_float.norm().clamp_min(1e-8)
    return {
        "finite": bool(torch.isfinite(actual_float).all().item()),
        "relative_l2_error": float((actual_float - reference_float).norm() / reference_norm),
        "max_abs_error": float((actual_float - reference_float).abs().max()),
        "cosine_similarity": float(
            torch.nn.functional.cosine_similarity(
                actual_float.flatten(), reference_float.flatten(), dim=0
            )
        ),
    }


def benchmark_backend(
    function: Callable[[], torch.Tensor],
    reference: torch.Tensor,
    warmup: int,
    repeats: int,
) -> dict:
    actual = function()
    torch.cuda.synchronize()
    numeric = numerical(reference, actual)
    del actual
    peak = peak_call_bytes(function)
    samples = latency_samples(function, warmup, repeats)
    return {
        **numeric,
        **peak,
        "latency_ms": samples,
        "latency_median_ms": statistics.median(samples),
        "latency_p95_ms": percentile(samples, 95),
    }


def check_matrix(
    name: str,
    weight: torch.Tensor,
    source_keys: list[str],
    row_counts: list[int],
    warmup: int,
    repeats: int,
    seed: int,
) -> dict:
    out_features, in_features = weight.shape
    verify_marlin_supports_shape(
        output_size_per_partition=out_features,
        input_size_per_partition=in_features,
        input_size=in_features,
        group_size=GROUP_SIZE,
    )
    qweight, qzeros, scales = autoawq_rtn_layout(weight)
    autoawq_layout_bytes = {
        "qweight": tensor_bytes(qweight),
        "scales": tensor_bytes(scales),
        "qzeros": tensor_bytes(qzeros),
    }
    marlin = MarlinMatrix(qweight, qzeros, scales, in_features, out_features)
    del qweight, qzeros, scales
    nf4 = NF4Matrix(weight)
    torch.cuda.synchronize()

    fp16_weight_bytes = tensor_bytes(weight)
    marlin_components = marlin.component_bytes
    nf4_components = nf4.component_bytes
    checks: dict[str, dict] = {}
    generator = torch.Generator(device="cuda").manual_seed(seed)
    for rows in row_counts:
        inputs = torch.randn(
            rows,
            in_features,
            dtype=torch.float16,
            device="cuda",
            generator=generator,
        )
        reference = torch.nn.functional.linear(inputs, weight)
        torch.cuda.synchronize()
        backends = {
            "fp16": benchmark_backend(
                lambda: torch.nn.functional.linear(inputs, weight),
                reference,
                warmup,
                repeats,
            ),
            "awq_marlin": benchmark_backend(
                lambda: marlin.apply(inputs), reference, warmup, repeats
            ),
            "nf4_bitsandbytes": benchmark_backend(
                lambda: nf4.apply(inputs), reference, warmup, repeats
            ),
        }
        checks[str(rows)] = {
            "activation_rows": rows,
            "input_bytes": tensor_bytes(inputs),
            "output_bytes": tensor_bytes(reference),
            "backends": backends,
            "awq_over_fp16_median_latency": (
                backends["awq_marlin"]["latency_median_ms"]
                / backends["fp16"]["latency_median_ms"]
            ),
            "nf4_over_fp16_median_latency": (
                backends["nf4_bitsandbytes"]["latency_median_ms"]
                / backends["fp16"]["latency_median_ms"]
            ),
        }
        del inputs, reference
        gc.collect()
        torch.cuda.empty_cache()

    result = {
        "name": name,
        "source_keys": source_keys,
        "source_shape_out_in": [out_features, in_features],
        "marlin_shape_mkn": ["M", in_features, out_features],
        "shape_supported": True,
        "fp16_weight_bytes": fp16_weight_bytes,
        "autoawq_layout_bytes": autoawq_layout_bytes,
        "marlin_component_bytes": marlin_components,
        "marlin_total_bytes": sum(marlin_components.values()),
        "marlin_fraction_of_fp16": sum(marlin_components.values()) / fp16_weight_bytes,
        "nf4_component_bytes": nf4_components,
        "nf4_total_bytes": sum(nf4_components.values()),
        "nf4_fraction_of_fp16": sum(nf4_components.values()) / fp16_weight_bytes,
        "checks": checks,
    }
    del marlin, nf4
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 16, 128, 1024])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("AWQ-Marlin feasibility requires CUDA")

    import bitsandbytes
    import transformers
    import vllm

    capability = torch.cuda.get_device_capability()
    marlin_supported = check_marlin_supported(
        scalar_types.uint4,
        GROUP_SIZE,
        has_zp=True,
        device_capability=capability[0] * 10 + capability[1],
    )
    if not marlin_supported:
        raise RuntimeError(
            f"AWQ-Marlin uint4/group-{GROUP_SIZE}/zero-point is unsupported on {capability}"
        )

    apply_source = inspect.getsource(apply_awq_marlin_linear)
    result = {
        "schema_version": 1,
        "protocol_manifest": "benchmark-results/packed-int4-backend-v6/protocol_manifest.json",
        "model_path": str(args.model_path),
        "model_config_sha256": sha256(args.model_path / "config.json"),
        "device": {
            "name": torch.cuda.get_device_name(),
            "compute_capability": list(capability),
            "multiprocessor_count": torch.cuda.get_device_properties(0).multi_processor_count,
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "software": {
            "python": os.sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "vllm": vllm.__version__,
            "transformers": transformers.__version__,
            "bitsandbytes": bitsandbytes.__version__,
        },
        "configuration": {
            "weight_bits": BITS,
            "group_size": GROUP_SIZE,
            "zero_point": True,
            "scales_dtype": "float16",
            "activation_dtype": "float16",
            "microbenchmark_quantizer": "AutoAWQ-formula asymmetric groupwise RTN on real checkpoint weights; format/kernel feasibility only",
            "runtime": "vLLM 0.11.2 AWQ-Marlin repack plus gptq_marlin_gemm",
            "rows": args.rows,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "compiled_ops": {
            "awq_marlin_repack_present": hasattr(torch.ops._C, "awq_marlin_repack"),
            "gptq_marlin_gemm_present": hasattr(torch.ops._C, "gptq_marlin_gemm"),
            "supported": marlin_supported,
        },
        "hot_path_source": {
            "call": "apply_awq_marlin_linear -> vllm._custom_ops.gptq_marlin_gemm",
            "contains_awq_dequantize": "awq_dequantize" in apply_source,
            "contains_torch_matmul": "torch.matmul" in apply_source,
            "full_fp16_weight_materialization_expected": False,
        },
        "matrices": [],
    }

    for index, (name, (key, shape)) in enumerate(REPRESENTATIVES.items()):
        weight = checkpoint_tensor(args.model_path, key)
        if tuple(weight.shape) != shape:
            raise ValueError(f"{key} expected {shape}, got {tuple(weight.shape)}")
        result["matrices"].append(
            check_matrix(
                name,
                weight,
                [key],
                args.rows,
                args.warmup,
                args.repeats,
                args.seed + index,
            )
        )
        del weight
        gc.collect()
        torch.cuda.empty_cache()

    # The FP16 SwiftLLM MLP concatenates up then gate. Verify that this exact
    # combined N is also accepted and numerically sane as a future launch-fusion
    # option. Use fewer rows only if explicitly requested by --rows.
    up_key = "model.layers.0.mlp.up_proj.weight"
    gate_key = "model.layers.0.mlp.gate_proj.weight"
    up = checkpoint_tensor(args.model_path, up_key)
    gate = checkpoint_tensor(args.model_path, gate_key)
    fused = torch.cat((up, gate), dim=0).contiguous()
    del up, gate
    result["matrices"].append(
        check_matrix(
            "fused_up_gate_feasibility",
            fused,
            [up_key, gate_key],
            args.rows,
            args.warmup,
            args.repeats,
            args.seed + len(REPRESENTATIVES),
        )
    )
    del fused
    gc.collect()
    torch.cuda.empty_cache()

    max_relative_l2 = max(
        row["backends"]["awq_marlin"]["relative_l2_error"]
        for matrix in result["matrices"]
        for row in matrix["checks"].values()
    )
    min_cosine = min(
        row["backends"]["awq_marlin"]["cosine_similarity"]
        for matrix in result["matrices"]
        for row in matrix["checks"].values()
    )
    max_temp = max(
        row["backends"]["awq_marlin"]["unexplained_temporary_bytes"]
        for matrix in result["matrices"]
        for row in matrix["checks"].values()
    )
    max_fraction = max(matrix["marlin_fraction_of_fp16"] for matrix in result["matrices"])
    gates = {
        "all_shapes_supported": all(matrix["shape_supported"] for matrix in result["matrices"]),
        "all_outputs_finite": all(
            row["backends"]["awq_marlin"]["finite"]
            for matrix in result["matrices"]
            for row in matrix["checks"].values()
        ),
        "max_relative_l2_error": max_relative_l2,
        "relative_l2_below_0p20": max_relative_l2 < 0.20,
        "min_cosine_similarity": min_cosine,
        "cosine_above_0p98": min_cosine > 0.98,
        "max_marlin_fraction_of_fp16": max_fraction,
        "packed_fraction_below_0p35": max_fraction < 0.35,
        "max_unexplained_temporary_bytes": max_temp,
        "temporary_below_64_mib": max_temp < 64 * 1024 * 1024,
        "no_static_dequantize_or_matmul_hot_path": not (
            result["hot_path_source"]["contains_awq_dequantize"]
            or result["hot_path_source"]["contains_torch_matmul"]
        ),
    }
    gates["phase_a_representation_and_numerical_gate_pass"] = all(
        (
            gates["all_shapes_supported"],
            gates["all_outputs_finite"],
            gates["relative_l2_below_0p20"],
            gates["cosine_above_0p98"],
            gates["packed_fraction_below_0p35"],
            gates["temporary_below_64_mib"],
            gates["no_static_dequantize_or_matmul_hot_path"],
        )
    )
    result["gates"] = gates

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(gates, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
