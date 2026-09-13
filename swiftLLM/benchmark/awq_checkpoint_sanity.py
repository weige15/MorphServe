"""Audit the offline AutoAWQ checkpoint and SwiftLLM's Marlin mapping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import safetensors
import torch
from awq.utils.packing_utils import dequantize_gemm

from swiftllm.worker.kernels.linear import linear
from swiftllm.worker.weight import (
    RegisteredWeightItem,
    _make_weight_getter,
    awq_component_items,
    load_awq_marlin_fused_matrix,
    load_awq_marlin_matrix,
)


SHAPES = {
    "self_attn.q_proj": (4096, 4096),
    "self_attn.k_proj": (1024, 4096),
    "self_attn.v_proj": (1024, 4096),
    "self_attn.o_proj": (4096, 4096),
    "mlp.up_proj": (14336, 4096),
    "mlp.gate_proj": (14336, 4096),
    "mlp.down_proj": (4096, 14336),
}


def metadata(model_path: Path, key: str, index: dict[str, str]) -> tuple[tuple[int, ...], str]:
    with safetensors.safe_open(
        model_path / index[key], framework="pt", device="cpu"
    ) as handle:
        value = handle.get_slice(key)
        return tuple(value.get_shape()), value.get_dtype()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--awq-model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = json.loads(
        (args.awq_model_path / "model.safetensors.index.json").read_text(
            encoding="utf-8"
        )
    )["weight_map"]

    component_checks = []
    for layer in range(32):
        for suffix, shape in SHAPES.items():
            item = RegisteredWeightItem(
                suffix.rsplit(".", 1)[-1],
                f"model.layers.{layer}.{suffix}.weight",
                shape,
                torch.float16,
                quantizable=True,
            )
            expected = awq_component_items(item)
            observed = []
            for component in expected:
                component_shape, dtype = metadata(args.awq_model_path, component.key, index)
                observed.append(
                    {
                        "key": component.key,
                        "expected_shape": list(component.shape),
                        "observed_shape": list(component_shape),
                        "expected_dtype": str(component.dtype),
                        "observed_dtype": dtype,
                        "pass": component_shape == component.shape
                        and dtype
                        == {torch.int32: "I32", torch.float16: "F16"}[component.dtype],
                    }
                )
            component_checks.append(
                {"layer": layer, "matrix": suffix, "components": observed}
            )

    getter = _make_weight_getter(str(args.awq_model_path))
    numerical_checks = []
    generator = torch.Generator(device="cuda").manual_seed(2025)
    for suffix, shape in SHAPES.items():
        item = RegisteredWeightItem(
            suffix.rsplit(".", 1)[-1],
            f"model.layers.0.{suffix}.weight",
            shape,
            torch.float16,
            quantizable=True,
        )
        qweight_item, qzeros_item, scales_item = awq_component_items(item)
        qweight = getter(qweight_item)
        qzeros = getter(qzeros_item)
        scales = getter(scales_item)
        packed = load_awq_marlin_matrix(item, getter)
        inputs = torch.randn(
            16,
            shape[1],
            dtype=torch.float16,
            device="cuda",
            generator=generator,
        )
        dense = dequantize_gemm(qweight, qzeros, scales, 4, 128).to(torch.float16)
        reference = torch.matmul(inputs, dense)
        actual = linear(inputs, packed)
        torch.cuda.synchronize()
        difference = actual.float() - reference.float()
        numerical_checks.append(
            {
                "matrix": suffix,
                "source_keys": list(packed.source_keys),
                "finite": bool(torch.isfinite(actual).all().item()),
                "relative_l2_error": float(
                    difference.norm() / reference.float().norm().clamp_min(1e-8)
                ),
                "max_abs_error": float(difference.abs().max()),
                "marlin_component_bytes": sum(
                    value.numel() * value.element_size()
                    for value in (
                        packed.qweight,
                        packed.scales,
                        packed.qzeros,
                        packed.workspace,
                    )
                ),
                "workspace_bytes": packed.workspace.numel()
                * packed.workspace.element_size(),
                "retained_dtypes": {
                    "qweight": str(packed.qweight.dtype),
                    "scales": str(packed.scales.dtype),
                    "qzeros": str(packed.qzeros.dtype),
                },
            }
        )
        del qweight, qzeros, scales, packed, inputs, dense, reference, actual
        torch.cuda.empty_cache()

    fused_items = []
    fused_dense = []
    for suffix in ("mlp.up_proj", "mlp.gate_proj"):
        item = RegisteredWeightItem(
            suffix.rsplit(".", 1)[-1],
            f"model.layers.0.{suffix}.weight",
            SHAPES[suffix],
            torch.float16,
            quantizable=True,
        )
        fused_items.append(item)
        qweight_item, qzeros_item, scales_item = awq_component_items(item)
        fused_dense.append(
            dequantize_gemm(
                getter(qweight_item), getter(qzeros_item), getter(scales_item), 4, 128
            ).to(torch.float16)
        )
    fused_packed = load_awq_marlin_fused_matrix(
        fused_items[0], fused_items[1], getter
    )
    fused_inputs = torch.randn(
        16, 4096, dtype=torch.float16, device="cuda", generator=generator
    )
    fused_reference = torch.matmul(fused_inputs, torch.cat(fused_dense, dim=1))
    fused_actual = linear(fused_inputs, fused_packed)
    torch.cuda.synchronize()
    fused_difference = fused_actual.float() - fused_reference.float()
    numerical_checks.append(
        {
            "matrix": "mlp.up_gate_fused",
            "source_keys": list(fused_packed.source_keys),
            "finite": bool(torch.isfinite(fused_actual).all().item()),
            "relative_l2_error": float(
                fused_difference.norm()
                / fused_reference.float().norm().clamp_min(1e-8)
            ),
            "max_abs_error": float(fused_difference.abs().max()),
            "marlin_component_bytes": sum(
                value.numel() * value.element_size()
                for value in (
                    fused_packed.qweight,
                    fused_packed.scales,
                    fused_packed.qzeros,
                    fused_packed.workspace,
                )
            ),
            "workspace_bytes": fused_packed.workspace.numel()
            * fused_packed.workspace.element_size(),
            "retained_dtypes": {
                "qweight": str(fused_packed.qweight.dtype),
                "scales": str(fused_packed.scales.dtype),
                "qzeros": str(fused_packed.qzeros.dtype),
            },
        }
    )

    all_component_checks_pass = all(
        component["pass"]
        for check in component_checks
        for component in check["components"]
    )
    result = {
        "schema_version": 1,
        "awq_model_path": str(args.awq_model_path),
        "expected_layers": 32,
        "expected_quantized_matrices": 32 * len(SHAPES),
        "expected_component_tensors": 32 * len(SHAPES) * 3,
        "component_checks": component_checks,
        "all_component_checks_pass": all_component_checks_pass,
        "numerical_checks": numerical_checks,
        "all_numerical_checks_finite": all(
            row["finite"] for row in numerical_checks
        ),
        "max_marlin_vs_autoawq_dequant_relative_l2": max(
            row["relative_l2_error"] for row in numerical_checks
        ),
        "full_fp16_matrix_retained_by_packed_object": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "all_component_checks_pass",
                    "all_numerical_checks_finite",
                    "max_marlin_vs_autoawq_dequant_relative_l2",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
