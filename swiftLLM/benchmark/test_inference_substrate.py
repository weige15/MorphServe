"""CPU-safe regression tests for checkpoint/model substrate decisions."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import tempfile
import unittest

import torch
from transformers import LlamaConfig
from transformers.modeling_rope_utils import _compute_llama3_parameters

from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.weight import resolve_lm_head_key


LLAMA31_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_act": "silu",
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "max_position_embeddings": 131072,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-5,
    "rope_scaling": {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
    "rope_theta": 500000.0,
    "tie_word_embeddings": False,
    "vocab_size": 128256,
}


class ModelConfigRegressionTests(unittest.TestCase):
    def test_llama31_rope_formula_is_not_model_version_detection(self):
        config = LlamaModelConfig(LLAMA31_CONFIG)
        frequencies = config.get_rope_inv_freq()
        hf_config = LlamaConfig(**LLAMA31_CONFIG)
        expected, _ = _compute_llama3_parameters(hf_config)
        torch.testing.assert_close(frequencies, expected)
        self.assertEqual(tuple(frequencies.shape), (64,))
        self.assertAlmostEqual(float(frequencies[0]), 1.0, places=6)
        # Llama 3.1's low-frequency tail is divided by factor 8, not by a
        # made-up low/high position split.
        self.assertAlmostEqual(float(frequencies[-1]), 1.0 / 8.0 / 500000.0 ** (126 / 128), places=7)

    def test_explicit_lm_head_wins_when_rope_scaling_is_a_dict(self):
        config = LlamaModelConfig(LLAMA31_CONFIG)
        key = resolve_lm_head_key(
            ".",
            config,
            {"model.embed_tokens.weight", "lm_head.weight"},
        )
        self.assertEqual(key, "lm_head.weight")

    def test_tied_checkpoint_uses_embeddings_only_when_head_is_absent(self):
        tied = dict(LLAMA31_CONFIG, tie_word_embeddings=True)
        config = LlamaModelConfig(tied)
        self.assertEqual(
            resolve_lm_head_key(".", config, {"model.embed_tokens.weight"}),
            "model.embed_tokens.weight",
        )

    def test_real_checkpoint_index_resolves_untied_head(self):
        config = LlamaModelConfig(LLAMA31_CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps(LLAMA31_CONFIG))
            (path / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.embed_tokens.weight": "model-00001.safetensors",
                            "lm_head.weight": "model-00004.safetensors",
                        }
                    }
                )
            )
            self.assertEqual(
                resolve_lm_head_key(str(path), config),
                "lm_head.weight",
            )


class W4HotPathRegressionTests(unittest.TestCase):
    def test_linear_source_has_no_full_matrix_decode_fallback(self):
        source = Path(__file__).resolve().parents[1] / "swiftllm/worker/kernels/linear.py"
        tree = ast.parse(source.read_text())
        names = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        }
        self.assertNotIn("dequantize_4bit", names)
        self.assertIn("gemv_4bit", names)
        self.assertIn("matmul_4bit", names)


if __name__ == "__main__":
    unittest.main()
