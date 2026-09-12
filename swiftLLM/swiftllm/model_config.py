import os
import json
import torch

class LlamaModelConfig:
    """
    The configuration of a LLaMA model (including LLaMA 1/2/3).
    """
    
    def __init__(
        self,
        model_config: dict
    ):
        """
        Initialize a LLaMA model configuration from a dict, which should be generated
        from a huggingface transformers config.json file.
        """
        
        assert model_config["model_type"] == "llama"
        self.num_layers = model_config["num_hidden_layers"]
        self.num_q_heads = model_config["num_attention_heads"]
        self.num_kv_heads = model_config.get("num_key_value_heads", self.num_q_heads)
        self.hidden_size = model_config["hidden_size"]
        self.head_dim = self.hidden_size // self.num_q_heads
        self.vocab_size = model_config["vocab_size"]
        self.max_position_embeddings = model_config["max_position_embeddings"]
        self.ffn_inter_dim = model_config["intermediate_size"]
        self.rotary_base = model_config.get("rope_theta", model_config.get("rotary_base", 10000))
        self.rms_norm_eps = model_config["rms_norm_eps"]
        self.rope_scaling = model_config.get("rope_scaling")
        self.rope_theta = model_config.get("rope_theta", 10000)
        self.tie_word_embeddings = bool(model_config.get("tie_word_embeddings", False))
        self.architectures = tuple(model_config.get("architectures", ()))
        if self.rope_scaling is None:
            self.rope_scaling = 1.0
        assert model_config["hidden_act"] == "silu"

    def get_rope_inv_freq(self, device: torch.device | str = "cpu") -> torch.Tensor:
        """Return the inverse frequencies used by Hugging Face Llama RoPE.

        Llama 3.1's ``rope_scaling`` dictionary is not a choice of model
        family and must not be used to infer the checkpoint's lm_head layout.
        Its ``llama3`` values describe a frequency-dependent interpolation;
        treating them as two position multipliers produces a different model.
        Keep this implementation dependency-free so the serving path and the
        parity tests use the same formula as Transformers.
        """
        dim = self.head_dim
        inv_freq = 1.0 / (
            self.rope_theta ** (
                torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim
            )
        )
        scaling = self.rope_scaling
        if not isinstance(scaling, dict):
            return inv_freq

        rope_type = scaling.get("rope_type", scaling.get("type", "default"))
        if rope_type in ("llama3", "llama3.1"):
            factor = float(scaling["factor"])
            low_freq_factor = float(scaling["low_freq_factor"])
            high_freq_factor = float(scaling["high_freq_factor"])
            old_context_len = float(
                scaling.get("original_max_position_embeddings", self.max_position_embeddings)
            )
            low_freq_wavelen = old_context_len / low_freq_factor
            high_freq_wavelen = old_context_len / high_freq_factor
            wavelen = 2 * torch.pi / inv_freq
            # This is Transformers' _compute_llama3_parameters formula.
            scaled = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
            smooth_factor = (
                old_context_len / wavelen - low_freq_factor
            ) / (high_freq_factor - low_freq_factor)
            smoothed = (1 - smooth_factor) * scaled / factor + smooth_factor * scaled
            medium = (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen)
            return torch.where(medium, smoothed, scaled)
        if rope_type == "linear":
            return inv_freq / float(scaling["factor"])
        return inv_freq

    def get_kvslot_size(self, dtype: torch.dtype = torch.float16) -> int:
        """
        Get the size of one kv slot (the kv cache of one token) (in bytes)
        """
        return (2 * self.num_layers * self.num_kv_heads * self.head_dim) * dtype.itemsize
    
    @staticmethod
    def load_from_model_path(model_path: str) -> "LlamaModelConfig":
        with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
            model_config_dict = json.loads(f.read())
        return LlamaModelConfig(model_config_dict)
