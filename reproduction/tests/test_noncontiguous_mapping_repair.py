import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
    import swiftllm_c
except (ImportError, OSError):
    torch = None
    swiftllm_c = None

from reproduction.tests.test_candidate_kv_mapping import load_candidate_kernel


@unittest.skipUnless(
    torch is not None and swiftllm_c is not None and torch.cuda.is_available() and "MORPHSERVE_CANDIDATE_PYTHON" in os.environ,
    "reconstructed kernel path required",
)
class ExplicitRegionRepairTests(unittest.TestCase):
    def test_profile_order_store_and_attention_use_registered_group(self):
        store_module = load_candidate_kernel("kvcache_mgmt.py")
        attention_module = load_candidate_kernel("paged_attn.py")
        self.assertTrue(hasattr(store_module, "store_kvcache_explicit_regions"))
        owner = torch.zeros(4 * 4096, dtype=torch.uint8, device="cuda")
        order = [25, 24, 26]
        offsets = {23: 0, 24: 4096, 25: 8192, 26: 12288}
        pinned, k_regions, v_regions = [], [], []
        swiftllm_c.register_kv_cache_info(2, 1, 4, 8)
        for layer in order:
            source = torch.full((1024,), layer, dtype=torch.uint8, pin_memory=True)
            pinned.append(source)
            base = owner.data_ptr() + offsets[layer]
            info = [{"packed": {"offset": 0, "shape": [1024], "dtype": torch.uint8, "size": 1024, "is_param": True}}]
            swiftllm_c.register_layer_memory_org_gpu(layer, base, 4096)
            swiftllm_c.register_layer_memory_quant(layer, source.data_ptr(), 1024)
            swiftllm_c.register_layer_memory_tensor_map_quant(layer, info)
            swiftllm_c.replace_layer_org2quant(layer)
            k_region, v_region = swiftllm_c.acquire_new_kvcache(layer)
            k_regions.append(k_region)
            v_regions.append(v_region)

        k_cache = torch.zeros((2, 2, 1, 4, 8), dtype=torch.float16, device="cuda")
        v_cache = torch.zeros_like(k_cache)
        block_table = torch.full((1, 4), -1, dtype=torch.int32, device="cuda")
        block_table[0, 0] = 26
        key = (torch.arange(8, device="cuda", dtype=torch.float32).reshape(1, 1, 8) + 1).half()
        value = (key + 20).half()
        model_config = SimpleNamespace(num_layers=2, num_kv_heads=1, num_q_heads=1, head_dim=8)
        engine_config = SimpleNamespace(block_size=4, max_blocks_per_seq=4)
        prefill = SimpleNamespace(
            num_prefill_seqs=1, max_prefill_len=1,
            seq_ids=torch.tensor([0], dtype=torch.int32, device="cuda"),
            prefill_seq_start_locs=torch.tensor([0], dtype=torch.int32, device="cuda"),
            prefill_seq_lens=torch.tensor([1], dtype=torch.int32, device="cuda"),
            num_decoding_seqs=0, num_prefill_tokens=1,
            decoding_seq_lens=torch.empty(0, dtype=torch.int32, device="cuda"),
            num_blocks_org=2, kv_cache_new_block_size=12, org_layer_param_size=2048,
        )
        store_module.store_kvcache_explicit_regions(
            key, value, k_cache, v_cache, k_regions, v_regions,
            block_table, model_config, engine_config, prefill, 0,
        )
        torch.cuda.synchronize()

        expected_region_exact = bool(
            torch.equal(k_regions[2][0, 0, 0, 0], key[0, 0])
            and torch.equal(v_regions[2][0, 0, 0, 0], value[0, 0])
        )
        unintended_layer23_unchanged = bool(torch.all(owner[1024:4096] == 0))
        self.assertTrue(expected_region_exact)
        self.assertTrue(unintended_layer23_unchanged)

        decode = SimpleNamespace(
            num_decoding_seqs=1, num_prefill_seqs=0, num_prefill_tokens=0,
            decoding_seq_lens=torch.tensor([1], dtype=torch.int32, device="cuda"),
            seq_ids=torch.tensor([0], dtype=torch.int32, device="cuda"),
            num_seq_blocks=1, seq_block_size=4, softmax_scale=8 ** -0.5,
            num_blocks_org=2, kv_cache_new_block_size=12, org_layer_param_size=2048,
        )
        query = torch.ones((1, 1, 8), dtype=torch.float16, device="cuda")
        output = torch.empty((1, 8), dtype=torch.float16, device="cuda")
        attention_module.paged_attention_multi_kernels(
            query, k_cache, v_cache, k_regions, v_regions, block_table,
            model_config, engine_config, decode, 0, output,
        )
        torch.cuda.synchronize()
        max_error = float((output[0] - value[0, 0]).abs().max())
        self.assertLessEqual(max_error, 0.001)

        path = os.environ.get("MORPHSERVE_TEST_OUTPUT")
        if path:
            Path(path).write_text(json.dumps({
                "profile_order": order,
                "expected_layer26_received_token": expected_region_exact,
                "unintended_layer23_unchanged": unintended_layer23_unchanged,
                "attention_max_abs_error": max_error,
                "output": output[0].float().cpu().tolist(),
                "expected": value[0, 0].float().cpu().tolist(),
            }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    unittest.main()
