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
    "candidate extension/kernel path required",
)
class NonContiguousProfileOrderTests(unittest.TestCase):
    def test_fixed_stride_misroutes_profile_group_two(self):
        kernel = load_candidate_kernel("kvcache_mgmt.py")
        owner = torch.zeros(4 * 4096, dtype=torch.uint8, device="cuda")
        order = [25, 24, 26]
        region_offset = {23: 0, 24: 4096, 25: 8192, 26: 12288}
        pinned = []
        k_regions, v_regions, addresses = [], [], []
        swiftllm_c.register_kv_cache_info(2, 1, 4, 8)
        for layer in order:
            source = torch.full((1024,), layer, dtype=torch.uint8, pin_memory=True)
            pinned.append(source)
            base = owner.data_ptr() + region_offset[layer]
            info = [{"packed": {"offset": 0, "shape": [1024], "dtype": torch.uint8, "size": 1024, "is_param": True}}]
            swiftllm_c.register_layer_memory_org_gpu(layer, base, 4096)
            swiftllm_c.register_layer_memory_quant(layer, source.data_ptr(), 1024)
            swiftllm_c.register_layer_memory_tensor_map_quant(layer, info)
            swiftllm_c.replace_layer_org2quant(layer)
            k_region, v_region = swiftllm_c.acquire_new_kvcache(layer)
            k_regions.append(k_region)
            v_regions.append(v_region)
            addresses.append(k_region.data_ptr())

        k_cache = torch.zeros((2, 2, 1, 4, 8), dtype=torch.float16, device="cuda")
        v_cache = torch.zeros_like(k_cache)
        block_table = torch.full((1, 4), -1, dtype=torch.int32, device="cuda")
        block_table[0, 0] = 26  # 2 original + group 2 * 12 blocks
        key = (torch.arange(8, device="cuda", dtype=torch.float32).reshape(1, 1, 8) + 1).half()
        value = (key + 20).half()
        model_config = SimpleNamespace(num_layers=2, num_kv_heads=1, head_dim=8)
        engine_config = SimpleNamespace(block_size=4, max_blocks_per_seq=4)
        state = SimpleNamespace(
            num_prefill_seqs=1,
            max_prefill_len=1,
            seq_ids=torch.tensor([0], dtype=torch.int32, device="cuda"),
            prefill_seq_start_locs=torch.tensor([0], dtype=torch.int32, device="cuda"),
            prefill_seq_lens=torch.tensor([1], dtype=torch.int32, device="cuda"),
            num_decoding_seqs=0,
            num_prefill_tokens=1,
            decoding_seq_lens=torch.empty(0, dtype=torch.int32, device="cuda"),
            num_blocks_org=2,
            kv_cache_new_block_size=12,
            org_layer_param_size=4096 // 2,
        )
        kernel.store_kvcache(
            key, value, k_cache, v_cache, k_regions, v_regions,
            block_table, model_config, engine_config, state, 0,
        )
        torch.cuda.synchronize()

        expected_layer26_unchanged = bool(torch.all(k_regions[2] == 0) and torch.all(v_regions[2] == 0))
        wrong_k = owner[1024:1024 + 1536].view(torch.float16).view(12, 2, 1, 4, 8)
        wrong_v = owner[2560:2560 + 1536].view(torch.float16).view(12, 2, 1, 4, 8)
        wrong_layer23_received = bool(
            torch.equal(wrong_k[0, 0, 0, 0], key[0, 0])
            and torch.equal(wrong_v[0, 0, 0, 0], value[0, 0])
        )
        fixed_stride_prediction = addresses[0] - 2 * 4096
        self.assertEqual(fixed_stride_prediction, owner.data_ptr() + 1024)
        self.assertNotEqual(fixed_stride_prediction, addresses[2])
        self.assertTrue(expected_layer26_unchanged)
        self.assertTrue(wrong_layer23_received)

        output = os.environ.get("MORPHSERVE_TEST_OUTPUT")
        if output:
            Path(output).write_text(json.dumps({
                "profile_order": order,
                "k_addresses": addresses,
                "layer_stride_bytes": 4096,
                "group2_fixed_stride_prediction": fixed_stride_prediction,
                "group2_expected_address": addresses[2],
                "expected_layer26_unchanged": expected_layer26_unchanged,
                "unintended_layer23_received_token": wrong_layer23_received,
            }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    unittest.main()
