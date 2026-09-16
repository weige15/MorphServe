import importlib.util
import json
import math
import os
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
    import swiftllm_c
except (ImportError, OSError):
    torch = None
    swiftllm_c = None


def load_candidate_kernel(filename):
    candidate_root = Path(os.environ["MORPHSERVE_CANDIDATE_PYTHON"])
    stubs = {
        "swiftllm": types.ModuleType("swiftllm"),
        "swiftllm.worker": types.ModuleType("swiftllm.worker"),
        "swiftllm.model_config": types.ModuleType("swiftllm.model_config"),
        "swiftllm.engine_config": types.ModuleType("swiftllm.engine_config"),
        "swiftllm.worker.infer_state": types.ModuleType("swiftllm.worker.infer_state"),
        "swiftllm.utils": types.ModuleType("swiftllm.utils"),
    }
    stubs["swiftllm.model_config"].LlamaModelConfig = object
    stubs["swiftllm.engine_config"].EngineConfig = object
    stubs["swiftllm.worker.infer_state"].LlamaInferState = object
    stubs["swiftllm.utils"].cdiv = lambda a, b: (a + b - 1) // b
    previous = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        path = candidate_root / "worker" / "kernels" / filename
        spec = importlib.util.spec_from_file_location(f"candidate_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


@unittest.skipUnless(
    torch is not None
    and swiftllm_c is not None
    and torch.cuda.is_available()
    and "MORPHSERVE_CANDIDATE_PYTHON" in os.environ,
    "candidate CUDA extension and Python kernel path required",
)
class CandidateKVMappingTests(unittest.TestCase):
    @staticmethod
    def _tensor_map(name, tensor):
        return [{name: {
            "offset": 0,
            "shape": list(tensor.shape),
            "dtype": tensor.dtype,
            "size": tensor.numel() * tensor.element_size(),
            "is_param": True,
        }}]

    def test_two_reclaimed_regions_store_and_attention(self):
        kvcache_mgmt = load_candidate_kernel("kvcache_mgmt.py")
        paged_attn = load_candidate_kernel("paged_attn.py")
        layer_bytes = 4096
        packed_bytes = 1024
        layer_stride_elements = layer_bytes // 2
        owner = torch.full((2 * layer_bytes,), 0xA5, dtype=torch.uint8, device="cuda")
        packed_high = torch.full((packed_bytes,), 0x11, dtype=torch.uint8, pin_memory=True)
        packed_low = torch.full((packed_bytes,), 0x22, dtype=torch.uint8, pin_memory=True)

        high_id, low_id = 2101, 2100
        swiftllm_c.register_layer_memory_org_gpu(high_id, owner.data_ptr() + layer_bytes, layer_bytes)
        swiftllm_c.register_layer_memory_quant(high_id, packed_high.data_ptr(), packed_bytes)
        swiftllm_c.register_layer_memory_tensor_map_quant(high_id, self._tensor_map("packed_high", packed_high))
        swiftllm_c.register_layer_memory_org_gpu(low_id, owner.data_ptr(), layer_bytes)
        swiftllm_c.register_layer_memory_quant(low_id, packed_low.data_ptr(), packed_bytes)
        swiftllm_c.register_layer_memory_tensor_map_quant(low_id, self._tensor_map("packed_low", packed_low))
        swiftllm_c.register_kv_cache_info(2, 1, 4, 8)

        # Match the candidate runtime's back-to-front append order.
        high_packed_view = swiftllm_c.replace_layer_org2quant(high_id)
        low_packed_view = swiftllm_c.replace_layer_org2quant(low_id)
        allocated_before = torch.cuda.memory_allocated()
        k_high, v_high = swiftllm_c.acquire_new_kvcache(high_id)
        k_low, v_low = swiftllm_c.acquire_new_kvcache(low_id)
        torch.cuda.synchronize()
        allocated_after = torch.cuda.memory_allocated()

        self.assertEqual(k_high.data_ptr() - layer_stride_elements * 2, k_low.data_ptr())
        self.assertEqual(v_high.data_ptr() - layer_stride_elements * 2, v_low.data_ptr())
        self.assertEqual(allocated_after - allocated_before, 0)
        self.assertEqual(k_high.data_ptr(), owner.data_ptr() + layer_bytes + packed_bytes)
        self.assertEqual(k_low.data_ptr(), owner.data_ptr() + packed_bytes)
        packed_prefix_before = torch.cat((owner[:packed_bytes], owner[layer_bytes:layer_bytes + packed_bytes])).clone()

        model_config = SimpleNamespace(num_layers=2, num_kv_heads=1, num_q_heads=1, head_dim=8)
        engine_config = SimpleNamespace(block_size=4, max_blocks_per_seq=4)
        k_cache = torch.zeros((2, 2, 1, 4, 8), dtype=torch.float16, device="cuda")
        v_cache = torch.zeros_like(k_cache)
        block_table = torch.tensor([[0, 2, 14, 15]], dtype=torch.int32, device="cuda")

        prefill_k = (torch.arange(12 * 8, device="cuda", dtype=torch.float32).reshape(12, 1, 8) / 100).half()
        prefill_v = (torch.arange(12 * 8, device="cuda", dtype=torch.float32).reshape(12, 1, 8) / 50 + 1).half()
        prefill_state = SimpleNamespace(
            num_prefill_seqs=1,
            max_prefill_len=12,
            seq_ids=torch.tensor([0], dtype=torch.int32, device="cuda"),
            prefill_seq_start_locs=torch.tensor([0], dtype=torch.int32, device="cuda"),
            prefill_seq_lens=torch.tensor([12], dtype=torch.int32, device="cuda"),
            num_decoding_seqs=0,
            num_prefill_tokens=12,
            decoding_seq_lens=torch.empty(0, dtype=torch.int32, device="cuda"),
            num_blocks_org=2,
            kv_cache_new_block_size=12,
            org_layer_param_size=layer_stride_elements,
        )
        kvcache_mgmt.store_kvcache(
            prefill_k, prefill_v, k_cache, v_cache,
            [k_high, k_low], [v_high, v_low], block_table,
            model_config, engine_config, prefill_state, 0,
        )
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(k_cache[0, 0, 0], prefill_k[:4, 0]))
        self.assertTrue(torch.equal(v_cache[0, 0, 0], prefill_v[:4, 0]))
        self.assertTrue(torch.equal(k_high[0, 0, 0], prefill_k[4:8, 0]))
        self.assertTrue(torch.equal(v_high[0, 0, 0], prefill_v[4:8, 0]))
        self.assertTrue(torch.equal(k_low[0, 0, 0], prefill_k[8:12, 0]))
        self.assertTrue(torch.equal(v_low[0, 0, 0], prefill_v[8:12, 0]))

        decode_k = (torch.arange(8, device="cuda", dtype=torch.float32).reshape(1, 1, 8) / 25 + 2).half()
        decode_v = (torch.arange(8, device="cuda", dtype=torch.float32).reshape(1, 1, 8) / 20 + 3).half()
        decode_state = SimpleNamespace(
            num_prefill_seqs=0,
            max_prefill_len=0,
            seq_ids=torch.tensor([0], dtype=torch.int32, device="cuda"),
            prefill_seq_start_locs=torch.empty(0, dtype=torch.int32, device="cuda"),
            prefill_seq_lens=torch.empty(0, dtype=torch.int32, device="cuda"),
            num_decoding_seqs=1,
            num_prefill_tokens=0,
            decoding_seq_lens=torch.tensor([13], dtype=torch.int32, device="cuda"),
            num_blocks_org=2,
            kv_cache_new_block_size=12,
            org_layer_param_size=layer_stride_elements,
            softmax_scale=8 ** -0.5,
            seq_block_size=16,
            num_seq_blocks=1,
        )
        kvcache_mgmt.store_kvcache(
            decode_k, decode_v, k_cache, v_cache,
            [k_high, k_low], [v_high, v_low], block_table,
            model_config, engine_config, decode_state, 0,
        )
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(k_low[1, 0, 0, 0], decode_k[0, 0]))
        self.assertTrue(torch.equal(v_low[1, 0, 0, 0], decode_v[0, 0]))

        q = (torch.arange(8, device="cuda", dtype=torch.float32).reshape(1, 1, 8) / 10 - 0.2).half()
        output = torch.empty((1, 8), dtype=torch.float16, device="cuda")
        paged_attn.paged_attention(
            q, k_cache, v_cache, [k_high, k_low], [v_high, v_low], block_table,
            model_config, engine_config, decode_state, 0, output,
        )
        torch.cuda.synchronize()

        keys = torch.cat((prefill_k[:, 0], decode_k[:, 0]), dim=0).float()
        values = torch.cat((prefill_v[:, 0], decode_v[:, 0]), dim=0).float()
        scores = torch.mv(keys, q[0, 0].float()) * decode_state.softmax_scale
        expected = torch.matmul(torch.softmax(scores, dim=0), values).half()
        torch.testing.assert_close(output[0], expected, atol=0.02, rtol=0.02)
        packed_prefix_after = torch.cat((owner[:packed_bytes], owner[layer_bytes:layer_bytes + packed_bytes]))
        self.assertTrue(torch.equal(packed_prefix_after, packed_prefix_before))

        metrics = {
            "device": torch.cuda.get_device_name(),
            "block_table": block_table.cpu().tolist(),
            "owner_base": owner.data_ptr(),
            "k_low_base": k_low.data_ptr(),
            "k_high_base": k_high.data_ptr(),
            "layer_stride_bytes": layer_bytes,
            "group_blocks": k_high.shape[0],
            "allocator_delta_bytes": allocated_after - allocated_before,
            "max_attention_abs_error": (output[0] - expected).abs().max().item(),
            "output": output[0].float().cpu().tolist(),
            "expected": expected.float().cpu().tolist(),
            "packed_prefixes_preserved": True,
        }
        output_path = os.environ.get("MORPHSERVE_TEST_OUTPUT")
        if output_path:
            Path(output_path).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

        # Keep all external-owner tensors and returned views alive through kernel completion.
        self.assertEqual(high_packed_view[0].data_ptr(), owner.data_ptr() + layer_bytes)
        self.assertEqual(low_packed_view[0].data_ptr(), owner.data_ptr())


if __name__ == "__main__":
    unittest.main()
