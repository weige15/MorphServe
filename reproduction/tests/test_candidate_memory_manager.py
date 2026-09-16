import gc
import json
import os
import unittest
from pathlib import Path

try:
    import torch
    import swiftllm_c
except (ImportError, OSError):
    torch = None
    swiftllm_c = None


@unittest.skipUnless(torch is not None and swiftllm_c is not None and torch.cuda.is_available(), "candidate CUDA extension required")
class CandidateMemoryManagerTests(unittest.TestCase):
    @staticmethod
    def _pinned_pattern(size, offset=0):
        return ((torch.arange(size, dtype=torch.int64) + offset) % 251).to(torch.uint8).pin_memory()

    @staticmethod
    def _tensor_map(name, tensor, offset=0):
        return [{name: {
            "offset": offset,
            "shape": list(tensor.shape),
            "dtype": tensor.dtype,
            "size": tensor.numel() * tensor.element_size(),
            "is_param": True,
        }}]

    def test_in_place_copy_reclaimed_views_and_restore(self):
        layer_id = 1001
        original_bytes = 4096
        packed_bytes = 1024
        owner = torch.full((original_bytes,), 255, dtype=torch.uint8, device="cuda")
        original_cpu = self._pinned_pattern(original_bytes, offset=17)
        packed_cpu = self._pinned_pattern(packed_bytes, offset=3)

        swiftllm_c.register_layer_memory_org_gpu(layer_id, owner.data_ptr(), original_bytes)
        swiftllm_c.register_layer_memory_org_cpu(layer_id, original_cpu.data_ptr(), original_bytes)
        swiftllm_c.register_layer_memory_quant(layer_id, packed_cpu.data_ptr(), packed_bytes)
        swiftllm_c.register_layer_memory_tensor_map_org(layer_id, self._tensor_map("original", original_cpu))
        swiftllm_c.register_layer_memory_tensor_map_quant(layer_id, self._tensor_map("packed", packed_cpu))
        swiftllm_c.register_kv_cache_info(2, 1, 4, 8)

        copy_start = torch.cuda.Event(enable_timing=True)
        copy_end = torch.cuda.Event(enable_timing=True)
        copy_start.record()
        packed_gpu = swiftllm_c.replace_layer_org2quant(layer_id)
        copy_end.record()
        copy_end.synchronize()

        self.assertEqual(len(packed_gpu), 1)
        self.assertEqual(packed_gpu[0].data_ptr(), owner.data_ptr())
        self.assertTrue(torch.equal(packed_gpu[0].cpu(), packed_cpu))

        allocated_before = torch.cuda.memory_allocated()
        acquire_start = torch.cuda.Event(enable_timing=True)
        acquire_end = torch.cuda.Event(enable_timing=True)
        acquire_start.record()
        k_cache, v_cache = swiftllm_c.acquire_new_kvcache(layer_id)
        acquire_end.record()
        acquire_end.synchronize()
        allocated_after = torch.cuda.memory_allocated()

        self.assertEqual(list(k_cache.shape), [12, 2, 1, 4, 8])
        self.assertEqual(list(v_cache.shape), [12, 2, 1, 4, 8])
        self.assertEqual(k_cache.data_ptr(), owner.data_ptr() + packed_bytes)
        self.assertEqual(v_cache.data_ptr(), k_cache.data_ptr() + k_cache.numel() * k_cache.element_size())
        self.assertEqual(v_cache.data_ptr() + v_cache.numel() * v_cache.element_size(), owner.data_ptr() + original_bytes)
        self.assertLessEqual(allocated_after - allocated_before, 4096)

        k_cache.fill_(1.5)
        v_cache.fill_(-0.5)
        torch.cuda.synchronize()
        self.assertFalse(torch.all(owner[packed_bytes:] == 255).item())

        metrics = {
            "device": torch.cuda.get_device_name(),
            "original_bytes": original_bytes,
            "packed_bytes": packed_bytes,
            "reclaimed_bytes": original_bytes - packed_bytes,
            "kv_blocks": k_cache.shape[0],
            "kv_shape": list(k_cache.shape),
            "copy_ms": copy_start.elapsed_time(copy_end),
            "acquire_and_zero_ms": acquire_start.elapsed_time(acquire_end),
            "allocator_delta_bytes": allocated_after - allocated_before,
            "owner_base": owner.data_ptr(),
            "packed_base": packed_gpu[0].data_ptr(),
            "k_base": k_cache.data_ptr(),
            "v_base": v_cache.data_ptr(),
            "owner_end": owner.data_ptr() + original_bytes,
        }

        del k_cache, v_cache, packed_gpu
        gc.collect()
        torch.cuda.synchronize()

        restore_start = torch.cuda.Event(enable_timing=True)
        restore_end = torch.cuda.Event(enable_timing=True)
        restore_start.record()
        restored_gpu = swiftllm_c.replace_layer_quant2org(layer_id)
        restore_end.record()
        restore_end.synchronize()
        metrics["restore_ms"] = restore_start.elapsed_time(restore_end)

        self.assertEqual(restored_gpu[0].data_ptr(), owner.data_ptr())
        self.assertTrue(torch.equal(owner.cpu(), original_cpu))

        output = os.environ.get("MORPHSERVE_TEST_OUTPUT")
        if output:
            Path(output).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    def test_rejects_alignment_past_registered_region(self):
        layer_id = 1003
        owner = torch.empty(201, dtype=torch.uint8, device="cuda")
        packed_cpu = torch.empty(1, dtype=torch.uint8, pin_memory=True)
        swiftllm_c.register_layer_memory_org_gpu(layer_id, owner.data_ptr() + 1, 200)
        swiftllm_c.register_layer_memory_quant(layer_id, packed_cpu.data_ptr(), packed_cpu.numel())
        swiftllm_c.register_kv_cache_info(2, 1, 4, 8)

        with self.assertRaisesRegex(RuntimeError, "No aligned space"):
            swiftllm_c.acquire_new_kvcache(layer_id)

    def test_rejects_tail_smaller_than_one_kv_block(self):
        layer_id = 1002
        owner = torch.empty(1024, dtype=torch.uint8, device="cuda")
        packed_cpu = torch.empty(1000, dtype=torch.uint8, pin_memory=True)
        swiftllm_c.register_layer_memory_org_gpu(layer_id, owner.data_ptr(), owner.numel())
        swiftllm_c.register_layer_memory_quant(layer_id, packed_cpu.data_ptr(), packed_cpu.numel())
        swiftllm_c.register_kv_cache_info(2, 1, 4, 8)

        with self.assertRaisesRegex(RuntimeError, r"No (aligned )?space available"):
            swiftllm_c.acquire_new_kvcache(layer_id)


if __name__ == "__main__":
    unittest.main()
