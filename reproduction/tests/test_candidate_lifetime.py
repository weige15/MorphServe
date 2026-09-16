import gc
import json
import os
import time
import unittest
from pathlib import Path

try:
    import torch
    import swiftllm_c
except (ImportError, OSError):
    torch = None
    swiftllm_c = None


@unittest.skipUnless(torch is not None and swiftllm_c is not None and torch.cuda.is_available(), "candidate CUDA extension required")
class CandidateLifetimeTests(unittest.TestCase):
    metrics = {}

    @staticmethod
    def _pattern(size, offset):
        return ((torch.arange(size, dtype=torch.int64) + offset) % 251).to(torch.uint8).pin_memory()

    @staticmethod
    def _map(name, tensor):
        return [{name: {
            "offset": 0,
            "shape": list(tensor.shape),
            "dtype": tensor.dtype,
            "size": tensor.numel() * tensor.element_size(),
            "is_param": True,
        }}]

    def _layer(self, layer_id, original_offset):
        owner = torch.full((4096,), 0xA5, dtype=torch.uint8, device="cuda")
        original = self._pattern(4096, original_offset)
        packed = self._pattern(1024, original_offset + 19)
        swiftllm_c.register_layer_memory_org_gpu(layer_id, owner.data_ptr(), owner.numel())
        swiftllm_c.register_layer_memory_org_cpu(layer_id, original.data_ptr(), original.numel())
        swiftllm_c.register_layer_memory_quant(layer_id, packed.data_ptr(), packed.numel())
        swiftllm_c.register_layer_memory_tensor_map_org(layer_id, self._map("original", original))
        swiftllm_c.register_layer_memory_tensor_map_quant(layer_id, self._map("packed", packed))
        swiftllm_c.register_kv_cache_info(2, 1, 4, 8)
        return owner, original, packed

    def test_in_flight_reclaimed_write_corrupts_uncoordinated_restore(self):
        owner, original, packed = self._layer(3101, 7)
        packed_view = swiftllm_c.replace_layer_org2quant(3101)
        k_cache, v_cache = swiftllm_c.acquire_new_kvcache(3101)
        writer = torch.cuda.Stream()
        done = torch.cuda.Event()
        with torch.cuda.stream(writer):
            torch.cuda._sleep(200_000_000)
            k_cache.fill_(7.0)
            done.record()
        self.assertFalse(done.query())

        restore_start = time.perf_counter()
        restored_view = swiftllm_c.replace_layer_quant2org(3101)
        restore_wall_ms = (time.perf_counter() - restore_start) * 1000
        writer_done_when_restore_returned = done.query()
        done.synchronize()
        torch.cuda.synchronize()
        corrupt_bytes = int((owner.cpu() != original).sum().item())

        self.assertFalse(writer_done_when_restore_returned)
        self.assertGreater(corrupt_bytes, 0)
        self.metrics["unsafe_race"] = {
            "writer_done_when_restore_returned": writer_done_when_restore_returned,
            "corrupt_bytes_after_writer": corrupt_bytes,
            "restore_wall_ms": restore_wall_ms,
        }
        del packed_view, restored_view, k_cache, v_cache, owner, original, packed
        gc.collect()

        control_owner, control_original, control_packed = self._layer(3102, 31)
        control_packed_view = swiftllm_c.replace_layer_org2quant(3102)
        control_k, control_v = swiftllm_c.acquire_new_kvcache(3102)
        control_writer = torch.cuda.Stream()
        control_done = torch.cuda.Event()
        with torch.cuda.stream(control_writer):
            torch.cuda._sleep(200_000_000)
            control_k.fill_(7.0)
            control_done.record()
        control_done.synchronize()
        control_restored_view = swiftllm_c.replace_layer_quant2org(3102)
        torch.cuda.synchronize()
        control_corrupt_bytes = int((control_owner.cpu() != control_original).sum().item())

        self.assertEqual(control_corrupt_bytes, 0)
        self.metrics["explicit_wait_control"] = {"corrupt_bytes": control_corrupt_bytes}
        del control_packed_view, control_restored_view, control_k, control_v
        del control_owner, control_original, control_packed
        gc.collect()

        event_owner, event_original, event_packed = self._layer(3104, 83)
        event_packed_view = swiftllm_c.replace_layer_org2quant(3104)
        event_k, event_v = swiftllm_c.acquire_new_kvcache(3104)
        event_writer = torch.cuda.Stream()
        event_done = torch.cuda.Event()
        with torch.cuda.stream(event_writer):
            torch.cuda._sleep(200_000_000)
            event_k.fill_(7.0)
            swiftllm_c.record_layer_memory_use(3104)
            event_done.record()
        writer_unfinished_before_restore = not event_done.query()
        event_restore_start = time.perf_counter()
        event_restored_view = swiftllm_c.replace_layer_quant2org(3104)
        event_restore_wall_ms = (time.perf_counter() - event_restore_start) * 1000
        event_done.synchronize()
        torch.cuda.synchronize()
        event_corrupt_bytes = int((event_owner.cpu() != event_original).sum().item())

        self.assertTrue(writer_unfinished_before_restore)
        self.assertEqual(event_corrupt_bytes, 0)
        self.metrics["recorded_event_control"] = {
            "writer_unfinished_before_restore": writer_unfinished_before_restore,
            "corrupt_bytes": event_corrupt_bytes,
            "restore_wall_ms": event_restore_wall_ms,
        }
        del event_packed_view, event_restored_view, event_k, event_v
        del event_owner, event_original, event_packed
        gc.collect()

    def test_repeated_synchronized_expand_restore(self):
        owner, original, packed = self._layer(3103, 53)
        owner_base = owner.data_ptr()
        allocator_deltas = []
        for cycle in range(5):
            packed_view = swiftllm_c.replace_layer_org2quant(3103)
            before = torch.cuda.memory_allocated()
            k_cache, v_cache = swiftllm_c.acquire_new_kvcache(3103)
            torch.cuda.synchronize()
            allocator_deltas.append(torch.cuda.memory_allocated() - before)
            self.assertEqual(packed_view[0].data_ptr(), owner_base)
            k_cache.fill_(cycle + 1)
            v_cache.fill_(-(cycle + 1))
            torch.cuda.synchronize()
            del k_cache, v_cache, packed_view
            gc.collect()
            torch.cuda.synchronize()
            restored_view = swiftllm_c.replace_layer_quant2org(3103)
            torch.cuda.synchronize()
            self.assertEqual(restored_view[0].data_ptr(), owner_base)
            self.assertTrue(torch.equal(owner.cpu(), original))
            del restored_view
            gc.collect()

        self.metrics["repeated_cycles"] = {
            "cycles": 5,
            "owner_base_stable": True,
            "allocator_deltas_bytes": allocator_deltas,
            "exact_restore_each_cycle": True,
        }
        del owner, original, packed
        gc.collect()

    def test_repeated_event_protected_restore(self):
        owner, original, packed = self._layer(3105, 107)
        owner_base = owner.data_ptr()
        allocator_deltas = []
        for cycle in range(3):
            packed_view = swiftllm_c.replace_layer_org2quant(3105)
            before = torch.cuda.memory_allocated()
            k_cache, v_cache = swiftllm_c.acquire_new_kvcache(3105)
            torch.cuda.synchronize()
            allocator_deltas.append(torch.cuda.memory_allocated() - before)
            writer = torch.cuda.Stream()
            with torch.cuda.stream(writer):
                torch.cuda._sleep(20_000_000)
                k_cache.fill_(cycle + 1)
                v_cache.fill_(-(cycle + 1))
                swiftllm_c.record_layer_memory_use(3105)
            restored_view = swiftllm_c.replace_layer_quant2org(3105)
            torch.cuda.synchronize()
            self.assertEqual(packed_view[0].data_ptr(), owner_base)
            self.assertEqual(restored_view[0].data_ptr(), owner_base)
            self.assertTrue(torch.equal(owner.cpu(), original))
            del k_cache, v_cache, packed_view, restored_view, writer
            gc.collect()

        self.metrics["event_protected_cycles"] = {
            "cycles": 3,
            "owner_base_stable": True,
            "allocator_deltas_bytes": allocator_deltas,
            "exact_restore_each_cycle": True,
        }
        del owner, original, packed
        gc.collect()

    @classmethod
    def tearDownClass(cls):
        output = os.environ.get("MORPHSERVE_TEST_OUTPUT")
        if output:
            Path(output).write_text(json.dumps(cls.metrics, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    unittest.main()
