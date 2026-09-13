"""Focused checks for the explicit runtime morphing substrate.

CUDA kernel checks are opt-in so the normal CPU-safe suite remains cheap:
``MORPHSERVE_RUN_CUDA_TESTS=1 python -m unittest benchmark.test_runtime_morphing``.
"""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
import unittest

import torch

from swiftllm.engine_config import EngineConfig
from swiftllm.model_config import LlamaModelConfig
from swiftllm.server.engine import Engine
from swiftllm.server.scheduler import Scheduler
from swiftllm.server.structs import RawRequest, Request


TINY_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_act": "silu",
    "hidden_size": 32,
    "intermediate_size": 64,
    "max_position_embeddings": 128,
    "model_type": "llama",
    "num_attention_heads": 2,
    "num_hidden_layers": 2,
    "num_key_value_heads": 1,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
    "vocab_size": 128,
}


def tiny_engine_config() -> EngineConfig:
    return EngineConfig(
        model_path=".",
        use_dummy=True,
        block_size=4,
        gpu_mem_utilization=0.9,
        num_cpu_blocks=4,
        max_seqs_in_block_table=8,
        max_blocks_per_seq=8,
        max_batch_size=4,
        max_tokens_in_batch=32,
    )


class RuntimeContractTests(unittest.TestCase):
    def test_runtime_config_is_explicit_and_disabled_by_default(self):
        config = tiny_engine_config()
        self.assertFalse(config.enable_runtime_morphing)
        self.assertEqual(config.runtime_awq_target_blocks, 4170)
        self.assertFalse(config.runtime_verify_kv)

    def test_restore_admission_hold_preserves_fcfs_queues(self):
        config = tiny_engine_config()
        scheduler = Scheduler(LlamaModelConfig(TINY_CONFIG), config, 4)
        first = Request(RawRequest("a", 2))
        second = Request(RawRequest("b", 2))
        first.prompt_len = second.prompt_len = 4
        scheduler.on_requests_arrival([first, second])
        scheduler.admissions_paused = True
        batch, swapped_in, swapped_out = scheduler.get_next_batch()
        self.assertEqual(batch, [])
        self.assertEqual(list(swapped_in), [])
        self.assertEqual(list(swapped_out), [])
        self.assertEqual(list(scheduler.waiting_q), [first, second])
        scheduler.admissions_paused = False
        batch, _, _ = scheduler.get_next_batch()
        self.assertEqual(batch, [first, second])


class TransitionSerializationTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def bare_engine(state="FP16", *, allocated=0, base=2):
        engine = object.__new__(Engine)
        engine.initialized = True
        engine.engine_config = SimpleNamespace(enable_runtime_morphing=True)
        engine.model = SimpleNamespace(
            runtime_precision_state=state,
            base_num_blocks=base,
            gpu_block_manager=SimpleNamespace(
                num_blocks=4,
                num_free_blocks=4 - allocated,
            ),
        )
        engine._main_loop_running = False
        engine._pending_transition = None
        engine._transition_lock = asyncio.Lock()
        return engine

    async def test_direct_manual_transitions_are_serialized(self):
        engine = self.bare_engine()
        active = 0
        maximum = 0

        async def apply(target):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.01)
            active -= 1
            return {"status": "success", "precision_after": target}

        engine._apply_transition = apply
        await asyncio.gather(
            engine._request_transition("AWQ_MARLIN_W4_16"),
            engine._request_transition("AWQ_MARLIN_W4_16"),
        )
        self.assertEqual(maximum, 1)

    async def test_direct_restore_rejects_over_capacity_without_poisoning_state(self):
        engine = self.bare_engine("AWQ_MARLIN_W4_16", allocated=3, base=2)
        with self.assertRaisesRegex(RuntimeError, "cannot drain"):
            await engine._request_transition("FP16")
        self.assertEqual(engine.model.runtime_precision_state, "AWQ_MARLIN_W4_16")


_RUN_CUDA = os.environ.get("MORPHSERVE_RUN_CUDA_TESTS") == "1" and torch.cuda.is_available()


@unittest.skipUnless(_RUN_CUDA, "set MORPHSERVE_RUN_CUDA_TESTS=1 with CUDA available")
class SegmentedKVKernelTests(unittest.TestCase):
    def setUp(self):
        from swiftllm.worker.infer_state import LlamaInferState

        self.LlamaInferState = LlamaInferState
        self.model_config = LlamaModelConfig(TINY_CONFIG)
        self.engine_config = tiny_engine_config()
        self.base_blocks = 2
        shape = (2, 2, 1, 4, 16)
        self.k_base = torch.zeros(shape, dtype=torch.float16, device="cuda")
        self.v_base = torch.zeros_like(self.k_base)
        self.k_extension = torch.zeros_like(self.k_base)
        self.v_extension = torch.zeros_like(self.v_base)
        self.block_table = torch.zeros((8, 8), dtype=torch.int32, device="cuda")
        self.block_table[0, :3] = torch.tensor([0, 2, 3], dtype=torch.int32, device="cuda")

    def _state(self, *, prefill_len=0, decode_len=0):
        prefill = torch.tensor([prefill_len], dtype=torch.int32, device="cuda") if prefill_len else torch.empty(0, dtype=torch.int32, device="cuda")
        decode = torch.tensor([decode_len], dtype=torch.int32, device="cuda") if decode_len else torch.empty(0, dtype=torch.int32, device="cuda")
        num_tokens = prefill_len + bool(decode_len)
        return self.LlamaInferState(
            batch_size=int(bool(prefill_len)) + int(bool(decode_len)),
            num_tokens=num_tokens,
            seq_ids=torch.tensor([0] * (int(bool(prefill_len)) + int(bool(decode_len))), dtype=torch.int32, device="cuda"),
            softmax_scale=16 ** -0.5,
            num_prefill_seqs=int(bool(prefill_len)),
            num_prefill_tokens=prefill_len,
            prefill_seq_start_locs=torch.tensor([0], dtype=torch.int32, device="cuda") if prefill_len else torch.empty(0, dtype=torch.int32, device="cuda"),
            prefill_seq_start_locs_with_end=torch.tensor([0, prefill_len], dtype=torch.int32, device="cuda") if prefill_len else torch.tensor([0], dtype=torch.int32, device="cuda"),
            prefill_seq_lens=prefill,
            max_prefill_len=prefill_len,
            num_decoding_seqs=int(bool(decode_len)),
            decoding_seq_lens=decode,
            max_decoding_len=decode_len,
            seq_block_size=64,
            num_seq_blocks=1 if decode_len else 0,
            position_cos=torch.empty((num_tokens, 8), dtype=torch.float16, device="cuda"),
            position_sin=torch.empty((num_tokens, 8), dtype=torch.float16, device="cuda"),
            ignore_kvcache=False,
        )

    def test_store_and_attention_cross_base_extension_boundary(self):
        from swiftllm.worker.kernels.kvcache_mgmt import store_kvcache
        from swiftllm.worker.kernels.paged_attn import paged_attention

        k = torch.arange(8 * 16, dtype=torch.float16, device="cuda").reshape(8, 1, 16) / 100
        v = torch.flip(k, dims=(0,)).contiguous()
        state = self._state(prefill_len=8)
        store_kvcache(
            k,
            v,
            self.k_base,
            self.v_base,
            self.block_table,
            self.model_config,
            self.engine_config,
            state,
            0,
            self.k_extension,
            self.v_extension,
            self.base_blocks,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(self.k_base[0, 0].reshape(4, 16), k[:4, 0])
        torch.testing.assert_close(self.k_extension[0, 0].reshape(4, 16), k[4:, 0])

        q = torch.linspace(-0.5, 0.5, 32, dtype=torch.float16, device="cuda").reshape(1, 2, 16)
        out = torch.empty_like(q)
        decode_state = self._state(decode_len=8)
        paged_attention(
            q,
            self.k_base,
            self.v_base,
            self.block_table,
            self.model_config,
            self.engine_config,
            decode_state,
            0,
            out,
            self.k_extension,
            self.v_extension,
            self.base_blocks,
        )
        torch.cuda.synchronize()
        keys = k[:, 0].float()
        values = v[:, 0].float()
        reference = []
        for head in q[0].float():
            scores = (keys @ head) * decode_state.softmax_scale
            reference.append(torch.softmax(scores, dim=0) @ values)
        reference = torch.stack(reference).to(torch.float16)
        torch.testing.assert_close(out[0], reference, rtol=3e-2, atol=3e-2)

    def test_decode_store_can_address_extension(self):
        from swiftllm.worker.kernels.kvcache_mgmt import store_kvcache

        k = torch.full((1, 1, 16), 3.0, dtype=torch.float16, device="cuda")
        v = torch.full((1, 1, 16), 5.0, dtype=torch.float16, device="cuda")
        state = self._state(decode_len=9)
        store_kvcache(
            k,
            v,
            self.k_base,
            self.v_base,
            self.block_table,
            self.model_config,
            self.engine_config,
            state,
            0,
            self.k_extension,
            self.v_extension,
            self.base_blocks,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(self.k_extension[1, 0, 0, 0], k[0, 0])
        torch.testing.assert_close(self.v_extension[1, 0, 0, 0], v[0, 0])

    def test_segmented_swap_round_trip_uses_virtual_ids(self):
        from swiftllm.worker.block_manager import BlockManager
        from swiftllm.worker.model import LlamaModel

        model = object.__new__(LlamaModel)
        model.engine_config = self.engine_config
        model.base_num_blocks = 2
        model.k_cache = self.k_base
        model.v_cache = self.v_base
        model.k_cache_extension = self.k_extension
        model.v_cache_extension = self.v_extension
        model.k_swap = torch.zeros((4, 2, 1, 4, 16), dtype=torch.float16)
        model.v_swap = torch.zeros_like(model.k_swap)
        model.gpu_block_manager = BlockManager("GPU", 4, 4, 4, 4)
        model.cpu_block_manager = BlockManager("CPU", 4, 4, 4, 4)
        seq0 = torch.tensor([0], dtype=torch.int32, device="cuda")
        seq1 = torch.tensor([1], dtype=torch.int32, device="cuda")
        with torch.inference_mode():
            model.gpu_block_manager.allocate_blocks_for_seqs(
                seq0, torch.tensor([4], dtype=torch.int32, device="cuda")
            )
            model.k_cache[0].fill_(7)
            model.v_cache[0].fill_(9)
            model.swap_out_seqs([0])
        torch.cuda.synchronize()
        self.assertTrue(bool((model.k_swap[0] == 7).all()))
        self.assertTrue(bool((model.v_swap[0] == 9).all()))

        with torch.inference_mode():
            model.gpu_block_manager.allocate_blocks_for_seqs(
                seq1, torch.tensor([8], dtype=torch.int32, device="cuda")
            )
            model.swap_in_seqs([0])
        torch.cuda.synchronize()
        self.assertEqual(model.gpu_block_manager.block_table[0, 0].item(), 2)
        self.assertTrue(bool((model.k_cache_extension[0] == 7).all()))
        self.assertTrue(bool((model.v_cache_extension[0] == 9).all()))

        with torch.inference_mode():
            model.k_cache_extension[0].fill_(11)
            model.v_cache_extension[0].fill_(13)
            model.swap_out_seqs([0])
        torch.cuda.synchronize()
        self.assertTrue(bool((model.k_swap[0] == 11).all()))
        self.assertTrue(bool((model.v_swap[0] == 13).all()))

    def test_shrink_rejects_usage_above_retained_capacity(self):
        from swiftllm.worker.block_manager import BlockManager

        manager = BlockManager("GPU", 2, 4, 4, 4)
        manager.extend(2)
        seq0 = torch.tensor([0], dtype=torch.int32, device="cuda")
        manager.allocate_blocks_for_seqs(
            seq0, torch.tensor([12], dtype=torch.int32, device="cuda")
        )
        torch.cuda.synchronize()
        with self.assertRaisesRegex(RuntimeError, "cannot compact"):
            manager.plan_compaction_to_prefix(2)
        with self.assertRaisesRegex(RuntimeError, "suffix blocks are allocated"):
            manager.shrink(2)

    def test_block_manager_grow_compact_shrink_preserves_logical_slots(self):
        from swiftllm.worker.block_manager import BlockManager

        manager = BlockManager("GPU", 2, 4, 4, 4)
        seq0 = torch.tensor([0], dtype=torch.int32, device="cuda")
        seq1 = torch.tensor([1], dtype=torch.int32, device="cuda")
        manager.allocate_blocks_for_seqs(seq0, torch.tensor([8], dtype=torch.int32, device="cuda"))
        manager.extend(2)
        manager.allocate_blocks_for_seqs(seq1, torch.tensor([8], dtype=torch.int32, device="cuda"))
        torch.cuda.synchronize()
        self.assertEqual(manager.block_table[1, :2].cpu().tolist(), [2, 3])
        manager.free_blocks_for_seqs(seq0)
        torch.cuda.synchronize()
        sources, targets = manager.plan_compaction_to_prefix(2)
        self.assertEqual(sources.cpu().tolist(), [2, 3])
        self.assertEqual(targets.cpu().tolist(), [0, 1])
        manager.commit_compaction(sources, targets)
        torch.cuda.synchronize()
        self.assertEqual(manager.block_table[1, :2].cpu().tolist(), [0, 1])
        manager.shrink(2)
        self.assertEqual(manager.num_blocks, 2)
        self.assertEqual(manager.num_free_blocks, 0)


if __name__ == "__main__":
    unittest.main()
