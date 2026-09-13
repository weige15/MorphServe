import gc
import hashlib
import itertools
import math
import time

import torch

from swiftllm.engine_config import EngineConfig
from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.weight import (
    layer_variant_bytes,
    layer_variant_layout,
    layer_variant_tensor_items,
    load_prepared_awq_layer,
    load_weights,
    materialize_layer_variant,
    prepare_layer_variant,
)
from swiftllm.worker.block_manager import BlockManager
from swiftllm.utils import GB
import swiftllm_c

from .layers.pre_layer import LlamaPreLayer
from .layers.transformer_layer import LlamaTransformerLayer
from .layers.post_layer import LlamaPostLayer
from .infer_state import LlamaInferState

class LlamaModel:
    """
    LlamaModel - A Llama model that can be used for inference.

    This class also acts as a "worker" that resides on a particular GPU, waiting
    for the control plane (the scheduler) to send commands.

    To initialize, please:
    - call __init__()
    - call load_weights()
    - call profile_num_blocks() on one worker
    - call init_kvcache_and_swap()
    """

    @torch.inference_mode()
    def __init__(
        self,
        engine_config: EngineConfig
    ):
        """
        Initialize the LlamaModel.
        """
        self.engine_config = engine_config

        # Load model config
        self.model_config = LlamaModelConfig.load_from_model_path(engine_config.model_path)

        # Weight and RoPE cache
        self.weight = None
        self._cos_cached = self._sin_cached = None

        # Layers
        self.pre_layer = None
        self.transformer_layers = None
        self.post_layer = None

        # KV Cache. Runtime morphing keeps the original allocation as an
        # immutable-ID base segment and adds at most one AWQ extension segment.
        self.num_blocks = None
        self.base_num_blocks = None
        self.k_cache = self.v_cache = None
        self.k_cache_extension = self.v_cache_extension = None
        self.k_swap = self.v_swap = None

        # Block manager
        self.cpu_block_manager = self.gpu_block_manager = None

        # Explicit two-state runtime morphing. Prepared variants live only on
        # pinned host memory; the inactive representation is never retained in
        # HBM.
        self.runtime_morphing_enabled = False
        self.runtime_precision_state = (
            "FP16"
            if engine_config.quantized_layer_count == 0
            else "AWQ_MARLIN_W4_16"
            if engine_config.quantized_layer_count == 16
            and engine_config.quantization_backend == "awq_marlin"
            else f"{engine_config.quantization_backend}:{engine_config.quantized_layer_count}"
        )
        self.prepared_fp16_layers = []
        self.prepared_awq_layers = []
        self.runtime_preparation_trace = None
        self.runtime_transition_traces = []
        self._transition_counter = 0
        
    @torch.inference_mode()
    def load_weights(self):
        """
        Load weights and initialize layers
        """
        # Load weights
        self.weight = load_weights(
            self.model_config,
            torch.float16,
            self.engine_config.model_path,
            self.engine_config.use_dummy,
            quantized_layer_count=self.engine_config.quantized_layer_count,
            quantization_backend=self.engine_config.quantization_backend,
            quantized_model_path=self.engine_config.quantized_model_path,
        )

        # Initialize rotary embeddings
        self._init_to_get_rotary()

        # Initialize layers
        decoding_piggyback_stream = torch.cuda.Stream()
        self.pre_layer = LlamaPreLayer(self.model_config, self.weight)
        self.transformer_layers = [
            LlamaTransformerLayer(
                self.model_config,
                self.engine_config,
                self.weight.layers[layer_id],
                decoding_piggyback_stream,
                layer_id
            )
            for layer_id in range(self.model_config.num_layers)
        ]
        self.post_layer = LlamaPostLayer(self.model_config, self.weight)

    @staticmethod
    def _memory_snapshot() -> dict[str, int]:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
        }

    @torch.inference_mode()
    def prepare_runtime_morphing(self, quantized_layer_count: int = 16) -> dict:
        """Prepare both selected-layer states before the base KV pool exists."""
        if self.runtime_morphing_enabled:
            return self.runtime_preparation_trace
        if quantized_layer_count != 16:
            raise ValueError("runtime morphing supports exactly AWQ-W4-16")
        if self.engine_config.quantized_layer_count != 0:
            raise ValueError("runtime morphing must initialize in FP16")
        if self.engine_config.quantization_backend != "awq_marlin":
            raise ValueError("runtime morphing requires the validated AWQ-Marlin backend")
        if not self.engine_config.quantized_model_path:
            raise ValueError("runtime morphing requires quantized_model_path")

        started_ns = time.perf_counter_ns()
        trace = {
            "schema_version": 1,
            "event": "runtime_variant_preparation",
            "started_ns": started_ns,
            "memory_before": self._memory_snapshot(),
            "layers": [],
            "d2h_bytes": 0,
        }
        for layer_id in range(quantized_layer_count):
            layer_started_ns = time.perf_counter_ns()
            fp16_variant = prepare_layer_variant(self.weight.layers[layer_id])
            torch.cuda.synchronize()
            fp16_bytes = layer_variant_bytes(fp16_variant)

            awq_variant, awq_bytes = load_prepared_awq_layer(
                layer_id,
                self.model_config,
                self.engine_config.quantized_model_path,
            )
            torch.cuda.synchronize()
            self.prepared_fp16_layers.append(fp16_variant)
            self.prepared_awq_layers.append(awq_variant)
            trace["d2h_bytes"] += fp16_bytes + awq_bytes
            trace["layers"].append({
                "layer_id": layer_id,
                "elapsed_ns": time.perf_counter_ns() - layer_started_ns,
                "fp16_bytes": fp16_bytes,
                "awq_immutable_bytes": awq_bytes,
                "fp16_layout": layer_variant_layout(fp16_variant),
                "awq_layout": layer_variant_layout(awq_variant),
            })
            gc.collect()
            torch.cuda.empty_cache()

        for variant in (*self.prepared_fp16_layers, *self.prepared_awq_layers):
            for _, tensor in layer_variant_tensor_items(variant):
                if tensor.device.type != "cpu" or not tensor.is_pinned():
                    raise RuntimeError("prepared runtime variant is not pinned host memory")
        if any(layer.quantized for layer in self.weight.layers[:quantized_layer_count]):
            raise RuntimeError("AWQ representation unexpectedly active after preparation")

        self.runtime_morphing_enabled = True
        trace["ended_ns"] = time.perf_counter_ns()
        trace["elapsed_ns"] = trace["ended_ns"] - started_ns
        trace["memory_after"] = self._memory_snapshot()
        self.runtime_preparation_trace = trace
        return trace

    def _install_layer(self, layer_id, replacement):
        self.weight.layers[layer_id] = replacement
        self.transformer_layers[layer_id].weight = replacement

    @torch.inference_mode()
    def _switch_runtime_layers(self, target: str) -> dict:
        variants = (
            self.prepared_awq_layers
            if target == "AWQ_MARLIN_W4_16"
            else self.prepared_fp16_layers
        )
        rollback_variants = (
            self.prepared_fp16_layers
            if target == "AWQ_MARLIN_W4_16"
            else self.prepared_awq_layers
        )
        expected_quantized = target == "AWQ_MARLIN_W4_16"
        switch_started_ns = time.perf_counter_ns()
        changed = []
        rows = []
        total_h2d = 0
        try:
            for layer_id, prepared in enumerate(variants):
                started_ns = time.perf_counter_ns()
                memory_before = self._memory_snapshot()
                replacement = materialize_layer_variant(prepared, self.model_config)
                sync_started_ns = time.perf_counter_ns()
                torch.cuda.synchronize()
                sync_ns = time.perf_counter_ns() - sync_started_ns
                memory_with_both_representations = self._memory_snapshot()
                if replacement.quantized != expected_quantized:
                    raise RuntimeError("materialized layer has the wrong quantization flag")
                outgoing = self.weight.layers[layer_id]
                self._install_layer(layer_id, replacement)
                changed.append(layer_id)
                del outgoing
                h2d_bytes = layer_variant_bytes(prepared)
                total_h2d += h2d_bytes
                rows.append({
                    "layer_id": layer_id,
                    "elapsed_ns": time.perf_counter_ns() - started_ns,
                    "cuda_sync_ns": sync_ns,
                    "h2d_bytes": h2d_bytes,
                    "d2h_bytes": 0,
                    "backend_before": (
                        "FP16" if expected_quantized else "AWQ_MARLIN_W4_16"
                    ),
                    "backend_after": target,
                    "memory_before": memory_before,
                    "memory_with_both_representations": memory_with_both_representations,
                    "memory_after": self._memory_snapshot(),
                })
        except Exception:
            for layer_id in reversed(changed):
                rollback = materialize_layer_variant(
                    rollback_variants[layer_id], self.model_config
                )
                torch.cuda.synchronize()
                outgoing = self.weight.layers[layer_id]
                self._install_layer(layer_id, rollback)
                del outgoing
            gc.collect()
            torch.cuda.empty_cache()
            raise

        cleanup_started_ns = time.perf_counter_ns()
        gc.collect()
        torch.cuda.empty_cache()
        cleanup_ns = time.perf_counter_ns() - cleanup_started_ns
        self.weight.quantized_layer_count = 16 if expected_quantized else 0
        return {
            "target": target,
            "elapsed_ns": time.perf_counter_ns() - switch_started_ns,
            "cleanup_ns": cleanup_ns,
            "h2d_bytes": total_h2d,
            "layers": rows,
        }

    def _read_virtual_block(self, virtual_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        if virtual_id < self.base_num_blocks:
            return self.k_cache[virtual_id], self.v_cache[virtual_id]
        extension_id = virtual_id - self.base_num_blocks
        return (
            self.k_cache_extension[extension_id],
            self.v_cache_extension[extension_id],
        )

    @torch.inference_mode()
    def sample_active_kv(self, max_blocks: int = 8) -> dict:
        """Hash logical sequence blocks, independent of their physical IDs."""
        if max_blocks <= 0 or self.gpu_block_manager is None:
            return {"logical_digest": hashlib.sha256(b"").hexdigest(), "samples": [], "d2h_bytes": 0}
        counts = self.gpu_block_manager.num_seq_allocated_blocks.cpu().tolist()
        logical_blocks = []
        for seq_id, count in enumerate(counts):
            if count:
                virtual_ids = self.gpu_block_manager.block_table[
                    seq_id, :count
                ].cpu().tolist()
                logical_blocks.extend(
                    (seq_id, block_index, int(virtual_id))
                    for block_index, virtual_id in enumerate(virtual_ids)
                )
        logical_blocks = logical_blocks[:max_blocks]
        aggregate = hashlib.sha256()
        samples = []
        d2h_bytes = 0
        for seq_id, block_index, virtual_id in logical_blocks:
            k_block, v_block = self._read_virtual_block(virtual_id)
            k_host = k_block.detach().contiguous().cpu()
            v_host = v_block.detach().contiguous().cpu()
            block_hash = hashlib.sha256()
            block_hash.update(memoryview(k_host.numpy()))
            block_hash.update(memoryview(v_host.numpy()))
            digest = block_hash.hexdigest()
            logical_key = f"{seq_id}:{block_index}"
            aggregate.update(logical_key.encode())
            aggregate.update(digest.encode())
            copied = (
                k_host.numel() * k_host.element_size()
                + v_host.numel() * v_host.element_size()
            )
            d2h_bytes += copied
            samples.append({
                "seq_id": seq_id,
                "sequence_block_index": block_index,
                "virtual_block_id": virtual_id,
                "sha256": digest,
                "bytes": copied,
            })
        return {
            "logical_digest": aggregate.hexdigest(),
            "samples": samples,
            "d2h_bytes": d2h_bytes,
        }

    @torch.inference_mode()
    def _extend_kv_capacity(self, target_num_blocks: int) -> dict:
        if target_num_blocks <= self.num_blocks:
            return {
                "requested_total_blocks": target_num_blocks,
                "actual_total_blocks": self.num_blocks,
                "extension_blocks": self.num_blocks - self.base_num_blocks,
                "elapsed_ns": 0,
            }
        if self.k_cache_extension.shape[0]:
            raise RuntimeError("only one KV extension segment is supported")
        additional = target_num_blocks - self.base_num_blocks
        shape = (additional, *self.k_cache.shape[1:])
        started_ns = time.perf_counter_ns()
        memory_before = self._memory_snapshot()
        torch.cuda.empty_cache()
        new_k = new_v = None
        try:
            new_k = torch.zeros(shape, dtype=self.k_cache.dtype, device=self.k_cache.device)
            new_v = torch.zeros(shape, dtype=self.v_cache.dtype, device=self.v_cache.device)
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            del new_k, new_v
            gc.collect()
            torch.cuda.empty_cache()
            raise RuntimeError(
                f"physical KV extension to {target_num_blocks} blocks does not fit"
            ) from None

        self.k_cache_extension = new_k
        self.v_cache_extension = new_v
        self.gpu_block_manager.extend(additional)
        self.num_blocks = target_num_blocks
        return {
            "requested_total_blocks": target_num_blocks,
            "actual_total_blocks": self.num_blocks,
            "extension_blocks": additional,
            "extension_bytes": additional * self.engine_config.block_size * self.model_config.get_kvslot_size(),
            "elapsed_ns": time.perf_counter_ns() - started_ns,
            "memory_before": memory_before,
            "memory_after": self._memory_snapshot(),
        }

    @torch.inference_mode()
    def _shrink_kv_to_base(self, verify_kv: bool) -> dict:
        allocated = self.gpu_block_manager.num_blocks - self.gpu_block_manager.num_free_blocks
        if allocated > self.base_num_blocks:
            raise RuntimeError(
                f"restoration requires draining to <= {self.base_num_blocks} blocks; {allocated} remain"
            )
        started_ns = time.perf_counter_ns()
        memory_before = self._memory_snapshot()
        integrity_before = self.sample_active_kv() if verify_kv else None
        extension_ids, base_ids = self.gpu_block_manager.plan_compaction_to_prefix(
            self.base_num_blocks
        )
        block_bytes = self.engine_config.block_size * self.model_config.get_kvslot_size()
        for extension_id, base_id in zip(extension_ids.tolist(), base_ids.tolist()):
            local_id = extension_id - self.base_num_blocks
            self.k_cache[base_id].copy_(self.k_cache_extension[local_id])
            self.v_cache[base_id].copy_(self.v_cache_extension[local_id])
        torch.cuda.synchronize()
        self.gpu_block_manager.commit_compaction(extension_ids, base_ids)
        torch.cuda.synchronize()
        integrity_after = self.sample_active_kv() if verify_kv else None
        if verify_kv and integrity_before["logical_digest"] != integrity_after["logical_digest"]:
            raise RuntimeError("logical KV digest changed during extension compaction")

        self.gpu_block_manager.shrink(self.base_num_blocks)
        empty_shape = (0, *self.k_cache.shape[1:])
        old_k = self.k_cache_extension
        old_v = self.v_cache_extension
        self.k_cache_extension = torch.empty(
            empty_shape, dtype=self.k_cache.dtype, device=self.k_cache.device
        )
        self.v_cache_extension = torch.empty(
            empty_shape, dtype=self.v_cache.dtype, device=self.v_cache.device
        )
        self.num_blocks = self.base_num_blocks
        del old_k, old_v
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "actual_total_blocks": self.num_blocks,
            "remapped_blocks": int(extension_ids.numel()),
            "d2d_bytes": int(extension_ids.numel()) * block_bytes,
            "d2h_bytes": (
                (integrity_before or {}).get("d2h_bytes", 0)
                + (integrity_after or {}).get("d2h_bytes", 0)
            ),
            "elapsed_ns": time.perf_counter_ns() - started_ns,
            "memory_before": memory_before,
            "memory_after": self._memory_snapshot(),
            "integrity_before": integrity_before,
            "integrity_after": integrity_after,
        }

    @torch.inference_mode()
    def morph_to_awq_w4_16(
        self,
        target_num_blocks: int,
        *,
        active_request_count: int = 0,
        used_kv_blocks: int = 0,
        verify_kv: bool = False,
    ) -> dict:
        if not self.runtime_morphing_enabled:
            raise RuntimeError("runtime morphing was not enabled at initialization")
        if self.runtime_precision_state == "AWQ_MARLIN_W4_16":
            return {"schema_version": 1, "status": "noop", "precision_after": self.runtime_precision_state}
        self._transition_counter += 1
        started_ns = time.perf_counter_ns()
        trace = {
            "schema_version": 1,
            "transition_id": self._transition_counter,
            "direction": "FP16_TO_AWQ_MARLIN_W4_16",
            "status": "started",
            "started_ns": started_ns,
            "precision_before": "FP16",
            "active_request_count": active_request_count,
            "used_kv_blocks": used_kv_blocks,
            "base_blocks": self.base_num_blocks,
            "scheduler_visible_blocks_before": self.base_num_blocks,
            "memory_before": self._memory_snapshot(),
        }
        self.runtime_precision_state = "MORPHING_TO_AWQ"
        try:
            sync_started_ns = time.perf_counter_ns()
            torch.cuda.synchronize()
            trace["boundary_cuda_sync_ns"] = time.perf_counter_ns() - sync_started_ns
            torch.cuda.reset_peak_memory_stats()
            trace["memory_after_boundary_sync"] = self._memory_snapshot()
            weights = self._switch_runtime_layers("AWQ_MARLIN_W4_16")
            trace["weight_transition"] = weights
            trace["memory_after_weights"] = self._memory_snapshot()
            integrity_before = self.sample_active_kv() if verify_kv else None
            kv_resize = self._extend_kv_capacity(target_num_blocks)
            integrity_after = self.sample_active_kv() if verify_kv else None
            if verify_kv and integrity_before["logical_digest"] != integrity_after["logical_digest"]:
                raise RuntimeError("logical KV digest changed during capacity expansion")
            kv_resize["integrity_before"] = integrity_before
            kv_resize["integrity_after"] = integrity_after
            kv_resize["d2h_bytes"] = (
                (integrity_before or {}).get("d2h_bytes", 0)
                + (integrity_after or {}).get("d2h_bytes", 0)
            )
            trace["kv_resize"] = kv_resize
            self.runtime_precision_state = "AWQ_MARLIN_W4_16"
            trace["status"] = "success"
        except Exception as exc:
            if self.num_blocks != self.base_num_blocks:
                # A newly-created extension has no allocated blocks while the
                # transition boundary is closed, so rollback is lossless.
                self._shrink_kv_to_base(False)
            if self.weight.layers[0].quantized:
                self._switch_runtime_layers("FP16")
            self.runtime_precision_state = "FP16"
            trace["status"] = "failed_rolled_back"
            trace["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            trace["ended_ns"] = time.perf_counter_ns()
            trace["elapsed_ns"] = trace["ended_ns"] - started_ns
            trace["precision_after"] = self.runtime_precision_state
            trace["physical_blocks_after"] = self.num_blocks
            trace["transition_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            trace["memory_after"] = self._memory_snapshot()
            self.runtime_transition_traces.append(trace)
        return trace

    @torch.inference_mode()
    def restore_to_fp16(
        self,
        *,
        active_request_count: int = 0,
        used_kv_blocks: int = 0,
        verify_kv: bool = False,
    ) -> dict:
        if not self.runtime_morphing_enabled:
            raise RuntimeError("runtime morphing was not enabled at initialization")
        if self.runtime_precision_state == "FP16":
            return {"schema_version": 1, "status": "noop", "precision_after": self.runtime_precision_state}
        self._transition_counter += 1
        started_ns = time.perf_counter_ns()
        trace = {
            "schema_version": 1,
            "transition_id": self._transition_counter,
            "direction": "AWQ_MARLIN_W4_16_TO_FP16",
            "status": "started",
            "started_ns": started_ns,
            "precision_before": "AWQ_MARLIN_W4_16",
            "active_request_count": active_request_count,
            "used_kv_blocks": used_kv_blocks,
            "base_blocks": self.base_num_blocks,
            "scheduler_visible_blocks_before": self.num_blocks,
            "memory_before": self._memory_snapshot(),
        }
        self.runtime_precision_state = "RESTORING_TO_FP16"
        try:
            sync_started_ns = time.perf_counter_ns()
            torch.cuda.synchronize()
            trace["boundary_cuda_sync_ns"] = time.perf_counter_ns() - sync_started_ns
            torch.cuda.reset_peak_memory_stats()
            trace["memory_after_boundary_sync"] = self._memory_snapshot()
            trace["kv_resize"] = self._shrink_kv_to_base(verify_kv)
            trace["memory_after_kv_shrink"] = self._memory_snapshot()
            trace["weight_transition"] = self._switch_runtime_layers("FP16")
            self.runtime_precision_state = "FP16"
            trace["status"] = "success"
        except Exception as exc:
            self.runtime_precision_state = "FAILED"
            trace["status"] = "failed"
            trace["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            trace["ended_ns"] = time.perf_counter_ns()
            trace["elapsed_ns"] = trace["ended_ns"] - started_ns
            trace["precision_after"] = self.runtime_precision_state
            trace["physical_blocks_after"] = self.num_blocks
            trace["transition_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            trace["memory_after"] = self._memory_snapshot()
            self.runtime_transition_traces.append(trace)
        return trace

    @torch.inference_mode()
    def verify_active_runtime_variant(self, target: str) -> dict:
        """Copy active immutable tensors back and compare exact bytes to staging."""
        variants = (
            self.prepared_awq_layers
            if target == "AWQ_MARLIN_W4_16"
            else self.prepared_fp16_layers
        )
        expected_hash = hashlib.sha256()
        observed_hash = hashlib.sha256()
        compared_bytes = 0
        layouts_match = True
        for active, prepared in zip(self.weight.layers[:16], variants):
            expected_items = dict(layer_variant_tensor_items(prepared))
            active_items = {
                name: tensor
                for name, tensor in layer_variant_tensor_items(active)
                if not name.endswith((".workspace", ".g_idx", ".g_idx_sort_indices"))
            }
            if expected_items.keys() != active_items.keys():
                layouts_match = False
                continue
            for name, expected in expected_items.items():
                observed = active_items[name].detach().contiguous().cpu()
                if observed.shape != expected.shape or observed.dtype != expected.dtype:
                    layouts_match = False
                    continue
                expected_hash.update(name.encode())
                expected_hash.update(memoryview(expected.contiguous().numpy()))
                observed_hash.update(name.encode())
                observed_hash.update(memoryview(observed.numpy()))
                compared_bytes += observed.numel() * observed.element_size()
        return {
            "target": target,
            "layouts_match": layouts_match,
            "expected_sha256": expected_hash.hexdigest(),
            "observed_sha256": observed_hash.hexdigest(),
            "byte_exact": layouts_match and expected_hash.digest() == observed_hash.digest(),
            "compared_bytes": compared_bytes,
            "active_layouts": [
                layer_variant_layout(layer) for layer in self.weight.layers[:16]
            ],
        }

    @torch.inference_mode()
    def profile_num_blocks(self) -> int:
        """
        Profiler the number of GPU blocks

        We run a forged prefill batch with the maximum number of tokens and
        sequences, record the peak memory usage, and infer the number of blocks
        that can be allocated.
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Synthesis a prefill batch
        num_tokens = self.engine_config.max_tokens_in_batch
        batch_size = self.engine_config.max_batch_size
        input_lens = [num_tokens // batch_size] * batch_size
        input_lens[-1] += num_tokens % batch_size
        input_ids = [
            [0 for _ in range(input_len)]
            for input_len in input_lens
        ]
        seq_ids = list(range(batch_size))
        self.k_cache = self.v_cache = None # pylint: disable=attribute-defined-outside-init
        _ = self.forward(input_ids, seq_ids, [], ignore_kvcache=True)
        torch.cuda.synchronize()

        # peak_memory = torch.cuda.max_memory_allocated()
        # total_memory = torch.cuda.get_device_properties(0).total_memory
        free_memory, total_memory = torch.cuda.mem_get_info()
        peak_memory = total_memory - free_memory
        useable_memory = total_memory*self.engine_config.gpu_mem_utilization
        print(f"[Model.profile] GPU total memory: {total_memory/GB:.2f} GB, runtime peak memory: {peak_memory/GB:.2f} GB")
        if useable_memory < peak_memory:
            raise RuntimeError(f"Peak memory {peak_memory/GB:.2f} GB exceeds usable memory {useable_memory/GB:.2f} GB ({total_memory/GB:.2f} GB * {self.engine_config.gpu_mem_utilization})")
        block_size_bytes = self.engine_config.block_size * self.model_config.get_kvslot_size()
        num_gpu_blocks = math.floor((useable_memory - peak_memory) / block_size_bytes)

        torch.cuda.empty_cache()
        return num_gpu_blocks
    
    @torch.inference_mode()
    def init_kvcache_and_swap(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.base_num_blocks = num_blocks

        # Initialize KV cache
        kvcache_shape = (
            self.num_blocks,
            self.model_config.num_layers,
            self.model_config.num_kv_heads,
            self.engine_config.block_size,
            self.model_config.head_dim
        )
        # Here we use torch.zeros instead of torch.empty, since that torch.empty
        # has the possibility to contain NaNs, which will cause the model to output NaNs.
        self.k_cache = torch.zeros(kvcache_shape, dtype=torch.float16, device="cuda")
        self.v_cache = torch.zeros(kvcache_shape, dtype=torch.float16, device="cuda")
        empty_extension_shape = (0, *kvcache_shape[1:])
        self.k_cache_extension = torch.empty(
            empty_extension_shape, dtype=torch.float16, device="cuda"
        )
        self.v_cache_extension = torch.empty(
            empty_extension_shape, dtype=torch.float16, device="cuda"
        )

        # Initialize KV swap space
        kvswap_shape = (
            self.engine_config.num_cpu_blocks,
            self.model_config.num_layers,
            self.model_config.num_kv_heads,
            self.engine_config.block_size,
            self.model_config.head_dim
        )
        self.k_swap = torch.zeros(kvswap_shape, dtype=torch.float16, device="cpu")
        self.v_swap = torch.zeros(kvswap_shape, dtype=torch.float16, device="cpu")

        # Initialize block manager
        self.gpu_block_manager = BlockManager(
            "GPU",
            self.num_blocks,
            self.engine_config.max_seqs_in_block_table,
            self.engine_config.max_blocks_per_seq,
            self.engine_config.block_size
        )
        self.cpu_block_manager = BlockManager(
            "CPU",
            self.engine_config.num_cpu_blocks,
            self.engine_config.max_seqs_in_block_table,
            self.engine_config.max_blocks_per_seq,
            self.engine_config.block_size
        )

    def _init_to_get_rotary(self):
        # Keep the cache layout expected by rotary_emb.py: one cosine/sine
        # value per pair of dimensions. The inverse-frequency calculation is
        # shared with the CPU-testable model config and follows Transformers'
        # Llama-3.1 implementation exactly.
        inv_freq = self.model_config.get_rope_inv_freq(device="cuda")
        max_seq_len = int(self.model_config.max_position_embeddings)
        positions = torch.arange(max_seq_len + 128, device="cuda", dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self._cos_cached = torch.cos(freqs).to(torch.float16)
        self._sin_cached = torch.sin(freqs).to(torch.float16)

    @torch.inference_mode()
    def _forward(
        self,
        input_ids: torch.Tensor,    # [total_token_num]
        infer_state: LlamaInferState,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """
        Run a forward pass of the LlamaModel.
        """
        input_embds = self.pre_layer.forward(input_ids)
        residual_buf = torch.zeros_like(input_embds)
        for layer in self.transformer_layers:
            input_embds = layer.forward(
                input_embds,
                residual_buf,
                self.k_cache,
                self.v_cache,
                self.k_cache_extension,
                self.v_cache_extension,
                self.base_num_blocks,
                self.gpu_block_manager.block_table if not infer_state.ignore_kvcache else None,
                infer_state,
            )
        input_embds += residual_buf
        return self.post_layer.forward(
            input_embds,
            infer_state,
            return_logits=return_logits,
        )
    
    @torch.inference_mode()
    def forward(
        self,
        input_ids_list: list[list[int]], # [batch_size, *]
        seq_ids_list: list[int],     # [batch_size]
        decoding_seq_lens_list: list[int], # [num_decoding_seqs]
        ignore_kvcache: bool = False,   # Skip actions related to kv cache, useful when profiling the number of kv blocks
        return_logits: bool = False,
    ) -> list[int] | torch.Tensor:
        """
        Run a forward pass of the LlamaModel.

        This function is a wrapper of the `_forward` function. It prepares the infer_state
        and calls the `_forward` function.

        This function is intended to be called by the server.
        """

        num_prefill_seqs = len(input_ids_list) - len(decoding_seq_lens_list)
        flattened_input_ids = list(itertools.chain(*input_ids_list))
        seq_lengths_list = [len(seq) for seq in input_ids_list[:num_prefill_seqs]] + decoding_seq_lens_list

        seq_ids = torch.tensor(seq_ids_list, dtype=torch.int32, device="cuda")
        seq_lengths = torch.tensor(seq_lengths_list, dtype=torch.int32, device="cuda")

        batch_size = len(input_ids_list)
        num_tokens = len(flattened_input_ids)

        prefill_seq_lens_list = seq_lengths_list[:num_prefill_seqs]
        prefill_seq_lens = torch.tensor(prefill_seq_lens_list, dtype=torch.int32, device="cuda")
        prefill_start_locs = torch.cumsum(prefill_seq_lens, dim=0, dtype=torch.int32) - prefill_seq_lens
        max_prefill_len = max(prefill_seq_lens_list) if prefill_seq_lens_list else 0

        decoding_seq_lens = torch.tensor(decoding_seq_lens_list, dtype=torch.int32, device="cuda")
        max_decoding_len = max(decoding_seq_lens_list) if decoding_seq_lens_list else 0

        position_indices = torch.cat((
            torch.concat([
                torch.arange(
                    0,
                    prefill_seq_len,
                    device="cuda",
                    dtype=torch.int32
                )
                for prefill_seq_len in prefill_seq_lens_list
            ]) if prefill_seq_lens_list else torch.empty(0, device="cuda", dtype=torch.int32),
            decoding_seq_lens - 1
        ), dim=0)

        if not ignore_kvcache:
            self.gpu_block_manager.allocate_blocks_for_seqs(
                seq_ids,
                seq_lengths
            )

        # Select the seq_block_size
        #
        # Here we use a simple heuristic:
        #
        # In paged attention phase 1, the grid shape is (num_decoding_seqs, num_kv_heads, cdiv(max_decoding_len, seq_block_size))
        # and among these blocks, num_kv_heads * sum(cdiv(decoding_seq_lens, seq_block_size)) blocks are useful.
        # Thus we set seq_block_size to be the largest integer that satisfies
        #      num_kv_heads * sum(cdiv(decoding_seq_lens, seq_block_size)) >= 1024
        # to fully utilize the GPU. Here 1024 is a magic number (since most high-end
        # GPUs have ~128 SMs, so ~512 SMSPs. Since the decoding-stage attention
        # is mostly a memory-bound operation, I think 1024 is a reasonable number.)
        #
        # In practice, we use `decoding_seq_lens_sum/seq_block_size` to approximate
        # sum(cdiv(decoding_seq_lens, seq_block_size))

        seq_block_size = 2048
        decoding_seq_lens_sum = sum(decoding_seq_lens_list)
        while self.model_config.num_kv_heads*(decoding_seq_lens_sum/seq_block_size) < 1024 and seq_block_size//2 >= 64 and \
            max_decoding_len / (seq_block_size//2) <= 128:
            seq_block_size //= 2

        infer_state = LlamaInferState(
            batch_size = batch_size,
            num_tokens = num_tokens,

            seq_ids = seq_ids,
            softmax_scale = self.model_config.head_dim ** -0.5,

            num_prefill_seqs = num_prefill_seqs,
            num_prefill_tokens = num_tokens - (batch_size - num_prefill_seqs),
            prefill_seq_start_locs = prefill_start_locs,
            prefill_seq_start_locs_with_end = torch.cat([
                prefill_start_locs,
                torch.tensor([num_tokens], dtype=torch.int32, device="cuda")
            ]),
            prefill_seq_lens = prefill_seq_lens,
            max_prefill_len = max_prefill_len,

            num_decoding_seqs = batch_size - num_prefill_seqs,
            decoding_seq_lens = decoding_seq_lens,
            max_decoding_len = max_decoding_len,

            seq_block_size = seq_block_size,
            num_seq_blocks = (max_decoding_len + seq_block_size-1) // seq_block_size,

            position_cos = self._cos_cached[position_indices],
            position_sin = self._sin_cached[position_indices],

            ignore_kvcache = ignore_kvcache
        )

        output = self._forward(
            torch.tensor(flattened_input_ids, dtype=torch.int32, device="cuda"),
            infer_state,
            return_logits=return_logits,
        )
        return output if return_logits else output.tolist()

    def _swap(
        self,
        seq_ids_list: list[int],
        is_swap_in: bool
    ):
        src_block_manager = self.cpu_block_manager if is_swap_in else self.gpu_block_manager
        dst_block_manager = self.gpu_block_manager if is_swap_in else self.cpu_block_manager
        seq_ids = torch.tensor(seq_ids_list, dtype=torch.int32, device="cuda")
        seq_lengths = src_block_manager.get_num_allocated_blocks(seq_ids) * self.engine_config.block_size
        src_block_ids = src_block_manager.gather_allocated_blocks_and_free(seq_ids)
        dst_block_ids = dst_block_manager.allocate_blocks_for_seqs(seq_ids, seq_lengths)
        source_ids = src_block_ids.tolist()
        target_ids = dst_block_ids.tolist()
        gpu_ids = target_ids if is_swap_in else source_ids
        base_pairs = []
        extension_pairs = []
        for source_id, target_id, gpu_id in zip(source_ids, target_ids, gpu_ids):
            pair = (source_id, target_id)
            (extension_pairs if gpu_id >= self.base_num_blocks else base_pairs).append(pair)

        def copy_pairs(pairs, k_segment, v_segment, extension=False):
            if not pairs:
                return
            local_sources = [source for source, _ in pairs]
            local_targets = [target for _, target in pairs]
            if extension:
                if is_swap_in:
                    local_targets = [value - self.base_num_blocks for value in local_targets]
                else:
                    local_sources = [value - self.base_num_blocks for value in local_sources]
            swiftllm_c.swap_blocks(
                local_sources,
                local_targets,
                is_swap_in,
                k_segment,
                v_segment,
                self.k_swap,
                self.v_swap,
            )

        copy_pairs(base_pairs, self.k_cache, self.v_cache)
        copy_pairs(
            extension_pairs,
            self.k_cache_extension,
            self.v_cache_extension,
            extension=True,
        )
        
    @torch.inference_mode()
    def swap_in_seqs(
        self,
        seq_ids_list: list[int]
    ):
        """
        Swap in (move blocks from CPU to GPU) the specified sequences.
        """
        self._swap(seq_ids_list, True)
    
    @torch.inference_mode()
    def swap_out_seqs(
        self,
        seq_ids_list: list[int]
    ):
        """
        Swap out (move blocks from GPU to CPU) the specified sequences.
        """
        self._swap(seq_ids_list, False)

    @torch.inference_mode()
    def free_seqs_resources(self, seq_ids_list: list[int]):
        """
        Free the resources of the specified sequences.
        """
        seq_ids = torch.tensor(seq_ids_list, dtype=torch.int32, device="cuda")
        self.gpu_block_manager.free_blocks_for_seqs(seq_ids)
        self.cpu_block_manager.free_blocks_for_seqs(seq_ids)
