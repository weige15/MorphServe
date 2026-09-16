"""Transactional GPU executor over the tested candidate reconstruction primitives."""

from __future__ import annotations

import torch

from .autoawq_adapter import AsyncLayerCopier, install_autoawq_layer


class RealMorphingExecutor:
    def __init__(self, model, extension, backups, packed, fail_expand_calls=()):
        self.model = model
        self.extension = extension
        self.backups = backups
        self.packed = packed
        self.fail_expand_calls = set(fail_expand_calls)
        self.expand_calls = 0
        self.active_layers = []
        self.quant_objects = {}
        self.kv_groups = []
        self.log = []
        self.poisoned = False
        self.model.async_layer_transfers = True
        self.copier = AsyncLayerCopier(model, extension)
        self.fp16_objects = {layer: model.transformer_layers[layer] for layer in backups}
        self.prepared_quant = {}
        for layer, source in packed.items():
            current = model.transformer_layers[layer]
            views = self.copier.views(layer, source["tensor_map"])
            wrapper, modules = install_autoawq_layer(model, layer, views, source["tensor_map"])
            self.prepared_quant[layer] = (wrapper, modules, views)
            model.transformer_layers[layer] = current
        sizes = {value["size"] for value in backups.values()}
        if len(sizes) != 1:
            raise ValueError("executor requires equal FP16 layer strides")
        self.layer_stride_bytes = sizes.pop()
        self.model.org_layer_param_size = self.layer_stride_bytes // self.model.k_cache.element_size()

    def _sync_model_state(self):
        active = set(self.active_layers)
        self.model.layer_quant_list = list(self.active_layers)
        self.model.is_layer_quant_list = [index in active for index in range(self.model.model_config.num_layers)]
        if len(self.kv_groups) < 2:
            self.model.explicit_kv_regions = False
        else:
            first = self.kv_groups[0]["k"].data_ptr()
            stride = self.layer_stride_bytes
            self.model.explicit_kv_regions = any(
                group["k"].data_ptr() != first - index * stride
                for index, group in enumerate(self.kv_groups)
            )

    def _snapshot_kv_state(self):
        manager = self.model.gpu_block_manager
        return {
            "num_blocks": int(manager.num_blocks),
            "num_free": manager.num_free_blocks.clone() if isinstance(manager.num_free_blocks, torch.Tensor) else int(manager.num_free_blocks),
            "is_free": manager.is_block_free.clone(),
            "k_cache": list(self.model.k_cache_new),
            "v_cache": list(self.model.v_cache_new),
            "groups": list(self.kv_groups),
            "group_size": self.model.kv_cache_new_block_size,
        }

    def _restore_kv_state(self, snapshot):
        manager = self.model.gpu_block_manager
        manager.num_blocks = snapshot["num_blocks"]
        manager.num_free_blocks = snapshot["num_free"]
        manager.is_block_free = snapshot["is_free"]
        self.model.k_cache_new[:] = snapshot["k_cache"]
        self.model.v_cache_new[:] = snapshot["v_cache"]
        self.kv_groups[:] = snapshot["groups"]
        self.model.kv_cache_new_block_size = snapshot["group_size"]
        self._sync_model_state()

    @torch.inference_mode()
    def morph_to_w4(self, layers):
        if self.poisoned or len(set(layers)) != len(layers) or any(
            layer in self.active_layers or layer not in self.prepared_quant
            for layer in layers
        ):
            return False
        committed = []
        try:
            for layer in layers:
                source = self.packed[layer]
                self.copier.enqueue(
                    layer, source["buffer"], source["size"], source["tensor_map"]
                )
                wrapper, modules, tensors = self.prepared_quant[layer]
                self.model.transformer_layers[layer] = wrapper
                self.quant_objects[layer] = (wrapper, modules, tensors)
                self.active_layers.append(layer)
                committed.append(layer)
                self.log.append(("morph", layer))
        except Exception as exc:
            rollback_error = None
            for layer in reversed(committed):
                try:
                    source = self.backups[layer]
                    self.copier.enqueue(
                        layer, source["buffer"], source["size"], source["tensor_map"]
                    )
                    self.model.transformer_layers[layer] = self.fp16_objects[layer]
                    self.quant_objects.pop(layer, None)
                    self.active_layers.remove(layer)
                    self.log.append(("morph_rollback", layer))
                except Exception as rollback_exc:
                    rollback_error = rollback_exc
                    self.poisoned = True
                    self.log.append(("morph_rollback_failed", layer, str(rollback_exc)))
                    break
            self.log.append(("morph_failed", str(exc)))
            if rollback_error is not None:
                self.log.append(("executor_poisoned", "morph rollback incomplete"))
            self._sync_model_state()
            return False
        self._sync_model_state()
        return True

    @torch.inference_mode()
    def expand_kv(self, layers):
        if self.poisoned:
            return False
        manager = self.model.gpu_block_manager
        snapshot = self._snapshot_kv_state()
        try:
            for layer in layers:
                self.expand_calls += 1
                if self.expand_calls in self.fail_expand_calls:
                    raise RuntimeError(f"injected KV expansion failure {self.expand_calls}")
                self.copier.wait_current(layer)
                k_cache, v_cache = self.extension.acquire_new_kvcache(layer)
                if self.kv_groups and k_cache.shape[0] != self.kv_groups[0]["k"].shape[0]:
                    raise RuntimeError("reclaimed groups have unequal block counts")
                self.model.k_cache_new.append(k_cache)
                self.model.v_cache_new.append(v_cache)
                self.kv_groups.append({"layer": layer, "k": k_cache, "v": v_cache})
                manager.is_block_free = torch.cat((
                    manager.is_block_free,
                    torch.ones(k_cache.shape[0], dtype=torch.bool, device="cuda"),
                ))
                manager.num_blocks += k_cache.shape[0]
                manager.num_free_blocks += k_cache.shape[0]
                self.model.kv_cache_new_block_size = k_cache.shape[0]
                self.log.append(("expand", layer, int(k_cache.shape[0])))
            self._sync_model_state()
            return True
        except Exception as exc:
            rollback_ready = torch.cuda.Event()
            rollback_ready.record()
            self.model.last_forward_event = rollback_ready
            manager.num_blocks = snapshot["num_blocks"]
            manager.num_free_blocks = snapshot["num_free"]
            manager.is_block_free = snapshot["is_free"]
            self.model.k_cache_new[:] = snapshot["k_cache"]
            self.model.v_cache_new[:] = snapshot["v_cache"]
            self.kv_groups[:] = snapshot["groups"]
            self.model.kv_cache_new_block_size = snapshot["group_size"]
            self.log.append(("expand_failed", str(exc)))
            self._sync_model_state()
            return False

    @torch.inference_mode()
    def shrink_kv_before_restore(self, layers):
        if self.poisoned:
            return False
        manager = self.model.gpu_block_manager
        expected = [group["layer"] for group in reversed(self.kv_groups[-len(layers):])]
        if list(layers) != expected:
            return False
        group_size = self.model.kv_cache_new_block_size
        original = manager.num_blocks_org
        for offset, _layer in enumerate(layers):
            group_index = len(self.kv_groups) - 1 - offset
            start = original + group_index * group_size
            end = start + group_size
            if not bool(manager.is_block_free[start:end].all()):
                return False
        for _layer in layers:
            group = self.kv_groups.pop()
            self.model.k_cache_new.pop()
            self.model.v_cache_new.pop()
            manager.is_block_free = manager.is_block_free[:-group_size]
            manager.num_blocks -= group_size
            manager.num_free_blocks -= group_size
            self.log.append(("shrink", group["layer"], group_size))
        if not self.kv_groups:
            self.model.kv_cache_new_block_size = 0
        self._sync_model_state()
        return True

    @torch.inference_mode()
    def restore_fp16(self, layers):
        grouped = {group["layer"] for group in self.kv_groups}
        if self.poisoned or len(set(layers)) != len(layers) or any(
            layer in grouped or layer not in self.active_layers
            for layer in layers
        ):
            return False
        active_before = list(self.active_layers)
        restored = []
        try:
            for layer in layers:
                source = self.backups[layer]
                self.copier.enqueue(
                    layer, source["buffer"], source["size"], source["tensor_map"]
                )
                self.model.transformer_layers[layer] = self.fp16_objects[layer]
                self.quant_objects.pop(layer, None)
                self.active_layers.remove(layer)
                restored.append(layer)
                self.log.append(("restore", layer))
        except Exception as exc:
            still_fp16 = set(restored)
            rollback_error = None
            for layer in reversed(restored):
                try:
                    source = self.packed[layer]
                    self.copier.enqueue(
                        layer, source["buffer"], source["size"], source["tensor_map"]
                    )
                    wrapper, modules, tensors = self.prepared_quant[layer]
                    self.model.transformer_layers[layer] = wrapper
                    self.quant_objects[layer] = (wrapper, modules, tensors)
                    still_fp16.remove(layer)
                    self.log.append(("restore_rollback", layer))
                except Exception as rollback_exc:
                    rollback_error = rollback_exc
                    self.poisoned = True
                    self.log.append(("restore_rollback_failed", layer, str(rollback_exc)))
                    break
            self.active_layers = [layer for layer in active_before if layer not in still_fp16]
            self.log.append(("restore_failed", str(exc)))
            if rollback_error is not None:
                self.log.append(("executor_poisoned", "restore rollback incomplete"))
            self._sync_model_state()
            return False
        self._sync_model_state()
        return True

    @torch.inference_mode()
    def recover_fp16(self, layers):
        """Atomically shrink reclaimed KV groups and restore their W4 layers."""
        if self.poisoned:
            return False
        snapshot = self._snapshot_kv_state()
        try:
            if not self.shrink_kv_before_restore(layers):
                return False
            if self.restore_fp16(layers):
                return True
        except Exception as exc:
            self.poisoned = True
            self.log.append(("recovery_failed", str(exc)))
        if self.poisoned:
            self.log.append(("recovery_snapshot_not_reattached", "executor state is uncertain"))
            return False
        self._restore_kv_state(snapshot)
        self.log.append(("recovery_rollback", list(layers)))
        return False
