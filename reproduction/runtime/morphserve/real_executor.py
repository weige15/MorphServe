"""Transactional GPU executor over the tested candidate reconstruction primitives."""

from __future__ import annotations

import gc

import torch

from .autoawq_adapter import AsyncLayerCopier, install_autoawq_layer, release_fp16_layer, restore_fp16_layer


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
        self.copier = AsyncLayerCopier(model, extension)
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

    @torch.inference_mode()
    def morph_to_w4(self, layers):
        for layer in layers:
            if layer in self.active_layers:
                raise RuntimeError(f"layer already W4: {layer}")
            source = self.packed[layer]
            tensors, _ = self.copier.enqueue(
                layer, source["buffer"], source["size"], source["tensor_map"]
            )
            release_fp16_layer(self.model, layer)
            wrapper, modules = install_autoawq_layer(
                self.model, layer, tensors, source["tensor_map"]
            )
            self.quant_objects[layer] = (wrapper, modules, tensors)
            self.active_layers.append(layer)
            self.log.append(("morph", layer))
        self._sync_model_state()
        return True

    @torch.inference_mode()
    def expand_kv(self, layers):
        manager = self.model.gpu_block_manager
        snapshot = {
            "num_blocks": manager.num_blocks,
            "num_free": manager.num_free_blocks,
            "is_free": manager.is_block_free,
            "k_len": len(self.model.k_cache_new),
            "v_len": len(self.model.v_cache_new),
            "group_len": len(self.kv_groups),
            "group_size": self.model.kv_cache_new_block_size,
        }
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
            manager.num_blocks = snapshot["num_blocks"]
            manager.num_free_blocks = snapshot["num_free"]
            manager.is_block_free = snapshot["is_free"]
            del self.model.k_cache_new[snapshot["k_len"]:]
            del self.model.v_cache_new[snapshot["v_len"]:]
            del self.kv_groups[snapshot["group_len"]:]
            self.model.kv_cache_new_block_size = snapshot["group_size"]
            self.log.append(("expand_failed", str(exc)))
            self._sync_model_state()
            return False

    @torch.inference_mode()
    def shrink_kv_before_restore(self, layers):
        manager = self.model.gpu_block_manager
        expected = [group["layer"] for group in reversed(self.kv_groups[-len(layers):])]
        if list(layers) != expected:
            return False
        group_size = self.model.kv_cache_new_block_size
        original = manager.num_blocks_org
        for _layer in layers:
            group_index = len(self.kv_groups) - 1
            start = original + group_index * group_size
            end = start + group_size
            if not bool(manager.is_block_free[start:end].all()):
                return False
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
        if any(layer in grouped for layer in layers):
            return False
        for layer in layers:
            self.model.transformer_layers[layer] = None
            objects = self.quant_objects.pop(layer, None)
            if objects is not None:
                del objects
            gc.collect()
            source = self.backups[layer]
            tensors, _ = self.copier.enqueue(
                layer, source["buffer"], source["size"], source["tensor_map"]
            )
            restore_fp16_layer(
                self.model, layer, tensors, source["tensor_map"]
            )
            self.active_layers.remove(layer)
            self.log.append(("restore", layer))
        self._sync_model_state()
        return True
