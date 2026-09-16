"""Controller-to-executor coordination that leaves FCFS queues untouched."""

from __future__ import annotations

from dataclasses import dataclass

from .controller import MorphingController, ServingMonitor


@dataclass(frozen=True)
class QueueState:
    waiting: tuple
    running: tuple
    swapped: tuple


def queue_state(scheduler) -> QueueState:
    def ids(queue):
        return tuple(getattr(item, "request_id", item) for item in queue)
    return QueueState(ids(scheduler.waiting_q), ids(scheduler.running_q), ids(scheduler.swapped_q))


class AdaptiveCoordinator:
    def __init__(self, profile_order, executor, mode="default", monitor=None):
        if len(set(profile_order)) != len(profile_order):
            raise ValueError("profile_order must contain unique layers")
        self.profile_order = list(profile_order)
        self.executor = executor
        self.monitor = monitor or ServingMonitor()
        self.controller = MorphingController(mode)
        self.active_layers = []
        self.events = []

    def step(self, scheduler, timestamp_s, metrics):
        queues_before = queue_state(scheduler)
        sample = self.monitor.observe(timestamp_s, **metrics)
        command = self.controller.update(sample)
        delta = command["layer_delta"]
        selected = []
        success = True
        error = None
        try:
            if delta > 0:
                selected = [layer for layer in self.profile_order if layer not in self.active_layers][:delta]
                if len(selected) != delta:
                    raise RuntimeError("profile exhausted before controller maximum")
                success = bool(self.executor.morph_to_w4(selected))
                if success:
                    success = bool(self.executor.expand_kv(selected))
                if success:
                    self.active_layers.extend(selected)
                else:
                    active = getattr(self.executor, "active_layers", None)
                    rollback = list(reversed(selected)) if active is None else [layer for layer in reversed(selected) if layer in active]
                    if rollback:
                        self.executor.restore_fp16(rollback)
                    self.controller.quantized_layers -= delta
            elif delta < 0:
                selected = list(reversed(self.active_layers[-abs(delta):]))
                success = bool(self.executor.shrink_kv_before_restore(selected))
                if success:
                    success = bool(self.executor.restore_fp16(selected))
                if success:
                    remove = set(selected)
                    self.active_layers = [layer for layer in self.active_layers if layer not in remove]
                else:
                    self.controller.quantized_layers -= delta
        except Exception as exc:  # fail closed and preserve controller state
            success = False
            error = f"{type(exc).__name__}: {exc}"
            if delta > 0 and selected:
                try:
                    active = getattr(self.executor, "active_layers", None)
                    rollback = list(reversed(selected)) if active is None else [layer for layer in reversed(selected) if layer in active]
                    if rollback:
                        self.executor.restore_fp16(rollback)
                except Exception as rollback_exc:
                    error += f"; rollback {type(rollback_exc).__name__}: {rollback_exc}"
            self.controller.quantized_layers -= delta
        queues_after = queue_state(scheduler)
        if queues_after != queues_before:
            raise RuntimeError("controller action changed FCFS queue order")
        if self.controller.quantized_layers != len(self.active_layers):
            raise RuntimeError("controller/executor layer state diverged")
        event = {
            "command": command,
            "selected_layers": selected,
            "success": success,
            "error": error,
            "queues_before": queues_before,
            "queues_after": queues_after,
            "active_layers": list(self.active_layers),
        }
        self.events.append(event)
        return event
