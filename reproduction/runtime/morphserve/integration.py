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
        self.poisoned = False

    def _executor_active(self):
        active = getattr(self.executor, "active_layers", None)
        return None if active is None else list(active)

    def step(self, scheduler, timestamp_s, metrics):
        if self.poisoned or bool(getattr(self.executor, "poisoned", False)):
            raise RuntimeError("coordinator/executor is poisoned; serving is disabled")
        queues_before = queue_state(scheduler)
        sample = self.monitor.observe(timestamp_s, **metrics)
        command = self.controller.update(sample)
        delta = command["layer_delta"]
        selected = []
        success = True
        error = None
        active_before = list(self.active_layers)

        def append_error(message):
            nonlocal error
            error = message if error is None else f"{error}; {message}"

        def rollback_pressure():
            active = self._executor_active()
            rollback = list(reversed(selected)) if active is None else [layer for layer in reversed(selected) if layer in active]
            rollback_ok = True
            if rollback:
                try:
                    rollback_ok = bool(self.executor.restore_fp16(rollback))
                except Exception as exc:
                    rollback_ok = False
                    append_error(f"rollback {type(exc).__name__}: {exc}")
            active_after = self._executor_active()
            if active_after is not None and active_after != active_before:
                rollback_ok = False
            if rollback_ok:
                self.active_layers = active_before
                self.controller.quantized_layers -= delta
            else:
                self.poisoned = True
                self.active_layers = active_after if active_after is not None else active_before + selected
                self.controller.quantized_layers = len(self.active_layers)
                append_error("pressure rollback incomplete; coordinator poisoned")

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
                    rollback_pressure()
            elif delta < 0:
                selected = list(reversed(self.active_layers[-abs(delta):]))
                recover = getattr(self.executor, "recover_fp16", None)
                if recover is None:
                    raise RuntimeError("executor lacks atomic recover_fp16")
                success = bool(recover(selected))
                if success:
                    remove = set(selected)
                    self.active_layers = [layer for layer in self.active_layers if layer not in remove]
                else:
                    active_after = self._executor_active()
                    if bool(getattr(self.executor, "poisoned", False)) or (active_after is not None and active_after != active_before):
                        self.poisoned = True
                        self.active_layers = active_after if active_after is not None else active_before
                        self.controller.quantized_layers = len(self.active_layers)
                        append_error("atomic recovery failed closed; coordinator poisoned")
                    else:
                        self.controller.quantized_layers -= delta
        except Exception as exc:
            success = False
            append_error(f"{type(exc).__name__}: {exc}")
            if delta > 0 and selected:
                rollback_pressure()
            elif delta < 0:
                self.poisoned = True
                active_after = self._executor_active()
                self.active_layers = active_after if active_after is not None else active_before
                self.controller.quantized_layers = len(self.active_layers)
                append_error("recovery raised; coordinator poisoned")
            else:
                self.controller.quantized_layers -= delta
        queues_after = queue_state(scheduler)
        if queues_after != queues_before:
            self.poisoned = True
            raise RuntimeError("controller action changed FCFS queue order")
        actual = self._executor_active()
        if actual is not None and actual != self.active_layers:
            self.poisoned = True
            raise RuntimeError("coordinator/executor active-layer sets diverged")
        if self.controller.quantized_layers != len(self.active_layers):
            self.poisoned = True
            raise RuntimeError("controller/executor layer counts diverged")
        event = {
            "command": command,
            "selected_layers": selected,
            "success": success,
            "error": error,
            "poisoned": self.poisoned,
            "queues_before": queues_before,
            "queues_after": queues_after,
            "active_layers": list(self.active_layers),
        }
        self.events.append(event)
        return event
