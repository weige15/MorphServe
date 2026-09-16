import unittest
from types import SimpleNamespace

from morphserve.controller import ServingMonitor
from morphserve.integration import AdaptiveCoordinator, queue_state


def metrics(kv, delay):
    return {
        "gpu_memory_usage": 0.7,
        "kv_usage": kv,
        "queue_depth": 2,
        "queue_delay_s": delay,
        "throughput_tokens_s": 100,
        "ttft_s": 0.2,
        "tpot_s": 0.02,
    }


class FakeExecutor:
    def __init__(self, expand=True, shrink=True, raise_expand=False):
        self.expand = expand
        self.shrink = shrink
        self.raise_expand = raise_expand
        self.calls = []

    def morph_to_w4(self, layers):
        self.calls.append(("morph", list(layers)))
        return True

    def expand_kv(self, layers):
        self.calls.append(("expand", list(layers)))
        if self.raise_expand:
            raise RuntimeError("synthetic expansion failure")
        return self.expand

    def shrink_kv_before_restore(self, layers):
        self.calls.append(("shrink", list(layers)))
        return self.shrink

    def restore_fp16(self, layers):
        self.calls.append(("restore", list(layers)))
        return True


def scheduler():
    return SimpleNamespace(waiting_q=[3, 4], running_q=[1, 2], swapped_q=[5])


def trigger_pressure(coordinator, sched, start=0):
    events = [coordinator.step(sched, (start + index) * 0.05, metrics(0.9, 0.11)) for index in range(3)]
    return events[-1]


class CoordinatorTests(unittest.TestCase):
    def test_pressure_uses_profile_order_and_preserves_fcfs(self):
        executor = FakeExecutor()
        sched = scheduler()
        before = queue_state(sched)
        coordinator = AdaptiveCoordinator([25, 24, 26, 27], executor, monitor=ServingMonitor(alpha=1))

        event = trigger_pressure(coordinator, sched)

        self.assertTrue(event["success"])
        self.assertEqual(event["selected_layers"], [25, 24])
        self.assertEqual(executor.calls, [("morph", [25, 24]), ("expand", [25, 24])])
        self.assertEqual(queue_state(sched), before)
        self.assertEqual(coordinator.active_layers, [25, 24])

    def test_failed_expand_rolls_back_in_reverse_order(self):
        executor = FakeExecutor(expand=False)
        coordinator = AdaptiveCoordinator([25, 24, 26], executor, monitor=ServingMonitor(alpha=1))

        event = trigger_pressure(coordinator, scheduler())

        self.assertFalse(event["success"])
        self.assertEqual(executor.calls, [
            ("morph", [25, 24]),
            ("expand", [25, 24]),
            ("restore", [24, 25]),
        ])
        self.assertEqual(coordinator.active_layers, [])
        self.assertEqual(coordinator.controller.quantized_layers, 0)

    def test_expand_exception_fails_closed_and_rolls_back(self):
        executor = FakeExecutor(raise_expand=True)
        coordinator = AdaptiveCoordinator([25, 24, 26], executor, monitor=ServingMonitor(alpha=1))

        event = trigger_pressure(coordinator, scheduler())

        self.assertFalse(event["success"])
        self.assertIn("synthetic expansion failure", event["error"])
        self.assertEqual(executor.calls[-1], ("restore", [24, 25]))
        self.assertEqual(coordinator.active_layers, [])
        self.assertEqual(coordinator.controller.quantized_layers, 0)

    def test_recovery_shrinks_before_restore_and_defers_if_occupied(self):
        executor = FakeExecutor(shrink=False)
        coordinator = AdaptiveCoordinator([25, 24, 26], executor, monitor=ServingMonitor(alpha=1))
        sched = scheduler()
        trigger_pressure(coordinator, sched)
        executor.calls.clear()
        event = None
        for index in range(5):
            event = coordinator.step(sched, 1 + index * 0.05, metrics(0.5, 0.01))

        self.assertFalse(event["success"])
        self.assertEqual(executor.calls, [("shrink", [24, 25])])
        self.assertEqual(coordinator.active_layers, [25, 24])
        self.assertEqual(coordinator.controller.quantized_layers, 2)

    def test_successful_recovery_restores_after_shrink(self):
        executor = FakeExecutor()
        coordinator = AdaptiveCoordinator([25, 24, 26], executor, monitor=ServingMonitor(alpha=1))
        sched = scheduler()
        trigger_pressure(coordinator, sched)
        executor.calls.clear()
        for index in range(5):
            event = coordinator.step(sched, 1 + index * 0.05, metrics(0.5, 0.01))

        self.assertTrue(event["success"])
        self.assertEqual(executor.calls, [("shrink", [24, 25]), ("restore", [24, 25])])
        self.assertEqual(coordinator.active_layers, [])


if __name__ == "__main__":
    unittest.main()
