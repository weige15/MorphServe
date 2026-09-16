import math
import unittest

from morphserve.controller import MODE_CONFIGS, SIGNALS, MorphingController, ServingMonitor


def metrics(kv=0.5, queue_delay=0.01):
    return {
        "gpu_memory_usage": 0.7,
        "kv_usage": kv,
        "queue_depth": 2,
        "queue_delay_s": queue_delay,
        "throughput_tokens_s": 100,
        "ttft_s": 0.2,
        "tpot_s": 0.02,
    }


def sample(index, kv, queue_delay):
    values = metrics(kv, queue_delay)
    return {"timestamp_s": index * 0.05, "raw": values, "smoothed": values}


class ServingMonitorTests(unittest.TestCase):
    def test_collects_and_smooths_every_paper_signal(self):
        monitor = ServingMonitor()
        first = monitor.observe(0.0, **metrics(kv=0.4, queue_delay=0.02))
        second = monitor.observe(0.05, **metrics(kv=0.8, queue_delay=0.10))

        self.assertEqual(set(first["raw"]), set(SIGNALS))
        self.assertEqual(second["smoothed"]["kv_usage"], 0.5)
        self.assertEqual(second["smoothed"]["queue_delay_s"], 0.04)
        self.assertEqual(len(monitor.history), 2)

    def test_rejects_invalid_metrics(self):
        monitor = ServingMonitor()
        invalid = metrics()
        invalid["ttft_s"] = math.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            monitor.observe(0.0, **invalid)
        with self.assertRaisesRegex(ValueError, "exactly"):
            monitor.observe(0.0, kv_usage=0.5)


class MorphingControllerTests(unittest.TestCase):
    def test_persistent_pressure_and_recovery_coordinate_kv(self):
        controller = MorphingController("default")
        first = controller.update(sample(0, 0.9, 0.11))
        second = controller.update(sample(1, 0.9, 0.11))
        third = controller.update(sample(2, 0.9, 0.11))
        self.assertEqual([first["layer_delta"], second["layer_delta"], third["layer_delta"]], [0, 0, 2])
        self.assertTrue(third["expand_kv"])
        for index in range(4):
            self.assertEqual(controller.update(sample(3 + index, 0.5, 0.01))["layer_delta"], 0)
        recovery = controller.update(sample(7, 0.5, 0.01))
        self.assertEqual(recovery["layer_delta"], -2)
        self.assertTrue(recovery["shrink_kv_before_restore"])

    def test_modes_obey_action_size_and_maximum(self):
        for mode, config in MODE_CONFIGS.items():
            controller = MorphingController(mode)
            deltas = []
            for index in range(60):
                command = controller.update(sample(index, 0.9, 0.11))
                if command["layer_delta"]:
                    deltas.append(command["layer_delta"])
            self.assertTrue(all(delta == config.layers_per_action for delta in deltas))
            self.assertEqual(controller.quantized_layers, config.max_quantized_layers)

    def test_oscillation_does_not_accumulate_persistence(self):
        controller = MorphingController("performance")
        for index in range(20):
            if index % 2:
                command = controller.update(sample(index, 0.84, 0.099))
            else:
                command = controller.update(sample(index, 0.86, 0.101))
            self.assertEqual(command["layer_delta"], 0)
        self.assertEqual(controller.quantized_layers, 0)

    def test_unknown_mode_and_invalid_sample_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "unknown mode"):
            MorphingController("paper-magic")
        controller = MorphingController()
        bad = sample(0, 0.5, 0.01)
        bad["smoothed"]["tpot_s"] = math.inf
        with self.assertRaisesRegex(ValueError, "invalid"):
            controller.update(bad)


if __name__ == "__main__":
    unittest.main()
