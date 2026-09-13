"""CPU-only checks for causal crossover signal calculations."""

from __future__ import annotations

import unittest

from .analyze_crossover import evaluate_signals, release_time


class CrossoverSignalTests(unittest.TestCase):
    @staticmethod
    def telemetry(high_pressure: bool) -> list[dict]:
        rows = []
        preemptions = 0
        for index in range(161):
            elapsed = index * 0.25
            pressure = high_pressure and 5.0 <= elapsed < 15.0
            if pressure and elapsed in (7.0, 8.0):
                preemptions += 1
            used = 1720 if pressure else 400
            rows.append(
                {
                    "elapsed_s": elapsed,
                    "waiting_q_depth": 5 if pressure else 0,
                    "logical_kv_utilization": used / 1768,
                    "num_decoding_gpu_blocks": used,
                    "num_gpu_blocks": 1768,
                    "preemption_count": preemptions,
                }
            )
        return rows

    def test_persistent_queue_kv_signals_reject_low_pressure(self) -> None:
        signals = evaluate_signals(self.telemetry(False), 40.0)
        self.assertFalse(signals["sustained_compound_pressure"]["entry_fired"])
        self.assertFalse(signals["capacity_margin_queue_integral"]["entry_fired"])
        self.assertFalse(signals["preemption_confirmed_pressure"]["entry_fired"])

    def test_high_pressure_fires_and_release_obeys_hysteresis(self) -> None:
        telemetry = self.telemetry(True)
        signals = evaluate_signals(telemetry, 40.0)
        self.assertTrue(signals["sustained_compound_pressure"]["entry_fired"])
        self.assertTrue(signals["capacity_margin_queue_integral"]["entry_fired"])
        entry = float(signals["sustained_compound_pressure"]["first_entry_s"])
        release = release_time(telemetry, 40.0, "sustained_compound_pressure", entry)
        self.assertIsNotNone(release)
        self.assertGreaterEqual(float(release), entry + 10.0)
        self.assertGreaterEqual(signals["sustained_compound_pressure"]["entry_condition_episode_count"], 1)


if __name__ == "__main__":
    unittest.main()
