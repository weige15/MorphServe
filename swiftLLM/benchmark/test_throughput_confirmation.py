"""CPU checks for v11 controller parity and paired inference."""

from __future__ import annotations

import math
import unittest

from .analyze_throughput_confirmation import inference_for
from .instrumented_release_intent_controller import InstrumentedReleaseIntentController
from .release_intent_controller import ReleaseIntentController
from .replay_throughput_controller_parity import compare_decisions, fake_trace


class ThroughputConfirmationTests(unittest.TestCase):
    @staticmethod
    def sample(elapsed: float, used: int, waiting: int, preemptions: int = 0) -> dict:
        return {
            "elapsed_s": elapsed,
            "scheduler_used_kv_blocks": used,
            "physical_used_kv_blocks": used,
            "waiting_q_depth": waiting,
            "preemption_count": preemptions,
        }

    def test_instrumentation_preserves_entry_and_release_decisions(self) -> None:
        original = ReleaseIntentController(0)
        instrumented = InstrumentedReleaseIntentController(0)
        actions = []
        for index in range(140):
            elapsed = index * 0.25
            if elapsed <= 3.0:
                sample = self.sample(elapsed, 1680, 4)
            elif elapsed < 20.0:
                sample = self.sample(elapsed, 2200, 0)
            else:
                sample = self.sample(elapsed, 1400, 0)
            left = original.evaluate(sample)
            right = instrumented.evaluate(sample)
            compare_decisions(left, right, f"synthetic:{index}")
            action = left["requested_action"]
            if action:
                timestamp_ns = int(elapsed * 1e9)
                left_id = original.request_transition(action, timestamp_ns)
                right_id = instrumented.request_transition(action, timestamp_ns)
                trace = fake_trace(timestamp_ns)
                original.complete_transition(left_id, trace, timestamp_ns)
                instrumented.complete_transition(right_id, trace, timestamp_ns)
                actions.append(action)
        self.assertEqual(
            actions,
            ["FP16_TO_AWQ_MARLIN_W4_16", "AWQ_MARLIN_W4_16_TO_FP16"],
        )
        self.assertGreater(instrumented.timing_totals_ns["evaluate_wall"], 0)
        self.assertGreater(instrumented.timing_totals_ns["window_wall"], 0)

    def test_paired_log_inference_uses_frozen_97p5_one_sided_bound(self) -> None:
        rows = [
            {"log_throughput_ratio": math.log(value)}
            for value in (0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998)
        ]
        result = inference_for("heldout_one_cycle", rows)
        self.assertEqual(result["pair_count"], 8)
        self.assertEqual(result["one_sided_alpha"], 0.025)
        self.assertGreater(result["one_sided_lower_confidence_bound"], 0.98)
        self.assertTrue(result["noninferior"])


if __name__ == "__main__":
    unittest.main()
