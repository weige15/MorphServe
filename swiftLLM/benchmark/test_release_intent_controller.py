"""CPU boundary checks for the frozen v10 release-intent controller."""

from __future__ import annotations

import unittest

from .release_intent_controller import ReleaseIntentController


class ReleaseIntentControllerTests(unittest.TestCase):
    @staticmethod
    def sample(
        elapsed: float, *, used: int, waiting: int, preemptions: int = 0, physical: int = 0
    ) -> dict:
        return {
            "elapsed_s": elapsed,
            "waiting_q_depth": waiting,
            "scheduler_used_kv_blocks": used,
            "physical_used_kv_blocks": physical,
            "preemption_count": preemptions,
        }

    @staticmethod
    def complete_entry(controller: ReleaseIntentController, at_s: float = 3.0) -> None:
        transition_id = controller.request_transition(
            "FP16_TO_AWQ_MARLIN_W4_16", int(at_s * 1e9)
        )
        controller.mark_api_started(transition_id, int(at_s * 1e9))
        controller.complete_transition(
            transition_id,
            {
                "status": "success",
                "started_ns": int(at_s * 1e9),
                "ended_ns": int((at_s + 0.1) * 1e9),
                "elapsed_ns": int(0.1 * 1e9),
            },
            int((at_s + 0.1) * 1e9),
        )

    def test_entry_rule_is_unchanged(self) -> None:
        controller = ReleaseIntentController(0)
        decision = None
        for index in range(13):
            decision = controller.evaluate(
                self.sample(index * 0.25, used=1680, waiting=4)
            )
        assert decision is not None
        self.assertEqual(decision["entry_kv_ge095_fraction"], 1.0)
        self.assertEqual(decision["entry_waiting_ge4_fraction"], 1.0)
        self.assertEqual(decision["requested_action"], "FP16_TO_AWQ_MARLIN_W4_16")

    def test_quiet_queue_without_pressure_drop_does_not_release(self) -> None:
        controller = ReleaseIntentController(0)
        self.complete_entry(controller)
        decision = None
        for index in range(13, 94):
            decision = controller.evaluate(
                self.sample(index * 0.25, used=2100, waiting=0)
            )
        assert decision is not None
        self.assertTrue(decision["release_history_complete"])
        self.assertGreater(decision["release_recent_to_prior_utilization_ratio"], 0.80)
        self.assertIsNone(decision["requested_action"])

    def test_pressure_drop_releases_even_when_physical_blocks_exceed_base(self) -> None:
        controller = ReleaseIntentController(0)
        self.complete_entry(controller)
        for index in range(13, 73):
            controller.evaluate(self.sample(index * 0.25, used=2200, waiting=0))
        decision = None
        for index in range(73, 94):
            decision = controller.evaluate(
                self.sample(index * 0.25, used=1700, waiting=0, physical=1844)
            )
            if decision["requested_action"]:
                break
        assert decision is not None
        self.assertEqual(decision["requested_action"], "AWQ_MARLIN_W4_16_TO_FP16")
        self.assertGreater(1844, 1759)

    def test_recent_preemption_blocks_release(self) -> None:
        controller = ReleaseIntentController(0)
        self.complete_entry(controller)
        for index in range(13, 73):
            controller.evaluate(self.sample(index * 0.25, used=2200, waiting=0))
        decision = None
        for index in range(73, 90):
            decision = controller.evaluate(
                self.sample(index * 0.25, used=1400, waiting=0, preemptions=1)
            )
        assert decision is not None
        self.assertEqual(decision["release_recent_preemptions"], 1)
        self.assertIsNone(decision["requested_action"])


if __name__ == "__main__":
    unittest.main()
