"""CPU checks for the frozen v9 causal controller state machine."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from .analyze_crossover import evaluate_signals
from .closed_loop_controller import (
    ENTRY_REQUESTED,
    STABLE_AWQ,
    STABLE_FP16,
    SustainedCompoundController,
)


class ClosedLoopControllerTests(unittest.TestCase):
    @staticmethod
    def sample(elapsed: float, *, used: int, waiting: int, preemptions: int = 0) -> dict:
        return {
            "elapsed_s": elapsed,
            "waiting_q_depth": waiting,
            "scheduler_used_kv_blocks": used,
            "physical_used_kv_blocks": used,
            "preemption_count": preemptions,
        }

    def test_low_pressure_never_enters(self) -> None:
        controller = SustainedCompoundController(0)
        for index in range(81):
            decision = controller.evaluate(
                self.sample(index * 0.25, used=1200, waiting=3)
            )
            self.assertIsNone(decision["requested_action"])
        self.assertEqual(controller.state, STABLE_FP16)

    def test_exact_entry_latches_one_transition(self) -> None:
        controller = SustainedCompoundController(0)
        decision = None
        for index in range(13):
            decision = controller.evaluate(
                self.sample(index * 0.25, used=1680, waiting=4)
            )
        assert decision is not None
        self.assertEqual(decision["entry_kv_ge095_fraction"], 1.0)
        self.assertEqual(decision["entry_waiting_ge4_fraction"], 1.0)
        self.assertEqual(decision["requested_action"], "FP16_TO_AWQ_MARLIN_W4_16")
        transition_id = controller.request_transition(decision["requested_action"], 3_000_000_000)
        self.assertEqual(controller.state, ENTRY_REQUESTED)
        self.assertIsNone(
            controller.evaluate(self.sample(3.25, used=1680, waiting=4))["requested_action"]
        )
        with self.assertRaises(RuntimeError):
            controller.request_transition("FP16_TO_AWQ_MARLIN_W4_16", 3_250_000_000)
        controller.mark_api_started(transition_id, 3_010_000_000)
        controller.complete_transition(
            transition_id,
            {
                "status": "success",
                "started_ns": 3_020_000_000,
                "ended_ns": 3_680_000_000,
                "elapsed_ns": 660_000_000,
            },
            3_690_000_000,
        )
        self.assertEqual(controller.state, STABLE_AWQ)
        self.assertAlmostEqual(
            controller.transitions[0]["pending_or_drain_duration_s"], 0.02
        )

    def test_release_uses_full_ten_seconds_and_fixed_reference(self) -> None:
        controller = SustainedCompoundController(0)
        for index in range(13):
            decision = controller.evaluate(
                self.sample(index * 0.25, used=1680, waiting=4)
            )
        transition_id = controller.request_transition(
            str(decision["requested_action"]), 3_000_000_000
        )
        controller.mark_api_started(transition_id, 3_000_000_000)
        controller.complete_transition(
            transition_id,
            {
                "status": "success",
                "started_ns": 3_000_000_000,
                "ended_ns": 3_100_000_000,
                "elapsed_ns": 100_000_000,
            },
            3_100_000_000,
        )
        releases = []
        for index in range(13, 54):
            now = index * 0.25
            current = controller.evaluate(self.sample(now, used=1237, waiting=0))
            if current["requested_action"]:
                releases.append((now, current))
                break
        self.assertTrue(releases)
        now, release = releases[0]
        self.assertGreaterEqual(now, 13.0)
        self.assertEqual(release["release_new_preemptions"], 0)
        self.assertGreaterEqual(release["release_fp16_equiv_le070_fraction"], 0.90)
        self.assertEqual(release["requested_action"], "AWQ_MARLIN_W4_16_TO_FP16")

    def test_v7_replay_matches_authoritative_first_entry(self) -> None:
        root = Path(__file__).resolve().parents[2] / "benchmark-results/fp16-awq-crossover-v7/runs"
        for run_id, should_fire in (
            ("cross-fp16_0-scale-5p75-rep0", False),
            ("cross-fp16_0-scale-4p25-rep0", True),
        ):
            run_dir = root / run_id
            telemetry = [
                json.loads(line)
                for line in (run_dir / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            expected = evaluate_signals(
                telemetry, float(metadata["measurement_duration_s"])
            )["sustained_compound_pressure"]["first_entry_s"]
            controller = SustainedCompoundController(0)
            observed = None
            for row in telemetry:
                decision = controller.evaluate(
                    self.sample(
                        float(row["elapsed_s"]),
                        used=int(row["num_decoding_gpu_blocks"]),
                        waiting=int(row["waiting_q_depth"]),
                        preemptions=int(row.get("preemption_count", 0)),
                    )
                )
                if decision["requested_action"] is not None:
                    observed = float(row["elapsed_s"])
                    controller.request_transition(
                        str(decision["requested_action"]), int(observed * 1_000_000_000)
                    )
                    break
            self.assertEqual(observed is not None, should_fire)
            self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
