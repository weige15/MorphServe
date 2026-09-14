"""Checks for the frozen release-side v10 workload and run-plan bytes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


class ReleaseWorkloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = (
            Path(__file__).resolve().parents[2]
            / "benchmark-results"
            / "release-side-runtime-v10"
        )
        cls.metadata = json.loads(
            (cls.root / "input" / "workload_metadata.json").read_text(encoding="utf-8")
        )

    @staticmethod
    def rows(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_frozen_outputs_match_hashes_and_shapes(self) -> None:
        expected_counts = {
            "low_only": 64,
            "heldout_one_cycle": 192,
            "heldout_two_cycle": 320,
        }
        for name, expected in expected_counts.items():
            record = self.metadata["outputs"][name]
            path = Path(record["path"])
            rows = self.rows(path)
            self.assertEqual(len(rows), expected)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"])
            self.assertEqual([row["sequence"] for row in rows], list(range(expected)))
            self.assertEqual(len({row["request_id"] for row in rows}), expected)
            self.assertTrue(all(row["prompt_token_count"] == 1024 for row in rows))
            self.assertTrue(all(row["requested_output_token_count"] == 512 for row in rows))

    def test_alternating_phases_and_continuing_final_low_are_exact(self) -> None:
        self.assertTrue(self.metadata["no_artificial_idle_gap"])
        expected = {
            "heldout_one_cycle": ["low", "high", "low"],
            "heldout_two_cycle": ["low", "high", "low", "high", "low"],
        }
        for name, regimes in expected.items():
            boundaries = self.metadata["phase_boundaries"][name]
            self.assertEqual([row["regime"] for row in boundaries], regimes)
            self.assertTrue(all(row["request_count"] == 64 for row in boundaries))
            self.assertEqual(
                boundaries[-1]["last_arrival_offset_s"]
                - boundaries[-1]["start_offset_s"],
                90.5625,
            )

    def test_content_offset_differs_from_v9_and_plan_has_16_fresh_cells(self) -> None:
        selection = self.metadata["content_selection"]
        self.assertEqual(selection["source_sequence_start"], 0)
        self.assertEqual(selection["source_sequence_end"], 63)
        self.assertEqual(selection["new_questions_relative_to_v9_42_105"], 42)
        plan = json.loads((self.root / "run-plan.json").read_text(encoding="utf-8"))
        self.assertEqual(plan["run_count"], 16)
        cells = {
            (row["workload_class"], row["condition"], row["repeat"])
            for row in plan["runs"]
        }
        self.assertEqual(len(cells), 16)
        self.assertTrue(all(row["run_id"].startswith("v10-") for row in plan["runs"]))


if __name__ == "__main__":
    unittest.main()
