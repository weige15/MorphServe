"""Small checks for release-side timeline/gate helpers."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from .analyze_release_side import catch_up_end, phase_percentiles, write_csv


class ReleaseAnalysisTests(unittest.TestCase):
    def test_empty_no_go_evidence_table_keeps_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.csv"
            write_csv(path, [], ("run_id", "lifecycle_complete"))
            self.assertEqual(path.read_text(encoding="utf-8"), "run_id,lifecycle_complete\n")

    def test_phase_percentiles_use_zero_to_one_hundred_api(self) -> None:
        result = phase_percentiles([float(value) for value in range(100)])
        self.assertEqual(result["p50"], 49.5)
        self.assertEqual(result["p95"], 94.05)
        self.assertEqual(result["p99"], 98.01)

    def test_catch_up_requires_three_continuous_zero_wait_seconds(self) -> None:
        telemetry = [
            {"elapsed_s": 0.0, "waiting_q_depth": 2},
            {"elapsed_s": 1.0, "waiting_q_depth": 0},
            {"elapsed_s": 2.0, "waiting_q_depth": 1},
            {"elapsed_s": 3.0, "waiting_q_depth": 0},
            {"elapsed_s": 6.5, "waiting_q_depth": 0},
        ]
        self.assertEqual(catch_up_end(telemetry, 7.0, 0.5), 6.0)


if __name__ == "__main__":
    unittest.main()
