"""CPU-only tests for arrival scheduling and metric formulas."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from .analyze_static_frontier import bootstrap_mean_ci, longest_common_prefix, token_agreement
from .metrics import derive_request_metrics
from .prepare_static_frontier_workloads import derive_rows
from .summarize import summarize_run
from .workload import build_schedule


class WorkloadTests(unittest.TestCase):
    def test_fixed_schedule_is_independent_of_completion(self) -> None:
        schedule = build_schedule(2.0, "fixed", request_count=4, seed=99)
        self.assertEqual([item.sequence for item in schedule], [0, 1, 2, 3])
        self.assertEqual([item.planned_offset_s for item in schedule], [0.0, 0.5, 1.0, 1.5])

    def test_poisson_schedule_is_seeded_and_not_completion_driven(self) -> None:
        first = build_schedule(5.0, "poisson", request_count=5, seed=7)
        second = build_schedule(5.0, "poisson", request_count=5, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first[0].planned_offset_s, 0.0)
        self.assertTrue(all(b.planned_offset_s > a.planned_offset_s for a, b in zip(first, first[1:])))

    def test_duration_schedule_has_only_offsets_before_duration(self) -> None:
        schedule = build_schedule(2.0, "fixed", duration_s=1.1)
        self.assertEqual([item.planned_offset_s for item in schedule], [0.0, 0.5, 1.0])


class StaticFrontierWorkloadTests(unittest.TestCase):
    def test_dense_trace_scaling_preserves_content_and_reindexes(self) -> None:
        source = [
            {
                "sequence": i,
                "request_id": f"r{i}",
                "planned_arrival_offset_s": float(i * 2),
                "source_trace_timestamp_s": 100 + i,
                "prompt": f"p{i}",
            }
            for i in range(5)
        ]
        rows = derive_rows(source, start=1, count=3, time_scale=4.0, kind="calibration")
        self.assertEqual([row["sequence"] for row in rows], [0, 1, 2])
        self.assertEqual([row["source_sequence"] for row in rows], [1, 2, 3])
        self.assertEqual([row["planned_arrival_offset_s"] for row in rows], [0.0, 8.0, 16.0])
        self.assertEqual([row["prompt"] for row in rows], ["p1", "p2", "p3"])
        self.assertAlmostEqual(rows[0]["frontier_nominal_offered_rps"], 2 / 16)


class StaticFrontierAnalysisTests(unittest.TestCase):
    def test_paired_bootstrap_and_output_agreement_helpers(self) -> None:
        self.assertEqual(bootstrap_mean_ci([2.0, 2.0], repeats=100, seed=7), (2.0, 2.0))
        self.assertEqual(longest_common_prefix([1, 2, 3], [1, 2, 4]), 2)
        self.assertAlmostEqual(token_agreement([1, 2, 3], [1, 4, 3]), 2 / 3)
        self.assertAlmostEqual(token_agreement([1, 2], [1, 2, 3]), 2 / 3)


class MetricTests(unittest.TestCase):
    def test_request_formulas(self) -> None:
        row = {
            "arrival_time_ns": 1_000,
            "scheduler_eligible_time_ns": 2_000,
            "first_prefill_time_ns": 5_000,
            "first_output_token_time_ns": 6_000,
            "first_stream_token_received_time_ns": 7_000,
            "completion_time_ns": 16_000,
            "stream_completion_time_ns": 17_000,
            "output_token_count": 3,
        }
        metrics = derive_request_metrics(row)
        self.assertAlmostEqual(metrics["ttft_s"], 6e-6)
        self.assertAlmostEqual(metrics["engine_ttft_s"], 5e-6)
        self.assertAlmostEqual(metrics["queueing_delay_s"], 3e-6)
        self.assertAlmostEqual(metrics["tpot_s"], 5e-6)
        self.assertAlmostEqual(metrics["engine_tpot_s"], 5e-6)

    def test_one_token_tpot_is_zero(self) -> None:
        row = {
            "arrival_time_ns": 0,
            "first_stream_token_received_time_ns": 10,
            "output_token_count": 1,
        }
        self.assertEqual(derive_request_metrics(row)["tpot_s"], 0.0)


class SummaryTests(unittest.TestCase):
    def test_summary_can_be_regenerated_from_saved_raw_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "metadata.json").write_text(json.dumps({
                "schema_version": 1,
                "run_id": "test",
                "target_rps": 1.0,
                "schedule_window_duration_s": 2.0,
                "measurement_start_time_ns": 0,
                "measurement_end_time_ns": 20_000_000_000,
                "measurement_duration_s": 20.0,
            }), encoding="utf-8")
            request = {
                "status": "completed",
                "arrival_time_ns": 0,
                "scheduler_eligible_time_ns": 1_000_000,
                "first_prefill_time_ns": 2_000_000,
                "first_output_token_time_ns": 3_000_000,
                "first_stream_token_received_time_ns": 4_000_000,
                "completion_time_ns": 13_000_000,
                "stream_completion_time_ns": 14_000_000,
                "output_token_count": 2,
            }
            request2 = dict(request)
            for field in (
                "arrival_time_ns",
                "scheduler_eligible_time_ns",
                "first_prefill_time_ns",
                "first_output_token_time_ns",
                "first_stream_token_received_time_ns",
                "completion_time_ns",
                "stream_completion_time_ns",
            ):
                request2[field] += 2_000_000_000
            (run_dir / "requests.jsonl").write_text(
                json.dumps(request) + "\n" + json.dumps(request2) + "\n",
                encoding="utf-8",
            )
            telemetry = {
                "waiting_q_depth": 2,
                "running_q_count": 1,
                "logical_kv_utilization": 0.5,
                "num_gpu_blocks": 3880,
                "swap_in_count": 0,
                "swap_out_count": 0,
                "gpu_memory_used_bytes": None,
                "gpu_memory_free_bytes": None,
                "gpu_memory_total_bytes": None,
            }
            (run_dir / "telemetry.jsonl").write_text(json.dumps(telemetry) + "\n", encoding="utf-8")
            summary = summarize_run(run_dir)
            self.assertEqual(summary["completed_request_count"], 2)
            self.assertAlmostEqual(summary["actual_interarrival_rate_rps"], 0.5)
            self.assertEqual(summary["peak_waiting_queue_depth"], 2)
            self.assertEqual(summary["num_gpu_blocks_observed"], [3880])
            self.assertAlmostEqual(summary["ttft_s"]["mean"], 0.004)


if __name__ == "__main__":
    unittest.main()
