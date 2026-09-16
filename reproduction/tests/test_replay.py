import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from morphserve.replay import RequestSpec, percentile, run_replay, summarize, write_artifacts


class ReplayTests(unittest.TestCase):
    def test_arrivals_are_independent_and_all_records_accounted(self):
        specs = [RequestSpec(str(i), i * 0.01, f"prompt-{i}", 3, f"ref-{i}") for i in range(3)]

        async def submit(spec, emit):
            await asyncio.sleep(0.04)
            for token in range(spec.output_len):
                emit(100 * int(spec.request_id) + token, {"precision": "FP16"})
                await asyncio.sleep(0.005)
            return {"preemptions": 0, "kv_capacity": 10, "kv_occupancy": 3}

        records = asyncio.run(run_replay(specs, submit, 1.0))

        self.assertEqual([record["request_id"] for record in records], ["0", "1", "2"])
        self.assertLess(records[-1]["actual_submit_s"] - records[0]["actual_submit_s"], 0.035)
        self.assertTrue(all(record["token_count"] == 3 and record["error"] is None for record in records))
        self.assertTrue(all(len(record["tpot_intervals_s"]) == 2 for record in records))
        self.assertEqual(summarize(records)["total_emitted_tokens"], 9)

    def test_timeout_and_error_remain_in_output(self):
        specs = [RequestSpec("ok", 0, "a", 1), RequestSpec("timeout", 0, "b", 1), RequestSpec("error", 0, "c", 1)]

        async def submit(spec, emit):
            if spec.request_id == "timeout":
                await asyncio.sleep(0.1)
            elif spec.request_id == "error":
                raise RuntimeError("synthetic")
            else:
                emit(7)

        records = asyncio.run(run_replay(specs, submit, 0.02))
        summary = summarize(records)
        self.assertEqual(len(records), 3)
        self.assertEqual(summary["completed_count"], 1)
        self.assertEqual(summary["timeout_count"], 1)
        self.assertEqual(summary["error_count"], 2)
        self.assertEqual(records[1]["error"], "TimeoutError")
        self.assertIn("synthetic", records[2]["error"])

    def test_artifacts_regenerate_summary(self):
        specs = [RequestSpec("x", 0, "prompt", 2)]
        async def submit(_spec, emit):
            emit(1); await asyncio.sleep(0.001); emit(2)
            return {"queued_time_s": 0.0, "precision_changes": []}
        records = asyncio.run(run_replay(specs, submit, 1.0))
        with tempfile.TemporaryDirectory() as directory:
            jsonl = Path(directory) / "raw.jsonl"
            summary_path = Path(directory) / "summary.json"
            write_artifacts(records, jsonl, summary_path)
            reread = [json.loads(line) for line in jsonl.read_text().splitlines()]
            saved_summary = json.loads(summary_path.read_text())
        self.assertEqual(saved_summary, summarize(reread))

    def test_percentile_definition_and_invalid_trace(self):
        self.assertEqual(percentile([1, 2, 3, 4], 0.5), 2.5)
        with self.assertRaisesRegex(ValueError, "monotonic"):
            asyncio.run(run_replay([RequestSpec("a", 1, "", 0), RequestSpec("b", 0, "", 0)], lambda *_: None, 1))
        with self.assertRaisesRegex(ValueError, "unique"):
            asyncio.run(run_replay([RequestSpec("a", 0, "", 0), RequestSpec("a", 1, "", 0)], lambda *_: None, 1))


if __name__ == "__main__":
    unittest.main()
