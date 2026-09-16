import json
import tempfile
import unittest
from pathlib import Path

from morphserve.trace_analysis import analyze_cuda_overlap_trace


def event(name, category, start, duration, stream=None, byte_count=None):
    args = {}
    if stream is not None:
        args["stream"] = stream
    if byte_count is not None:
        args["bytes"] = byte_count
    return {"ph": "X", "name": name, "cat": category, "ts": start, "dur": duration, "args": args}


class TraceAnalysisTests(unittest.TestCase):
    def analyze(self, events):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps({"traceEvents": events}))
            return analyze_cuda_overlap_trace(path, {"W4": 100, "FP16": 400})

    def test_requires_size_matched_htod_and_actual_kernel_intersection(self):
        events = []
        for offset, precision, byte_count in ((0, "W4", 100), (1000, "FP16", 400)):
            events.extend([
                event(f"morphserve_overlap_{precision}", "cpu_op", offset, 500),
                event("Memcpy HtoD (Pinned -> Device)", "gpu_memcpy", offset + 100, 250, stream=9, byte_count=byte_count),
                event("pre_layer_gemm_kernel", "kernel", offset + 200, 100, stream=7),
            ])

        result = self.analyze(events)

        self.assertTrue(result["passed"])
        self.assertGreater(result["phases"]["W4"]["intersection_us"], 0)
        self.assertNotEqual(result["phases"]["FP16"]["copy_stream"], result["phases"]["FP16"]["kernel_stream"])

    def test_rejects_enclosing_interval_without_kernel_copy_overlap(self):
        events = [
            event("morphserve_overlap_W4", "cpu_op", 0, 500),
            event("Memcpy HtoD", "gpu_memcpy", 100, 100, stream=9, byte_count=100),
            event("kernel_after_copy", "kernel", 250, 100, stream=7),
            event("morphserve_overlap_FP16", "cpu_op", 1000, 500),
            event("Memcpy HtoD", "gpu_memcpy", 1100, 100, stream=9, byte_count=400),
            event("kernel_after_copy", "kernel", 1250, 100, stream=7),
        ]

        result = self.analyze(events)

        self.assertFalse(result["passed"])
        self.assertIn("did not overlap", result["phases"]["W4"]["error"])

    def test_rejects_missing_copy_or_kernel_stream_metadata(self):
        for missing in ("copy", "kernel"):
            with self.subTest(missing=missing):
                events = []
                for offset, precision, byte_count in ((0, "W4", 100), (1000, "FP16", 400)):
                    events.extend([
                        event(f"morphserve_overlap_{precision}", "cpu_op", offset, 500),
                        event("Memcpy HtoD", "gpu_memcpy", offset + 100, 200, stream=None if missing == "copy" else 9, byte_count=byte_count),
                        event("kernel", "kernel", offset + 150, 50, stream=None if missing == "kernel" else 7),
                    ])

                result = self.analyze(events)

                self.assertFalse(result["passed"])
                self.assertTrue(all(not row["passed"] for row in result["phases"].values()))

    def test_rejects_wrong_size_and_same_stream_kernel(self):
        events = [
            event("morphserve_overlap_W4", "cpu_op", 0, 500),
            event("Memcpy HtoD", "gpu_memcpy", 100, 200, stream=9, byte_count=99),
            event("kernel", "kernel", 150, 50, stream=7),
            event("morphserve_overlap_FP16", "cpu_op", 1000, 500),
            event("Memcpy HtoD", "gpu_memcpy", 1100, 200, stream=9, byte_count=400),
            event("kernel", "kernel", 1150, 50, stream=9),
        ]

        result = self.analyze(events)

        self.assertFalse(result["passed"])
        self.assertIn("no size-matched", result["phases"]["W4"]["error"])
        self.assertIn("another stream", result["phases"]["FP16"]["error"])


if __name__ == "__main__":
    unittest.main()
