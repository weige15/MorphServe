"""Run one v11 throughput-confirmation condition using the frozen v10 runtime."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from . import run_release_condition as v10_runner
from .instrumented_release_intent_controller import InstrumentedReleaseIntentController
from .run import _git_info


ARTIFACT_VERSION = "throughput_confirmation_v11"
EVENT_LOOP_PROBE_INTERVAL_S = 0.05
SOURCE_PATHS = tuple(
    dict.fromkeys(
        v10_runner.SOURCE_PATHS
        + (
            "swiftLLM/benchmark/instrumented_release_intent_controller.py",
            "swiftLLM/benchmark/run_throughput_confirmation_condition.py",
            "swiftLLM/benchmark/run_throughput_confirmation_plan.py",
            "swiftLLM/benchmark/replay_throughput_controller_parity.py",
            "swiftLLM/benchmark/analyze_throughput_confirmation.py",
            "swiftLLM/benchmark/audit_throughput_confirmation.py",
            "swiftLLM/benchmark/test_throughput_confirmation.py",
        )
    )
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_provenance() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in SOURCE_PATHS
    }


def validate_preregistration(args: argparse.Namespace) -> None:
    repo = Path(__file__).resolve().parents[2]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("artifact_version") != ARTIFACT_VERSION:
        raise RuntimeError("wrong throughput-confirmation protocol manifest")
    if source_provenance() != manifest.get("source_sha256_at_preregistration"):
        raise RuntimeError("runtime/source hashes differ from the preregistered v11 manifest")
    protocol = repo / manifest["repository"]["protocol_path"]
    if sha256_file(protocol) != manifest["repository"]["protocol_sha256"]:
        raise RuntimeError("protocol bytes differ from the preregistered v11 manifest")
    workload_hash = sha256_file(args.workload)
    if workload_hash not in {
        record["sha256"] for record in manifest["workloads"]["files"].values()
    }:
        raise RuntimeError("workload is not one of the unchanged v10 inputs")
    _, dirty, _, status = _git_info()
    if dirty:
        raise RuntimeError(f"v11 runs require a clean Git tree:\n{status}")


async def _probe_event_loop(
    stop: asyncio.Event, rows: list[dict[str, Any]]
) -> None:
    loop = asyncio.get_running_loop()
    anchor_loop = loop.time()
    tick = 0
    while not stop.is_set():
        deadline = anchor_loop + tick * EVENT_LOOP_PROBE_INTERVAL_S
        delay = deadline - loop.time()
        if delay > 0:
            try:
                await asyncio.wait_for(stop.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
        if stop.is_set():
            return
        observed_loop = loop.time()
        rows.append(
            {
                "schema_version": 1,
                "sample_index": tick,
                "timestamp_ns": time.perf_counter_ns(),
                "scheduled_loop_time_s": deadline,
                "observed_loop_time_s": observed_loop,
                "event_loop_lag_s": max(0.0, observed_loop - deadline),
            }
        )
        tick += 1


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


async def run(args: argparse.Namespace) -> Path:
    """Delegate serving to v10 while replacing only observation/provenance hooks."""
    original_controller = v10_runner.ReleaseIntentController
    original_source_provenance = v10_runner.source_provenance
    original_validate = v10_runner.validate_preregistration
    v10_runner.ReleaseIntentController = InstrumentedReleaseIntentController
    v10_runner.source_provenance = source_provenance
    v10_runner.validate_preregistration = validate_preregistration

    probe_rows: list[dict[str, Any]] = []
    stop_probe = asyncio.Event()
    probe_task = asyncio.create_task(_probe_event_loop(stop_probe, probe_rows))
    try:
        run_dir = await v10_runner.run(args)
    finally:
        stop_probe.set()
        await asyncio.gather(probe_task, return_exceptions=True)
        v10_runner.ReleaseIntentController = original_controller
        v10_runner.source_provenance = original_source_provenance
        v10_runner.validate_preregistration = original_validate

    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    measurement_start_ns = int(metadata["measurement_start_time_ns"])
    measurement_end_ns = int(metadata["measurement_end_time_ns"])
    measurement_probe_rows = [
        {
            **row,
            "elapsed_s": (int(row["timestamp_ns"]) - measurement_start_ns) / 1e9,
        }
        for row in probe_rows
        if measurement_start_ns <= int(row["timestamp_ns"]) <= measurement_end_ns
    ]
    _write_jsonl(run_dir / "event_loop_lag.jsonl", measurement_probe_rows)

    metadata["experiment"] = ARTIFACT_VERSION
    metadata["instrumentation"] = {
        "policy_effect": "none; output-only monotonic/CPU timing fields",
        "controller_class": "InstrumentedReleaseIntentController",
        "event_loop_probe_interval_s": EVENT_LOOP_PROBE_INTERVAL_S,
        "telemetry_lateness": "existing sampling_jitter_s on the frozen 0.25-second grid",
        "request_launch_jitter": "existing arrival_jitter_s",
        "output_consumer_lag": (
            "derived from first_stream_token_received_time_ns-first_output_token_time_ns "
            "and stream_completion_time_ns-completion_time_ns"
        ),
        "controller_trace_bookkeeping_boundary": (
            "decision construction plus trace_fields; outer runner row copy/append is not separably timed"
        ),
        "process_cpu_clock_caveat": "process_time_ns is process-wide; wall time is primary attribution",
    }
    metadata["raw_files"] = list(metadata["raw_files"]) + ["event_loop_lag.jsonl"]
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    v10_runner.add_args(parser)
    args = parser.parse_args()
    path = asyncio.run(run(args))
    print(f"raw throughput-confirmation v11 run: {path}")


if __name__ == "__main__":
    main()
