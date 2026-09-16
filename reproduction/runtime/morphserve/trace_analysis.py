"""Fail-closed analysis of CUDA kernel/H2D overlap in Chrome traces."""

from __future__ import annotations

import json
from pathlib import Path


def _interval(event):
    if event.get("ph") != "X" or "ts" not in event or "dur" not in event:
        return None
    return float(event["ts"]), float(event["ts"]) + float(event["dur"])


def _overlap(left, right):
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def _bytes(event):
    args = event.get("args", {})
    for key, value in args.items():
        normalized = key.lower().replace(" ", "_")
        if normalized in {"bytes", "size", "num_bytes", "memory_bandwidth_(bytes)"}:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return None


def _stream(event):
    for key, value in event.get("args", {}).items():
        if key.lower() in {"stream", "stream_id"}:
            return str(value)
    return None


def analyze_cuda_overlap_trace(path, expected_bytes):
    """Require a size-matched H2D copy to overlap a CUDA kernel in each phase."""
    events = json.loads(Path(path).read_text()).get("traceEvents", [])
    phases = {}
    for precision, byte_count in expected_bytes.items():
        marker_name = f"morphserve_overlap_{precision}"
        markers = [event for event in events if event.get("name") == marker_name and _interval(event)]
        if len(markers) != 1:
            phases[precision] = {"passed": False, "error": f"expected one {marker_name} marker, found {len(markers)}"}
            continue
        marker_interval = _interval(markers[0])
        memcpy = []
        kernels = []
        for event in events:
            interval = _interval(event)
            if not interval or _overlap(interval, marker_interval) <= 0:
                continue
            name = str(event.get("name", "")); category = str(event.get("cat", "")).lower()
            lower_name = name.lower()
            if "memcpy" in lower_name or "gpu_memcpy" in category:
                direction = " ".join([lower_name, *(str(value).lower() for value in event.get("args", {}).values())])
                if "htod" in direction or "host to device" in direction:
                    memcpy.append(event)
            elif "kernel" in category:
                kernels.append(event)
        size_matched = [event for event in memcpy if _bytes(event) == int(byte_count)]
        if not size_matched:
            phases[precision] = {
                "passed": False,
                "error": "no size-matched H2D CUDA activity",
                "expected_copy_bytes": int(byte_count),
                "observed_copy_bytes": sorted({value for value in (_bytes(event) for event in memcpy) if value is not None}),
            }
            continue
        copy_event = max(size_matched, key=lambda event: float(event["dur"]))
        copy_interval = _interval(copy_event); copy_stream = _stream(copy_event)
        if copy_stream is None:
            phases[precision] = {
                "passed": False,
                "error": "size-matched H2D activity lacks stream metadata",
                "expected_copy_bytes": int(byte_count),
                "copy_duration_us": float(copy_event["dur"]),
            }
            continue
        overlaps = []
        for event in kernels:
            kernel_stream = _stream(event)
            if kernel_stream is None or kernel_stream == copy_stream:
                continue
            overlap_us = _overlap(copy_interval, _interval(event))
            if overlap_us > 0:
                overlaps.append((overlap_us, event))
        if not overlaps:
            phases[precision] = {
                "passed": False,
                "error": "size-matched H2D activity did not overlap a CUDA kernel on another stream",
                "expected_copy_bytes": int(byte_count),
                "copy_duration_us": float(copy_event["dur"]),
                "copy_stream": copy_stream,
            }
            continue
        overlap_us, kernel = max(overlaps, key=lambda item: item[0])
        phases[precision] = {
            "passed": True,
            "expected_copy_bytes": int(byte_count),
            "copy_name": copy_event.get("name"),
            "copy_duration_us": float(copy_event["dur"]),
            "copy_stream": copy_stream,
            "overlap_kernel_name": kernel.get("name"),
            "overlap_kernel_duration_us": float(kernel["dur"]),
            "kernel_stream": _stream(kernel),
            "intersection_us": overlap_us,
        }
    return {"schema_version": 1, "phases": phases, "passed": bool(phases) and all(row.get("passed", False) for row in phases.values())}
