"""Arrival-scheduled replay with complete request/token accounting."""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    scheduled_s: float
    prompt: str
    output_len: int
    reference: str | None = None


def percentile(values, q):
    """Hyndman-Fan type 7 / NumPy linear percentile."""
    values = sorted(float(value) for value in values)
    if not values:
        return None
    position = (len(values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (position - lower) * (values[upper] - values[lower])


async def run_replay(specs, submit, request_timeout_s):
    specs = list(specs)
    if request_timeout_s <= 0:
        raise ValueError("request_timeout_s must be positive")
    if len({spec.request_id for spec in specs}) != len(specs):
        raise ValueError("request IDs must be unique")
    if any(spec.scheduled_s < 0 or spec.output_len < 0 for spec in specs):
        raise ValueError("scheduled time and output length must be non-negative")
    if any(left.scheduled_s > right.scheduled_s for left, right in zip(specs, specs[1:])):
        raise ValueError("trace must be monotonic by scheduled time")

    loop = asyncio.get_running_loop()
    origin = loop.time()

    async def launch(spec):
        await asyncio.sleep(max(0.0, origin + spec.scheduled_s - loop.time()))
        actual = loop.time() - origin
        token_ids, token_times = [], []
        adapter_events = []

        def emit(token_id, event=None):
            token_ids.append(int(token_id))
            token_times.append(loop.time() - origin)
            if event is not None:
                adapter_events.append(event)

        metadata, error, timed_out = {}, None, False
        try:
            metadata = await asyncio.wait_for(submit(spec, emit), timeout=request_timeout_s) or {}
        except asyncio.TimeoutError:
            timed_out = True
            error = "TimeoutError"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        completed = loop.time() - origin
        first = token_times[0] if token_times else None
        intervals = [right - left for left, right in zip(token_times, token_times[1:])]
        ttft = None if first is None else first - actual
        record = {
            **asdict(spec),
            "actual_submit_s": actual,
            "submit_lag_s": actual - spec.scheduled_s,
            "first_token_s": first,
            "completion_s": completed,
            "ttft_s": ttft,
            "scheduled_ttft_s": None if first is None else first - spec.scheduled_s,
            "token_ids": token_ids,
            "token_count": len(token_ids),
            "token_timestamps_s": token_times,
            "tpot_intervals_s": intervals,
            "mean_tpot_s": sum(intervals) / len(intervals) if intervals else None,
            "error": error,
            "timed_out": timed_out,
            "adapter_events": adapter_events,
            "metadata": metadata,
        }
        return record

    records = await asyncio.gather(*(launch(spec) for spec in specs))
    return list(records)


def summarize(records):
    records = list(records)
    completed = [record for record in records if not record["error"]]
    ttft = [record["ttft_s"] for record in completed if record["ttft_s"] is not None]
    tpot = [value for record in completed for value in record["tpot_intervals_s"]]
    return {
        "schema_version": 1,
        "request_count": len(records),
        "completed_count": len(completed),
        "error_count": sum(record["error"] is not None for record in records),
        "timeout_count": sum(record["timed_out"] for record in records),
        "submitted_ids": [record["request_id"] for record in records],
        "total_emitted_tokens": sum(record["token_count"] for record in records),
        "p95_ttft_s": percentile(ttft, 0.95),
        "p99_tpot_s": percentile(tpot, 0.99),
        "percentile_definition": "Hyndman-Fan type 7 linear interpolation over per-request TTFT and per-token TPOT intervals",
    }


def write_artifacts(records, jsonl_path, summary_path):
    jsonl_path, summary_path = Path(jsonl_path), Path(summary_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    summary_path.write_text(json.dumps(summarize(records), indent=2, sort_keys=True) + "\n")
