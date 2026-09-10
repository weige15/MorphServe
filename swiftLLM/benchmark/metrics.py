"""Pure metric definitions shared by the runner, summarizer, and tests.

All timestamps are monotonic nanoseconds from one process.  A request's
client-observed streaming timestamps are used for TTFT/TPOT; engine timestamps
remain in the raw row so queueing and model-side timing can be audited.
"""

from __future__ import annotations

from typing import Any, Mapping

NS_PER_SECOND = 1_000_000_000


def _delta_seconds(end: Any, start: Any) -> float | None:
    if end is None or start is None:
        return None
    return (int(end) - int(start)) / NS_PER_SECOND


def derive_request_metrics(row: Mapping[str, Any]) -> dict[str, float | int | None]:
    """Derive auditable request metrics from raw timestamps only."""
    output_count = int(row.get("output_token_count", 0) or 0)
    ttft = _delta_seconds(
        row.get("first_stream_token_received_time_ns"),
        row.get("arrival_time_ns"),
    )
    engine_ttft = _delta_seconds(
        row.get("first_output_token_time_ns"),
        row.get("arrival_time_ns"),
    )
    queueing = _delta_seconds(
        row.get("first_prefill_time_ns"),
        row.get("scheduler_eligible_time_ns"),
    )
    client_tpot = None
    engine_tpot = None
    if output_count > 1:
        client_tpot = _delta_seconds(
            row.get("stream_completion_time_ns"),
            row.get("first_stream_token_received_time_ns"),
        )
        engine_tpot = _delta_seconds(
            row.get("completion_time_ns"),
            row.get("first_output_token_time_ns"),
        )
        if client_tpot is not None:
            client_tpot /= output_count - 1
        if engine_tpot is not None:
            engine_tpot /= output_count - 1
    elif output_count == 1:
        client_tpot = 0.0
        engine_tpot = 0.0

    return {
        "ttft_s": ttft,
        "engine_ttft_s": engine_ttft,
        "queueing_delay_s": queueing,
        "tpot_s": client_tpot,
        "engine_tpot_s": engine_tpot,
    }


def percentile(values: list[float], percentile_value: float) -> float | None:
    """Return a linear-interpolated percentile, or None for no observations."""
    if not values:
        return None
    if not 0 <= percentile_value <= 100:
        raise ValueError("percentile must be in [0, 100]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile_value / 100
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def distribution(values: list[float]) -> dict[str, float | int | None]:
    """Return count/mean/P50/P95/P99 for one metric."""
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }
