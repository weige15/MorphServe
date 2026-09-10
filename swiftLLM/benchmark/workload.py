"""Deterministic open-loop arrival schedules."""

from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class ScheduledRequest:
    sequence: int
    planned_offset_s: float


def build_schedule(
    target_rps: float,
    arrival_mode: str,
    *,
    request_count: int | None = None,
    duration_s: float | None = None,
    seed: int = 0,
) -> list[ScheduledRequest]:
    """Build a schedule without consulting request completion state.

    The first request is scheduled at offset zero. For a fixed schedule,
    subsequent requests are exactly 1 / target_rps apart. For a Poisson
    schedule, subsequent gaps are independent exponential samples. A count
    gives an exact number of requests; a duration includes offsets strictly
    before that duration and always yields the first request for positive
    duration.
    """
    if target_rps <= 0:
        raise ValueError("target_rps must be positive")
    if arrival_mode not in {"fixed", "poisson"}:
        raise ValueError("arrival_mode must be 'fixed' or 'poisson'")
    if (request_count is None) == (duration_s is None):
        raise ValueError("provide exactly one of request_count or duration_s")
    if request_count is not None and request_count <= 0:
        raise ValueError("request_count must be positive")
    if duration_s is not None and duration_s <= 0:
        raise ValueError("duration_s must be positive")

    rng = random.Random(seed)
    offsets: list[float] = []
    offset = 0.0
    limit = request_count
    while limit is None or len(offsets) < limit:
        if duration_s is not None and offset >= duration_s:
            break
        offsets.append(offset)
        if arrival_mode == "fixed":
            gap = 1.0 / target_rps
        else:
            gap = rng.expovariate(target_rps)
        offset += gap

    if not offsets:
        raise ValueError("duration_s is too short to schedule a request")
    return [ScheduledRequest(i, planned) for i, planned in enumerate(offsets)]
