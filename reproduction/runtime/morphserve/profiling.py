"""Paper-faithful greedy selection for MorphServe Algorithm 1."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path


def rank_layers(
    lts: Mapping[int, float],
    lrs: Mapping[int, float],
    mds: Callable[[int, Sequence[int]], float],
) -> dict:
    """Rank layers by recomputing conditioned MDS for every remaining candidate."""
    if set(lts) != set(lrs):
        raise ValueError("LTS and LRS must contain the same layer indices")
    if not lts:
        raise ValueError("at least one layer is required")
    if any(not math.isfinite(float(value)) for value in (*lts.values(), *lrs.values())):
        raise ValueError("LTS and LRS values must be finite")
    remaining = set(lts)
    order: list[int] = []
    steps = []
    while remaining:
        candidates = []
        for layer in sorted(remaining):
            mds_value = float(mds(layer, tuple(order)))
            if not math.isfinite(mds_value):
                raise ValueError("MDS values must be finite")
            score = 0.25 * float(lts[layer]) + 0.25 * float(lrs[layer]) + 0.5 * mds_value
            candidates.append(
                {"layer": layer, "lts": float(lts[layer]), "lrs": float(lrs[layer]), "mds": mds_value, "lis": score}
            )
        selected = max(candidates, key=lambda candidate: (candidate["lis"], -candidate["layer"]))["layer"]
        steps.append({"step": len(order), "quantized_before": list(order), "candidates": candidates, "selected": selected})
        order.append(selected)
        remaining.remove(selected)
    return {
        "schema_version": 1,
        "algorithm": "MorphServe Algorithm 1 conditioned-MDS greedy LIS",
        "weights": {"lts": 0.25, "lrs": 0.25, "mds": 0.5},
        "tie_break": "lowest_layer_index",
        "order": order,
        "steps": steps,
    }


def save_profile(profile: Mapping, path: str | Path) -> None:
    """Write a profile as stable, human-readable JSON."""
    Path(path).write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
