#!/usr/bin/env python3
"""Regenerate the modified-condition layer-transfer diagnostic figure from raw JSON.

This figure intentionally excludes decode/overlap values because the source run's
steady-state overlap gate failed. It visualizes only the separately valid transfer
measurements retained from that run.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLORS = {"W4": "#E69F00", "FP16": "#0072B2"}  # Okabe-Ito


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.input.read_text())
    if data.get("passed") is not False:
        raise ValueError("expected a rejected source run; refusing to relabel it")
    rows = data["rows"]
    if not rows or any(row["precision"] not in COLORS for row in rows):
        raise ValueError("missing or unknown precision rows")

    records = []
    for row in rows:
        records.append(
            {
                "precision": row["precision"],
                "transfer_bytes": int(row["transfer_bytes"]),
                "copy_ms": float(row["copy_ms"]),
                "effective_gbps": row["transfer_bytes"] / 1e9 / (row["copy_ms"] / 1000),
                "enqueue_host_ms": float(row["enqueue_host_ms"]),
                "ready_at_enqueue_return": bool(row["ready_at_enqueue_return"]),
            }
        )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.65))
    labels = ["W4\n113.3 MB", "FP16\n436.2 MB"]
    for x, precision in enumerate(("W4", "FP16")):
        subset = [record for record in records if record["precision"] == precision]
        offsets = np.linspace(-0.08, 0.08, len(subset))
        for metric, ax in (("copy_ms", axes[0]), ("effective_gbps", axes[1])):
            values = [record[metric] for record in subset]
            ax.scatter(
                x + offsets,
                values,
                color=COLORS[precision],
                edgecolor="white",
                linewidth=0.6,
                s=38,
                zorder=3,
                label=precision if metric == "copy_ms" else None,
            )
            median = float(np.median(values))
            ax.plot([x - 0.18, x + 0.18], [median, median], color="#333333", linewidth=1.3, zorder=4)

    axes[0].set_ylabel("Pinned H2D transfer time (ms)")
    axes[1].set_ylabel("Effective transfer rate (GB/s)")
    for ax in axes:
        ax.set_xticks([0, 1], labels)
        ax.set_xlim(-0.45, 1.45)
    axes[0].set_title("Observed transfer duration")
    axes[1].set_title("Size-normalized rate")
    fig.suptitle("Llama 3.1 8B layer-25 transfer diagnostics (3 repeats)", fontweight="bold", y=1.02)
    fig.text(
        0.5,
        -0.02,
        "Modified condition; transfer subset only. Source run failed the steady-state overlap gate.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout()
    metadata = {"Title": "MorphServe modified-condition transfer diagnostics", "Creator": "gen_fig_transfer_diagnostics.py", "CreationDate": None, "ModDate": None}
    fig.savefig(args.output_prefix.with_suffix(".pdf"), metadata=metadata)
    fig.savefig(args.output_prefix.with_suffix(".png"), metadata={"Software": "gen_fig_transfer_diagnostics.py"})
    plt.close(fig)


if __name__ == "__main__":
    main()
