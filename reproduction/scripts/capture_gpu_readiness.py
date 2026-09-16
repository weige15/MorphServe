#!/usr/bin/env python3
"""Capture whether any local GPU can safely run the frozen 8B rerun protocols."""

from __future__ import annotations

import argparse
import csv
import datetime
import io
import json
import subprocess
from pathlib import Path


def run(*args: str) -> str:
    return subprocess.run(args, check=True, text=True, capture_output=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-free-mib", type=int, default=17_408)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    gpu_query = (
        "index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu"
    )
    gpu_text = run(
        "nvidia-smi", f"--query-gpu={gpu_query}", "--format=csv,noheader,nounits"
    )
    fields = ("index", "uuid", "name", "memory_total_mib", "memory_used_mib", "memory_free_mib", "utilization_percent")
    gpus = []
    for row in csv.reader(io.StringIO(gpu_text), skipinitialspace=True):
        record = dict(zip(fields, row))
        for key in ("index", "memory_total_mib", "memory_used_mib", "memory_free_mib", "utilization_percent"):
            record[key] = int(record[key])
        record["meets_free_memory_threshold"] = record["memory_free_mib"] >= args.minimum_free_mib
        gpus.append(record)

    apps_text = run(
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"
    )
    apps = []
    if apps_text:
        for row in csv.reader(io.StringIO(apps_text), skipinitialspace=True):
            apps.append({"gpu_uuid": row[0], "pid": int(row[1]), "used_memory_mib": int(row[2])})

    eligible = [gpu["index"] for gpu in gpus if gpu["meets_free_memory_threshold"]]
    payload = {
        "schema_version": 1,
        "captured_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_revision": run("git", "-C", str(root), "rev-parse", "HEAD"),
        "minimum_free_mib": args.minimum_free_mib,
        "minimum_free_bytes": args.minimum_free_mib * 1024 * 1024,
        "gpu_query_command": f"nvidia-smi --query-gpu={gpu_query} --format=csv,noheader,nounits",
        "compute_apps_query_command": "nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits",
        "gpus": gpus,
        "compute_apps": apps,
        "eligible_gpu_indices": eligible,
        "max_free_mib": max(gpu["memory_free_mib"] for gpu in gpus),
        "decision": "ready" if eligible else "blocked_below_frozen_headroom_threshold",
        "scope": "Point-in-time resource readiness only; not GPU time, ownership, or experiment evidence.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in ("decision", "eligible_gpu_indices", "max_free_mib", "minimum_free_mib")}, sort_keys=True))


if __name__ == "__main__":
    main()
