#!/usr/bin/env python3
"""Inventory candidate CPython 3.11 bytecode provenance without executing it."""

import argparse
import datetime
import hashlib
import json
import marshal
import struct
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    records = []
    for pyc in sorted(root.rglob("*.cpython-311.pyc")):
        data = pyc.read_bytes()
        if len(data) < 16:
            raise ValueError(f"truncated pyc: {pyc}")
        magic, flags, timestamp, source_size = struct.unpack("<4sIII", data[:16])
        if flags != 0:
            raise ValueError(f"expected timestamp-based pyc: {pyc}")
        code = marshal.loads(data[16:])
        source_name = pyc.name.split(".cpython-")[0] + ".py"
        source = pyc.parent.parent / source_name
        records.append(
            {
                "pyc": pyc.relative_to(root).as_posix(),
                "pyc_sha256": hashlib.sha256(data).hexdigest(),
                "magic_hex": magic.hex(),
                "compiled_source_filename": code.co_filename,
                "header_timestamp_utc": datetime.datetime.fromtimestamp(timestamp, datetime.timezone.utc).isoformat(),
                "header_source_bytes": source_size,
                "current_source": source.relative_to(root).as_posix() if source.is_file() else None,
                "current_source_bytes": source.stat().st_size if source.is_file() else None,
                "source_size_matches": source.is_file() and source.stat().st_size == source_size,
            }
        )
    payload = {
        "schema_version": 1,
        "classification": "candidate-artifact bytecode provenance; no bytecode was executed",
        "root": args.root.as_posix(),
        "records": records,
        "summary": {
            "cpython311_files": len(records),
            "source_size_matches": sum(row["source_size_matches"] for row in records),
            "source_size_differs": sum(not row["source_size_matches"] for row in records),
            "earliest_header_timestamp_utc": min(row["header_timestamp_utc"] for row in records),
            "latest_header_timestamp_utc": max(row["header_timestamp_utc"] for row in records),
            "compiled_path_prefixes": sorted({str(Path(row["compiled_source_filename"]).parent) for row in records}),
        },
        "limitations": [
            "A timestamp and source-size match is provenance metadata, not proof that source and bytecode are semantically identical.",
            "A size mismatch proves the checked-in source is not byte-for-byte the source named by that timestamp-based pyc header.",
            "The candidate repository is not verified as an author release."
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
