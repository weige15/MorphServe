#!/usr/bin/env python3
"""Capture the local MorphServe reproduction environment as JSON."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> dict[str, object]:
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=60, check=False)
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except Exception as exc:  # environment evidence must survive missing tools
        return {"command": command, "error": f"{type(exc).__name__}: {exc}"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in (
        "torch",
        "transformers",
        "safetensors",
        "datasets",
        "triton",
        "ray",
        "accelerate",
        "vllm_flash_attn",
        "awq",
        "evaluate",
    ):
        try:
            module = importlib.import_module(name)
            result[name] = str(getattr(module, "__version__", "installed-version-unknown"))
        except Exception as exc:
            result[name] = f"unavailable: {type(exc).__name__}: {exc}"
    return result


def gpu_sysfs() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    query = run(
        [
            "nvidia-smi",
            "--query-gpu=index,pci.bus_id,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    stdout = str(query.get("stdout", ""))
    for line in stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 4)]
        if len(parts) != 5:
            continue
        index, bus_id, name, memory_mib, driver = parts
        sysfs_bus_id = bus_id.lower()
        if sysfs_bus_id.startswith("00000000:"):
            sysfs_bus_id = "0000:" + sysfs_bus_id.split(":", 1)[1]
        device_path = Path("/sys/bus/pci/devices") / sysfs_bus_id
        row = {
            "index": index,
            "pci_bus_id": bus_id,
            "name": name,
            "memory_total_mib": memory_mib,
            "driver_version": driver,
        }
        for key in ("current_link_speed", "current_link_width", "max_link_speed", "max_link_width"):
            path = device_path / key
            if path.exists():
                row[key] = path.read_text().strip()
        rows.append(row)
    return rows


def inventory(path_string: str) -> dict[str, object]:
    path = Path(path_string).expanduser().resolve()
    record: dict[str, object] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return record
    files = []
    for item in sorted(path.rglob("*")):
        if item.is_file():
            resolved = item.resolve()
            files.append(
                {
                    "relative_path": str(item.relative_to(path)),
                    "resolved_path": str(resolved),
                    "bytes": resolved.stat().st_size,
                    "sha256": sha256(resolved),
                }
            )
    record["files"] = files
    record["total_bytes"] = sum(int(item["bytes"]) for item in files)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--asset", action="append", default=[])
    args = parser.parse_args()

    torch_details: dict[str, object] = {}
    try:
        import torch

        torch_details = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cudnn": torch.backends.cudnn.version(),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    except Exception as exc:
        torch_details = {"error": f"{type(exc).__name__}: {exc}"}

    commands = {
        "uname": run(["uname", "-a"]),
        "nvidia_smi_full": run(["nvidia-smi"]),
        "nvidia_smi_processes": run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader",
            ]
        ),
        "nvidia_topology": run(["nvidia-smi", "topo", "-m"]),
        "nvcc": run(["nvcc", "--version"]),
        "lscpu": run(["lscpu"]),
        "free": run(["free", "-b"]),
        "df": run(["df", "-B1", "."]),
        "ulimit_memlock": run(["bash", "-lc", "ulimit -l"]),
        "ulimit_all": run(["bash", "-lc", "ulimit -a"]),
        "pip_freeze": run([sys.executable, "-m", "pip", "freeze"]),
    }

    payload = {
        "schema_version": 1,
        "captured_at_utc": run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"]).get("stdout", "").strip(),
        "cwd": os.getcwd(),
        "python": {"executable": sys.executable, "version": sys.version},
        "platform": platform.platform(),
        "hostname": platform.node(),
        "packages": package_versions(),
        "torch": torch_details,
        "gpus": gpu_sysfs(),
        "commands": commands,
        "assets": [inventory(asset) for asset in args.asset],
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key in {"CUDA_VISIBLE_DEVICES", "HF_HOME", "TRANSFORMERS_CACHE", "CONDA_PREFIX", "VIRTUAL_ENV"}
        },
        "executables": {
            name: shutil.which(name)
            for name in ("python", "python3", "uv", "nvidia-smi", "nvcc", "git", "mutool")
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
