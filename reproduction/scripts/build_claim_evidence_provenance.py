#!/usr/bin/env python3
"""Hash and Git-audit every path referenced by the H1-H30 evidence map."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    map_path = args.map.resolve()
    reproduction = map_path.parents[1]
    repository = reproduction.parent
    claim_map = json.loads(map_path.read_text())
    fields = ("paper_reference", "commands", "configs", "raw", "comparison")
    referenced_by = {}
    for claim, row in claim_map["claims"].items():
        for field in fields:
            for relative in row[field]:
                referenced_by.setdefault(relative, []).append({"claim": claim, "field": field})

    artifacts = {}
    for relative in sorted(referenced_by):
        path = reproduction / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        repository_relative = path.relative_to(repository).as_posix()
        status = git(repository, "status", "--short", "--", repository_relative)
        tracked = subprocess.run(
            ["git", "-C", str(repository), "ls-files", "--error-unmatch", repository_relative],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0
        last_commit = git(repository, "log", "-1", "--format=%H", "--", repository_relative) if tracked else None
        source_revision_path = path.parent / "source-revision.txt"
        run_source_revision = source_revision_path.read_text().strip() if source_revision_path.is_file() else None
        data = path.read_bytes()
        artifacts[relative] = {
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "tracked": tracked,
            "artifact_last_commit": last_commit or None,
            "worktree_status": status or None,
            "run_source_revision": run_source_revision,
            "referenced_by": referenced_by[relative],
        }

    payload = {
        "schema_version": 1,
        "claim_map": map_path.relative_to(reproduction).as_posix(),
        "repository_head_at_generation": git(repository, "rev-parse", "HEAD"),
        "source_revision_caveat": "run_source_revision is null for historical runners that predate source-revision capture; artifact_last_commit is artifact provenance, not proof of the exact executed source tree",
        "artifacts": artifacts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"artifacts": len(artifacts), "claims": len(claim_map["claims"]), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
