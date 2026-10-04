#!/usr/bin/env python3
"""Validate dedicated runtime directories before any destructive staging."""
import os
from pathlib import Path
import sys


def validate(env):
    def path(key):
        value = env.get(key, "")
        if not value or "/path/to/" in value:
            raise ValueError(f"Configure {key} with a real path")
        raw = Path(value).expanduser()
        if not raw.is_absolute():
            raise ValueError(f"{key} must be absolute")
        # Reject symlink ancestors rather than following them into shared data.
        if any(part.is_symlink() for part in (raw, *raw.parents)):
            raise ValueError(f"{key} must not traverse symlinks")
        return raw.resolve()
    root = path("MMPTB_ROOT")
    repo = path("REPO_ROOT")
    model = path("MODEL_DIR")
    data = path("DATA_DIR")
    cache = path("HF_CACHE_DIR")
    work, logs = path("WORKSPACE_HOST"), path("LOGS_HOST")
    for target, kind in [(work, "workspaces"), (logs, "logs")]:
        boundary = root / kind
        if target == boundary or not target.is_relative_to(boundary):
            raise ValueError(f"Run directory must be a child of {boundary}")
        for protected in (repo, model, data, cache):
            if target == protected or target.is_relative_to(protected) or protected.is_relative_to(target):
                raise ValueError("Run directory overlaps protected repository/model/data/cache")
    if work == logs or work.is_relative_to(logs) or logs.is_relative_to(work):
        raise ValueError("Workspace and logs must be disjoint")
    return work, logs

if __name__ == "__main__":
    try:
        validate(os.environ)
    except ValueError as exc:
        print(f"Unsafe runtime paths: {exc}", file=sys.stderr)
        sys.exit(2)
