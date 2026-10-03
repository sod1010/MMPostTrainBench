#!/usr/bin/env python3
"""Fetch pinned upstream source, apply checked patches, verify every source file."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent


def ignored(path):
    return any(p in (".git", "__pycache__") or p.endswith(".egg-info") for p in path.parts) or path.suffix in (".pyc", ".pyo", ".log")


def source_tree(root):
    rows = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if ignored(rel):
            continue
        if path.is_symlink():
            raise ValueError(f"unexpected source symlink: {rel}")
        if path.is_file():
            rows.append(f"{rel.as_posix()}\t{hashlib.sha256(path.read_bytes()).hexdigest()}\n")
    return {"file_count": len(rows), "sha256": hashlib.sha256("".join(rows).encode()).hexdigest()}


def verify(root, spec):
    if not root.is_dir():
        raise ValueError(f"{spec['name']}: source directory missing")
    if source_tree(root) != spec["source_tree"]:
        raise ValueError(f"{spec['name']}: source tree differs from the lock; existing files were not overwritten")


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


def verify_patches(spec, manifest_dir):
    for item in spec["patches"]:
        patch = manifest_dir / item["path"]
        if hashlib.sha256(patch.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("patch checksum mismatch")


def prepare(spec, dest, manifest_dir):
    verify_patches(spec, manifest_dir)
    target = dest / spec["name"]
    if target.exists():
        verify(target, spec)
        return
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".harness-", dir=dest) as td:
        root = Path(td)
        checkout, staged = root / "checkout", root / "source"
        checkout.mkdir()
        git(checkout, "init", "--quiet")
        git(checkout, "fetch", "--quiet", "--depth=1", spec["url"], spec["commit"])
        git(checkout, "checkout", "--quiet", "--detach", "FETCH_HEAD")
        if git(checkout, "rev-parse", "HEAD") != spec["commit"]:
            raise ValueError("upstream commit mismatch")
        for item in spec["patches"]:
            patch = manifest_dir / item["path"]
            git(checkout, "apply", "--check", str(patch.resolve()))
            git(checkout, "apply", str(patch.resolve()))
        staged.mkdir()
        for entry in spec["paths"]:
            if Path(entry).is_absolute() or ".." in Path(entry).parts:
                raise ValueError("invalid source path in lock")
            origin = checkout / entry
            candidates = origin.rglob("*") if origin.is_dir() else [origin]
            for path in candidates:
                rel = path.relative_to(checkout)
                if ignored(rel):
                    continue
                if path.is_symlink():
                    raise ValueError(f"unexpected source symlink: {rel}")
                if path.is_file():
                    output = staged / rel
                    output.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, output)
        verify(staged, spec)
        staged.rename(target)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=HERE / "harness_repos")
    ap.add_argument("--manifest", type=Path, default=HERE / "harnesses.lock.json")
    ap.add_argument("--check", action="store_true", help="offline source verification; no network or edits")
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text())
    try:
        for spec in manifest["harnesses"]:
            if Path(spec["name"]).name != spec["name"]:
                raise ValueError("invalid harness name")
            verify_patches(spec, args.manifest.resolve().parent)
            if not args.check:
                prepare(spec, args.dest, args.manifest.resolve().parent)
            verify(args.dest / spec["name"], spec)
            print(f"VERIFIED {spec['name']} {spec['commit']} ({spec['source_tree']['file_count']} files)")
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"FAILED: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
