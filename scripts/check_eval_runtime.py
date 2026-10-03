#!/usr/bin/env python3
"""Bounded runtime probes; no model loading, image pull, training or loop runs."""
import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def probe(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return {"ok": result.returncode == 0, "rc": result.returncode,
                "stdout": result.stdout.strip()[-2000:], "stderr": result.stderr.strip()[-2000:]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"ok": False, "error": str(error)}


def check():
    result = {"platform": platform.system(), "scope": "preflight only; no container build or model inference"}
    result["docker"] = probe(["docker", "version", "--format", "{{.Server.Version}}"])
    result["gpu"] = probe(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    # Official test scripts may use su/setgroups. A rootless namespace that can
    # start bash but cannot set groups is insufficient for these cases.
    payload = "import os; os.setgroups([]); print('group-switch capability available')"
    common = ["--mount", "--uts", "--ipc", "--pid", "--fork", sys.executable, "-c", payload]
    result["native_namespace"] = probe(["unshare", *common])
    result["user_namespace"] = probe(["unshare", "--map-root-user", *common])
    result["mmswe_native_runtime_ready"] = result["native_namespace"]["ok"] or result["user_namespace"]["ok"]
    result["ready_for_container_smoke"] = result["docker"]["ok"] and result["gpu"]["ok"] and result["mmswe_native_runtime_ready"]
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    result = check()
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end="")
    return 0 if result["ready_for_container_smoke"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
