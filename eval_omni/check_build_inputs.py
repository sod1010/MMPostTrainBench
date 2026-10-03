#!/usr/bin/env python3
"""Fail early when the untracked third-party build inputs are absent."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REQUIRED = (
    "harness_repos/lmms-eval/lmms_eval/__init__.py",
    "harness_repos/lmms-eval/lmms_eval/tasks/videomme_v2/utils.py",
    "harness_repos/OmniVideoBench/eval/qwen3_omni_eval.py",
    "runners/split_util.py",
)

def main():
    missing = [path for path in REQUIRED if not (ROOT / path).is_file()]
    if missing:
        print("Missing evaluator build inputs:\n" + "\n".join(missing), file=sys.stderr)
        print("Run: python3 eval_omni/prepare_harnesses.py", file=sys.stderr)
        return 1
    import json
    from prepare_harnesses import verify, verify_patches
    try:
        manifest = json.loads((ROOT / "harnesses.lock.json").read_text())
        for spec in manifest["harnesses"]:
            verify_patches(spec, ROOT)
            verify(ROOT / "harness_repos" / spec["name"], spec)
    except (ValueError, OSError) as error:
        print(f"Evaluator source verification failed: {error}", file=sys.stderr)
        return 1
    print("Evaluator build inputs match pinned commits and patches (runtime validation is separate).")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
