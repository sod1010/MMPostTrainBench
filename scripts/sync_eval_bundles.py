#!/usr/bin/env python3
"""Regenerate evaluation copies, or check them without importing GPU libraries."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/harbor_adapter"))
from adapter import BENCHMARKS, LMMS_BENCHES, MMSWE_BENCHES


def copies():
    yield ROOT / "src/eval/split_util.py", ROOT / "eval_omni/runners/split_util.py"
    for bench in BENCHMARKS:
        for side in ("tests", "environment"):
            dest = ROOT / f"harbor_tasks/mmposttrainbench-{bench}-qwen3-omni-30b/{side}"
            yield ROOT / f"src/eval/tasks/{bench}/evaluate.py", dest / "evaluate.py"
            for name in ("split_util.py", "diag_util.py", "eval_util.py"):
                yield ROOT / "src/eval" / name, dest / name
            if bench in LMMS_BENCHES:
                yield ROOT / "src/eval/lmms_common/evaluate.py", dest / "lmms_common/evaluate.py"
            if bench == "mmswe":
                yield ROOT / "src/eval/tasks/mmswe/dlc_native_grade.py", dest / "dlc_native_grade.py"
            if side == "tests":
                yield ROOT / "src/harbor_adapter/template/tests/test.sh", dest / "test.sh"
                dockerfile = "Dockerfile.mmswe" if bench in MMSWE_BENCHES else "Dockerfile"
                yield ROOT / "src/harbor_adapter/template/tests" / dockerfile, dest / "Dockerfile"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if not args.check:
        subprocess.run([sys.executable, str(ROOT / "src/harbor_adapter/run_adapter.py"),
                        "--all", "-o", str(ROOT / "harbor_tasks")], cwd=ROOT, check=True)
        shutil.copyfile(ROOT / "src/eval/split_util.py", ROOT / "eval_omni/runners/split_util.py")
    mismatches = [str(dest.relative_to(ROOT)) for source, dest in copies()
                  if not dest.is_file() or source.read_bytes() != dest.read_bytes()]
    if mismatches:
        print("Mismatched evaluation copies:\n" + "\n".join(mismatches))
        return 1
    print(f"Evaluation copies verified: {sum(1 for _ in copies())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
