#!/usr/bin/env python3
"""JointAVBench evaluate.py — omni seam adapter (second bench, self-runner shape).

Same contract as the MMAU adapter: obeys the verifier's CLI
(`--model-path / --json-output-file / --limit`, tolerates the text-task retry
flags) and shells to the eval-deliver self-runner `run_jointav_official.py`
(Qwen3-Omni transformers + qwen_omni_utils; per-qid mp4 clip with
use_audio_in_video=True; letter exact-match MCQ accuracy), then normalizes to
the flat `{"accuracy": <0-1>}` contract.

JointAV needs a data json + a media root (the .mp4 clips), so this adapter
passes --data/--media-root to the runner. Override via env:
    JOINTAV_RUNNER      path to run_jointav_official.py
    JOINTAV_DATA        jointavbench.json
    JOINTAV_MEDIA_ROOT  dir holding videos/<qid>.mp4
    JOINTAV_MAX_FRAMES  frames per clip (default 32)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

DEFAULT_RUNNER = os.environ.get("JOINTAV_RUNNER", "/opt/eval/runners/run_jointav_official.py")
# JointAVBench data/media roots (env-driven; default under DATA_DIR).
_DATA_DIR = os.environ.get("DATA_DIR", "data")
DEFAULT_DATA = os.path.join(_DATA_DIR, "evaluationbench", "JointAVBench", "jointavbench.json")
DEFAULT_MEDIA_ROOT = os.path.join(_DATA_DIR, "evaluationbench", "JointAVBench")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="JointAVBench omni eval adapter")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--json-output-file", required=True)
    ap.add_argument("--limit", type=int, default=-1)
    # tolerated-but-ignored (text-task retry ladder passes these)
    ap.add_argument("--templates-dir", default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--max-connections", type=int, default=None)
    ap.add_argument("--gpu-memory-utilization", type=float, default=None)
    return ap.parse_known_args()[0]


def main() -> int:
    args = parse_args()
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from eval_util import prepare_output, configure_split, validate_runner_results
    prepare_output(args.json_output_file)
    try:
        configure_split()
    except (ValueError, ImportError):
        print("[adapter] invalid split configuration; no score", file=sys.stderr)
        return 6
    runner = os.environ.get("JOINTAV_RUNNER", DEFAULT_RUNNER)
    data = os.environ.get("JOINTAV_DATA", DEFAULT_DATA)
    media_root = os.environ.get("JOINTAV_MEDIA_ROOT", DEFAULT_MEDIA_ROOT)
    max_frames = os.environ.get("JOINTAV_MAX_FRAMES", "32")
    for p, what in ((runner, "runner"), (data, "data json")):
        if not os.path.isfile(p):
            print(f"[jointav-adapter] ERROR: {what} not found at {p}", file=sys.stderr)
            return 2

    with tempfile.TemporaryDirectory() as td:
        runner_out = os.path.join(td, "jointav_runner.json")
        cmd = [
            sys.executable, runner,
            "--model-path", args.model_path,
            "--data", data,
            "--media-root", media_root,
            "--max-frames", str(max_frames),
            "--limit", str(args.limit),
            "--out", runner_out,
        ]
        print(f"[jointav-adapter] running: {' '.join(cmd)}", flush=True)
        with open(os.path.join(td, "runner.log"), "w") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            print(f"[jointav-adapter] runner exited rc={rc}; not writing metrics.",
                  file=sys.stderr)
            return rc
        if not os.path.isfile(runner_out):
            print("[jointav-adapter] runner produced no output file.", file=sys.stderr)
            return 3
        with open(runner_out) as f:
            raw = json.load(f)

    try:
        validate_runner_results(raw)
    except (ValueError, AttributeError):
        print("[adapter] incomplete or invalid results; no score", file=sys.stderr)
        return 4
    acc = raw.get("accuracy")
    if acc is None:
        print(f"[jointav-adapter] runner output missing 'accuracy': {raw}", file=sys.stderr)
        return 4

    # --- val/eval split (Evo-Bench) -------------------------------------------
    # run_jointav_official.py already subset by EVAL_SPLIT and computed
    # accuracy/correct/n over that subset (results re-indexed from 0). Do NOT
    # re-apply the split (would double-filter to ~9%/~49% instead of 30/70).
    # Label the task, and persist per-item diagnostics for the 'val' split only.
    _task = raw.get("task", "jointavbench")
    _correct, _n = raw.get("correct"), raw.get("n")
    _split = os.environ.get("EVAL_SPLIT", "")
    _results = raw.get("results")
    if _split and _split not in ("all", "full", "none"):
        _task = f"{_task}@{_split}"
    _out_dir = os.path.dirname(os.path.abspath(args.json_output_file)) or "."
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # src/eval
        from diag_util import dump_val_diag
        _dp = dump_val_diag(_results, _out_dir, _split)
        if _dp:
            print(f"[jointav-adapter] wrote val diagnostics -> {_dp}", file=sys.stderr)
    except Exception as e:
        print(f"[jointav-adapter] diag dump skipped ({e})", file=sys.stderr)

    metrics = {
        "accuracy": float(acc),
        "task": _task,
        "correct": _correct,
        "n": _n,
    }
    if _split == "eval":
        metrics.update(benchmark="jointavbench", eval_split=_split)
    os.makedirs(os.path.dirname(os.path.abspath(args.json_output_file)) or ".", exist_ok=True)
    with open(args.json_output_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[jointav-adapter] accuracy={metrics['accuracy']:.4f} "
          f"({metrics['correct']}/{metrics['n']}) -> {args.json_output_file}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
