#!/usr/bin/env python3
"""MMAR evaluate.py — omni seam adapter (self-runner shape, like MMAU).

MMAR can't go through lmms-eval here: its HF schema needs datasets>=4 (`List`
feature) but datasets 4 wraps audio in a torchcodec AudioDecoder that the
lmms-eval qwen3_omni wrapper mis-handles. So MMAR uses a dedicated self-runner
(run_mmar_official.py) that loads with datasets 4.8.4, decodes audio itself, and
does MCQ letter scoring — same stable path as MMAU. Normalizes to {"accuracy"}.

env: MMAR_RUNNER (default eval-deliver path).
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, tempfile

DEFAULT_RUNNER = "/opt/eval/runners/run_mmar_official.py"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="MMAR omni eval adapter")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--json-output-file", required=True)
    ap.add_argument("--limit", type=int, default=-1)
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
    runner = os.environ.get("MMAR_RUNNER", DEFAULT_RUNNER)
    if not os.path.isfile(runner):
        print(f"[mmar-adapter] ERROR: runner not found at {runner}", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory() as td:
        runner_out = os.path.join(td, "mmar_runner.json")
        cmd = [sys.executable, runner, "--model-path", args.model_path,
               "--limit", str(args.limit), "--out", runner_out]
        print(f"[mmar-adapter] running: {' '.join(cmd)}", flush=True)
        with open(os.path.join(td, "runner.log"), "w") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            print(f"[mmar-adapter] runner exited rc={rc}", file=sys.stderr)
            return rc
        if not os.path.isfile(runner_out):
            print("[mmar-adapter] runner produced no output file.", file=sys.stderr)
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
        print(f"[mmar-adapter] runner output missing 'accuracy': {raw}", file=sys.stderr)
        return 4
    # --- val/eval split (Evo-Bench) -------------------------------------------
    # run_mmar_official.py already subset by EVAL_SPLIT and computed
    # accuracy/correct/n over that subset (results re-indexed from 0). Do NOT
    # re-apply the split (would double-filter to ~9%/~49% instead of 30/70).
    # Label the task, and persist per-item diagnostics for the 'val' split only.
    _task = raw.get("task", "mmar")
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
            print(f"[mmar-adapter] wrote val diagnostics -> {_dp}", file=sys.stderr)
    except Exception as e:
        print(f"[mmar-adapter] diag dump skipped ({e})", file=sys.stderr)

    metrics = {"accuracy": float(acc), "task": _task,
               "correct": _correct, "n": _n}
    if _split == "eval":
        metrics.update(benchmark="mmar", eval_split=_split)
    os.makedirs(os.path.dirname(os.path.abspath(args.json_output_file)) or ".", exist_ok=True)
    with open(args.json_output_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[mmar-adapter] accuracy={metrics['accuracy']:.4f} "
          f"({metrics['correct']}/{metrics['n']}) -> {args.json_output_file}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
