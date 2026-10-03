#!/usr/bin/env python3
"""MMAU evaluate.py — omni seam adapter for mmposttrainbench.

This is the *contract shim* between PostTrainBench's verifier
(tests/test.sh, which calls `python3 evaluate.py --model-path ... --limit ...
--json-output-file ...`) and the harbor/eval-deliver omni evaluation stack
(the transformers-based `run_mmau_official.py` self-runner).

Unlike the text PostTrainBench tasks, MMAU cannot use inspect-ai + vLLM:
Qwen3-Omni is a multimodal MoE served through transformers, and MMAU items
carry audio. So instead of running inspect-ai here, we shell out to the
eval-deliver runner (which loads the omni model, feeds audio explicitly, and
does the official MMAU string_match scoring), then normalize its output to the
flat `{"accuracy": <0-1 float>}` contract that test.sh reads.

Accepted CLI (superset of what test.sh passes; unused flags are tolerated so
the same verifier orchestrator works for text and omni tasks):
    --model-path PATH        (required) HF dir of the model to evaluate
    --json-output-file PATH  where to write the normalized metrics JSON
    --limit N                sample count; N<=0 (e.g. -1) = full run
    --templates-dir DIR      ignored (chat template baked into the runner)
    --max-tokens / --max-new-tokens / --max-connections /
      --gpu-memory-utilization
                             ignored (runner fixes its own generation config)

The runner is located via $MMAU_RUNNER, else the standard eval-deliver path
inside the omni-eval image (/opt/eval/runners/run_mmau_official.py).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile


DEFAULT_RUNNER = "/opt/eval/runners/run_mmau_official.py"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="MMAU omni eval adapter")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--json-output-file", required=True)
    ap.add_argument("--limit", type=int, default=-1)
    # Tolerated-but-ignored flags (kept so test.sh's text-oriented retry
    # ladder can pass them without breaking the omni path).
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

    runner = os.environ.get("MMAU_RUNNER", DEFAULT_RUNNER)
    if not os.path.isfile(runner):
        print(f"[mmau-adapter] ERROR: runner not found at {runner}. "
              f"Set $MMAU_RUNNER or build the verifier image FROM the "
              f"eval-deliver omni-eval image.", file=sys.stderr)
        return 2

    # The runner writes its own rich JSON ({task, accuracy, correct, n,
    # results}); we point it at a temp file and re-emit only the normalized
    # contract so test.sh's accuracy extractor (and, later, the aggregation
    # layer's load_metrics) see a clean {"accuracy": ...}.
    with tempfile.TemporaryDirectory() as td:
        runner_out = os.path.join(td, "mmau_runner.json")
        cmd = [
            sys.executable, runner,
            "--model-path", args.model_path,
            "--limit", str(args.limit),
            "--out", runner_out,
        ]
        print(f"[mmau-adapter] running: {' '.join(cmd)}", flush=True)
        with open(os.path.join(td, "runner.log"), "w") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            print(f"[mmau-adapter] runner exited rc={rc}; not writing "
                  f"metrics (verifier will retry).", file=sys.stderr)
            return rc
        if not os.path.isfile(runner_out):
            print("[mmau-adapter] runner produced no output file.",
                  file=sys.stderr)
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
        print(f"[mmau-adapter] runner output missing 'accuracy': {raw}",
              file=sys.stderr)
        return 4

    # --- val/eval split (Evo-Bench) -------------------------------------------
    # The runner (run_mmau_official.py) ALREADY subset the dataset by EVAL_SPLIT
    # and computed accuracy/correct/n over that subset, returning per-sample
    # 'results' re-indexed from 0. So we must NOT re-apply the split here — doing
    # so double-filters (val ~= 0.30*0.30 ~= 9%, eval ~= 49%) instead of 30/70.
    # Just label the task with the split, and — for the agent-visible 'val'
    # split ONLY — persist per-item diagnostics next to metrics.json.
    _task = raw.get("task", "mmau_test_mini")
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
            print(f"[mmau-adapter] wrote val diagnostics -> {_dp}", file=sys.stderr)
    except Exception as e:
        print(f"[mmau-adapter] diag dump skipped ({e})", file=sys.stderr)

    metrics = {
        "accuracy": float(acc),
        "task": _task,
        "correct": _correct,
        "n": _n,
    }
    if _split == "eval":
        metrics.update(benchmark="mmau", eval_split=_split)
    os.makedirs(os.path.dirname(os.path.abspath(args.json_output_file)) or ".",
                exist_ok=True)
    with open(args.json_output_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[mmau-adapter] accuracy={metrics['accuracy']:.4f} "
          f"({metrics['correct']}/{metrics['n']}) -> {args.json_output_file}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
