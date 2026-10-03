#!/usr/bin/env python3
"""OmniVideoBench evaluate.py — omni seam adapter (official-script shape).

OmniVideoBench uses its own official runner (OmniVideoBench/eval/qwen3_omni_eval.py,
use_audio_in_video=True, device_map=auto, 512 frames) which writes a FLAT JSON
LIST of per-item dicts ({video,question,correct_answer,model_answer,is_correct})
with NO aggregate metric. This adapter shells to it, then RECOMPUTES accuracy =
mean(is_correct) over *valid* items (excluding items whose model_answer starts
with 'Video file not found' or 'Error:' — e.g. missing video files), and emits
the flat {"accuracy": <0-1>} contract.

env overrides:
  OVB_RUNNER     path to qwen3_omni_eval.py
  OVB_DATA       data.json
  OVB_VIDEO_DIR  dir holding <video>.mp4 (data.json 'video' field, no ext)
  OVB_MAX_DURATION  seconds cap (default 6000)
--limit N -> OVB_LIMIT (the runner truncates QA pairs for a smoke).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

DEFAULT_RUNNER = os.environ.get("OVB_RUNNER", "/opt/eval/harness_repos/OmniVideoBench/eval/qwen3_omni_eval.py")
# OmniVideoBench data/media (env-driven; default under DATA_DIR).
_DATA_DIR = os.environ.get("DATA_DIR", "data")
DEFAULT_DATA = os.path.join(_DATA_DIR, "evaluationbench", "OmniVideoBench_local", "data.json")
DEFAULT_VIDEO_DIR = os.path.join(_DATA_DIR, "OmniVideoBench", "videos_local")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="OmniVideoBench omni eval adapter")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--json-output-file", required=True)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--templates-dir", default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--max-connections", type=int, default=None)
    ap.add_argument("--gpu-memory-utilization", type=float, default=None)
    return ap.parse_known_args()[0]


def _valid(r: dict) -> bool:
    ma = (r.get("model_answer") or "")
    return not ma.startswith(("Video file not found", "Error:"))


def main() -> int:
    args = parse_args()
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from eval_util import prepare_output, configure_split
    prepare_output(args.json_output_file)
    try:
        from split_util import keep as _keep
        _SPLIT = configure_split()
    except (ValueError, ImportError):
        print("[ovb-adapter] invalid split configuration; no score", file=sys.stderr)
        return 6
    runner = os.environ.get("OVB_RUNNER", DEFAULT_RUNNER)
    data = os.environ.get("OVB_DATA", DEFAULT_DATA)
    video_dir = os.environ.get("OVB_VIDEO_DIR", DEFAULT_VIDEO_DIR)
    max_dur = os.environ.get("OVB_MAX_DURATION", "6000")
    for p, what in ((runner, "runner"), (data, "data json")):
        if not os.path.isfile(p):
            print(f"[ovb-adapter] ERROR: {what} not found at {p}", file=sys.stderr)
            return 2

    env = dict(os.environ)
    if args.limit is not None and args.limit > 0:
        env["OVB_LIMIT"] = str(args.limit)

    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "ovb.json")
        cmd = [sys.executable, runner,
               "--data_json_file", data,
               "--video_dir", video_dir,
               "--model_path", args.model_path,
               "--output_file", out,
               "--max_duration", str(max_dur)]
        print(f"[ovb-adapter] running: {' '.join(cmd)}", flush=True)
        # Use the effective split established BEFORE inference. Raw runner output
        # remains private, including when the runner fails or the role is verifier.
        with open(os.path.join(td, "runner.log"), "w") as log:
            rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            print(f"[ovb-adapter] runner exited rc={rc}", file=sys.stderr)
            return rc
        if not os.path.isfile(out):
            print("[ovb-adapter] runner produced no output file.", file=sys.stderr)
            return 3
        with open(out) as f:
            results = json.load(f)

    if not isinstance(results, list):
        print(f"[ovb-adapter] expected a list, got {type(results)}", file=sys.stderr)
        return 4
    if any(not isinstance(r, dict) or type(r.get("is_correct")) is not bool
           or not isinstance(r.get("model_answer"), str) for r in results):
        print("[ovb-adapter] malformed runner records; no score", file=sys.stderr)
        return 5
    valid = [dict(r, idx=idx) for idx, r in enumerate(results) if _valid(r) and _keep(idx, _SPLIT)]
    if _SPLIT:
        print(f"[ovb-adapter] EVAL_SPLIT={_SPLIT} -> {len(valid)} valid items", file=sys.stderr)
    n = len(valid)
    correct = sum(1 for r in valid if r.get("is_correct"))
    # Fail closed: 0 valid items out of a non-empty result set means the scorer
    # parsed no answers at all (a harness break, e.g. the .sequences decode bug),
    # NOT a genuine model score of 0. Emitting accuracy=0.0 here would look like a
    # real reward. Refuse instead so the verifier surfaces the failure. (correct=0
    # with n>0 is a legitimate real 0.0 and is still allowed through.)
    if n == 0:
        print(f"[ovb-adapter] FATAL: 0 valid items in requested split "
              f"-- scorer produced no parseable answers; refusing to emit a fake "
              f"accuracy=0.0 (this is a harness failure, not a model score).",
              file=sys.stderr)
        return 5
    acc = (correct / n) if n else 0.0
    metrics = {"accuracy": acc, "task": "omnivideobench",
               "correct": correct, "n": n}
    if _SPLIT == "eval":
        metrics.update(benchmark="omnivideobench", eval_split=_SPLIT)
    # LEAK FIX: on the agent-visible val split, do NOT expose n_total (full-set
    # size reveals the held-out items were run). Eval/sealed keeps it for auditing.
    if _SPLIT != "val":
        metrics["n_total"] = len(results)
    # per-item diagnostics for the agent-visible 'val' split ONLY (never eval).
    # NOTE: OVB's split is applied exactly once (line above) — the runner does
    # NOT self-split — so this is the correct single-application path; we only
    # add diagnostics here, we do not re-split.
    _out_dir = os.path.dirname(os.path.abspath(args.json_output_file)) or "."
    try:
        from diag_util import dump_val_diag
        _dp = dump_val_diag(valid, _out_dir, _SPLIT)
        if _dp:
            print(f"[ovb-adapter] wrote val diagnostics -> {_dp}", file=sys.stderr)
    except Exception as e:
        print(f"[ovb-adapter] diag dump skipped ({e})", file=sys.stderr)
    os.makedirs(os.path.dirname(os.path.abspath(args.json_output_file)) or ".", exist_ok=True)
    with open(args.json_output_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[ovb-adapter] accuracy={acc:.4f} ({correct}/{n} valid) "
          f"-> {args.json_output_file}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
