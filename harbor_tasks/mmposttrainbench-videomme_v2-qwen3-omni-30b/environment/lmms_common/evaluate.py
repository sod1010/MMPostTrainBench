#!/usr/bin/env python3
"""Generalized lmms-eval seam adapter (shared by all lmms-eval-harness benches:
mmmu_pro, mmar, video_mmmu, videomme_v2).

Same verifier contract as the self-runner adapters (--model-path /
--json-output-file / --limit, tolerated extra flags), but the "runner" is the
lmms-eval harness. It:
  1. runs `python -m lmms_eval --model qwen3_omni --model_args pretrained=<m>,
     device_map=auto,attn_implementation=sdpa --tasks $LMMS_TASK --limit N
     --log_samples --output_path <tmp>`
  2. globs the newest `<tmp>/<model_sanitized>/<TS>_results.json` (lmms-eval
     does NOT write a fixed filename)
  3. reads results[task][f"{LMMS_METRIC},none"], divides by LMMS_DIVISOR
     (mmar reports a percentage -> divisor 100), writes {"accuracy": <0-1>}.

Per-bench config via env (set by bench_recipes.sh / run_verifier):
  LMMS_TASK      lmms-eval task name (e.g. mmmu_pro_standard, mmar,
                 video_mmmu_comprehension, videomme_v2)   [required]
  LMMS_METRIC    metric key before ',none' (default: mmmu_acc)
  LMMS_DIVISOR   divide the metric by this (default: 1;  mmar -> 100)
  LMMS_MODEL_ARGS  override the full --model_args string (optional)
Requires lmms_eval importable: set PYTHONPATH to the vendored
eval_omni/harness_repos/lmms-eval, run under an env with transformers 4.57 +
qwen_omni_utils (the omni-eval env works).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/eval
try:
    from split_util import keep as _split_keep
    _SPLIT_UTIL_OK = True
except Exception:
    _SPLIT_UTIL_OK = False
    def _split_keep(idx, split):  # KEEP-ALL is only safe when NO split is requested;
        return True               # main() FAILS CLOSED if a split is requested (see below).


def _mmmu_correct(ans, pred):
    """Replicate mmmu eval_multi_choice: exact-match of the parsed letter(s).
    parsed_pred is already normalized to a choice letter by process_results."""
    if isinstance(ans, list) and isinstance(pred, list):
        pairs = list(zip(ans, pred))
        return (sum(1.0 for g, p in pairs if g == p) / len(pairs)) if pairs else 0.0
    if isinstance(ans, list):
        return 1.0 if pred in ans else 0.0
    return 1.0 if str(ans).strip() == str(pred).strip() else 0.0


def _per_sample_correct(v):
    """Map one per-sample metric value to correctness in [0,1], or None if unknown.
    lmms-eval's --log_samples stores the process_results output under the metric
    name; for several omni tasks that value is a DICT, not a scalar:
      * mmmu_acc (mmmu_pro / video_mmmu)  -> {"answer","parsed_pred", ...}
      * videomme_v2_*_acc                 -> {"score": 0/1, ...}
    Older tasks log a bool/number directly. The previous code only handled the
    scalar case and fell through to a naive filtered_resps-vs-target string
    compare, which is wrong for these dict metrics (parsed letter vs raw sentence)
    and drove the split score to ~garbage."""
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        # videomme_v2_* : per-question 0/1 'score'
        s = v.get("score")
        if isinstance(s, (bool, int, float)):
            return float(s)
        # mmmu_acc : {answer, parsed_pred} exact-match
        if "answer" in v and "parsed_pred" in v:
            return _mmmu_correct(v.get("answer"), v.get("parsed_pred"))
        # generic correctness fields
        for kk in ("correct", "acc", "is_correct", "exact_match"):
            if isinstance(v.get(kk), (bool, int, float)):
                return float(v[kk])
    return None


def _recompute_over_split(td, task, metric, split):
    """Score the requested task only; missing/malformed rows invalidate the run."""
    hits = glob.glob(os.path.join(td, "**", f"*_samples_{task}.jsonl"), recursive=True)
    if len(hits) != 1:
        raise ValueError("expected one sample file for the requested task")
    vals, seen = [], set()
    with open(hits[0], encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            did = rec.get("doc_id")
            if type(did) is not int or did < 0 or did in seen:
                raise ValueError("missing, invalid or duplicate sample id")
            seen.add(did)
            if not _split_keep(did, split):
                continue
            c = _per_sample_correct(rec.get(metric))
            if c is None or not math.isfinite(c) or not 0 <= c <= 1:
                raise ValueError("requested sample metric is missing or invalid")
            vals.append(c)
    if not vals:
        raise ValueError("no scored samples in requested split")
    return sum(vals) / len(vals), len(vals), len(seen)


def _collect_val_diag(td, task, metric, split):
    """Best-effort per-item diagnostics for the 'val' split, from the samples
    jsonl (only present when --log_samples, i.e. when a split is active). Returns
    a list of {question, gold, pred, ok} restricted to the split's doc_ids, or []
    if unavailable. Same doc_id filter as _recompute_over_split, so the diag rows
    are exactly the items that produced the val accuracy."""
    if split != "val":
        return []
    hits = sorted(glob.glob(os.path.join(td, "**", f"*_samples_{task}.jsonl"), recursive=True),
                  key=os.path.getmtime)
    if not hits:
        hits = sorted(glob.glob(os.path.join(td, "**", "*_samples_*.jsonl"), recursive=True),
                      key=os.path.getmtime)
    if not hits:
        return []
    recs = []
    for line in open(hits[-1], errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        did = rec.get("doc_id")
        if did is None or not _split_keep(int(did), split):
            continue
        c = _per_sample_correct(rec.get(metric))
        fr = rec.get("filtered_resps")
        pred = (fr[0] if isinstance(fr, list) and fr else fr)
        doc = rec.get("doc") or {}
        question = None
        if isinstance(doc, dict):
            for qk in ("question", "query", "problem", "prompt"):
                if doc.get(qk):
                    question = doc[qk]
                    break
        # lmms-eval logs an EMPTY doc for several omni tasks (videomme_v2,
        # video_mmmu), so the human-readable question survives ONLY in the
        # templated prompt `input`. Fall back to it (truncated) so the val diag
        # panel isn't blank for those benches. Diagnostics-only: gold/pred/ok and
        # the accuracy are unaffected by this.
        if not question:
            inp = rec.get("input")
            if isinstance(inp, str) and inp.strip():
                question = inp.strip()[:500]
        recs.append({
            "idx": did,
            "question": question,
            "gold": rec.get("target"),
            "pred": str(pred)[:100] if pred is not None else None,
            "ok": (None if c is None else bool(c >= 0.5)),
        })
    return recs


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="lmms-eval omni seam adapter")
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
    from eval_util import prepare_output, configure_split
    prepare_output(args.json_output_file)
    if not _SPLIT_UTIL_OK:
        print("[lmms-adapter] split helper unavailable; refusing to score", file=sys.stderr)
        return 6
    try:
        split = configure_split()
        task = os.environ["LMMS_TASK"]
        metric = os.environ.get("LMMS_METRIC", "mmmu_acc")
        divisor = float(os.environ.get("LMMS_DIVISOR", "1") or "1")
        if not math.isfinite(divisor) or divisor <= 0:
            raise ValueError("invalid metric divisor")
        model_args = os.environ.get("LMMS_MODEL_ARGS",
            f"pretrained={args.model_path},device_map=auto,attn_implementation=sdpa")
        with tempfile.TemporaryDirectory() as td:
            cmd = [sys.executable, "-m", "lmms_eval", "--model", "qwen3_omni",
                   "--model_args", model_args, "--tasks", task, "--batch_size", "1",
                   "--output_path", td, "--log_samples"]
            if args.limit is not None and args.limit > 0:
                cmd += ["--limit", str(args.limit)]
            print(f"[lmms-adapter] running task={task} split={split or 'all'}", flush=True)
            # Raw harness output contains whole-pool scores and possibly per-item
            # content. Keep it private for both val and sealed calls, on failure too.
            with open(os.path.join(td, "runner.log"), "w") as log:
                rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
            if rc:
                print(f"[lmms-adapter] runner failed rc={rc}; no score", file=sys.stderr)
                return rc
            hits = glob.glob(os.path.join(td, "**", "*_results.json"), recursive=True)
            if len(hits) != 1:
                raise ValueError("missing or ambiguous aggregate output")
            with open(hits[0]) as f:
                res = json.load(f)
            aggregate = res["results"][task][f"{metric},none"]
            if type(aggregate) not in (int, float) or not math.isfinite(aggregate):
                raise ValueError("missing or invalid requested aggregate")
            acc, n, _ = _recompute_over_split(td, task, metric, split)
            # Keep the historical per-item split metric. Some native aggregates
            # use group weighting, so equality is not a universal validity gate.
            diag = _collect_val_diag(td, task, metric, split)
        key = f"{metric}@{split}" if split else f"{metric},none"
        metrics = {"accuracy": acc, "task": task, "raw_metric": key,
                   "raw_value": acc, "n": n}
        if split == "eval":
            metrics.update(eval_split=split, benchmark={
                "mmmu_pro_standard": "mmmu_pro", "video_mmmu_comprehension": "video_mmmu",
                "videomme_v2": "videomme_v2"}.get(task, task))
        out_dir = os.path.dirname(os.path.abspath(args.json_output_file))
        os.makedirs(out_dir, exist_ok=True)
        from diag_util import dump_val_diag
        dump_val_diag(diag, out_dir, split)
        with open(args.json_output_file, "w") as f:
            json.dump(metrics, f, indent=2, allow_nan=False)
        print(f"[lmms-adapter] split={split or 'all'} accuracy={acc:.4f} n={n}", flush=True)
        return 0
    except (ValueError, KeyError, TypeError, OSError, ImportError):
        # Do not echo malformed runner records or aggregate dictionaries.
        print("[lmms-adapter] invalid configuration or incomplete results; no score", file=sys.stderr)
        return 6


if __name__ == "__main__":
    raise SystemExit(main())
