#!/usr/bin/env python3
"""Aggregate the 8 task rewards into a single mmposttrainbench score.

Mirrors the text PostTrainBench metric: the agent's post-trained model is scored
on each benchmark; the headline is a WEIGHTED MEAN across benchmarks (weights in
factors.json), and — when baselines.json is filled with the base model's
full-sample scores — the IMPROVEMENT over baseline (post-trained − base).

Inputs: a results dir with per-bench <bench>/metrics.json (or <bench>/reward.txt),
as produced by the broker/verifier. Usage:
    python aggregate.py --results-dir <dir> [--baselines baselines.json]
      [--factors factors.json] [--out summary.json]
Each <bench> subdir name must match a factors.json benchmark key.
"""
from __future__ import annotations
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load_score(bench_dir: str):
    """Return the [0,1] accuracy for a bench result dir, or None."""
    mj = os.path.join(bench_dir, "metrics.json")
    if os.path.isfile(mj):
        try:
            return float(json.load(open(mj)).get("accuracy"))
        except Exception:
            pass
    rt = os.path.join(bench_dir, "reward.txt")
    if os.path.isfile(rt):
        try:
            return float(open(rt).read().strip())
        except Exception:
            pass
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True,
                    help="dir with <bench>/metrics.json (or reward.txt) per bench")
    ap.add_argument("--factors", default=os.path.join(HERE, "factors.json"))
    ap.add_argument("--baselines", default=os.path.join(HERE, "baselines.json"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    factors = json.load(open(args.factors))["benchmarks"]
    baselines = {}
    if os.path.isfile(args.baselines):
        baselines = (json.load(open(args.baselines)).get("scores") or {})

    per_bench, wsum, acc_w = {}, 0.0, 0.0
    imp_w, imp_wsum = 0.0, 0.0
    for bench, fac in factors.items():
        s = load_score(os.path.join(args.results_dir, bench))
        w = float(fac.get("weight", 1.0))
        base = baselines.get(bench)
        entry = {"score": s, "weight": w, "group": fac.get("group"), "baseline": base}
        if s is not None:
            entry["improvement"] = (s - base) if isinstance(base, (int, float)) else None
            acc_w += w * s; wsum += w
            if isinstance(base, (int, float)):
                imp_w += w * (s - base); imp_wsum += w
        else:
            entry["improvement"] = None
            entry["missing"] = True
        per_bench[bench] = entry

    # per-modality means (over present benches)
    groups = {}
    for b, e in per_bench.items():
        if e["score"] is None:
            continue
        g = e["group"] or "other"
        groups.setdefault(g, []).append(e["score"])
    by_group = {g: round(sum(v) / len(v), 4) for g, v in groups.items()}

    summary = {
        "aggregate_score": round(acc_w / wsum, 4) if wsum else None,
        "aggregate_improvement_over_baseline": round(imp_w / imp_wsum, 4) if imp_wsum else None,
        "n_benches_scored": sum(1 for e in per_bench.values() if e["score"] is not None),
        "n_benches_total": len(factors),
        "by_modality": by_group,
        "per_bench": per_bench,
    }
    out = args.out or os.path.join(args.results_dir, "aggregate_summary.json")
    json.dump(summary, open(out, "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\n-> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
