#!/usr/bin/env python3
"""③ Anytime Validation Score — a PROCESS metric (Evo-Bench).

Overall (held-out) only measures the frozen final model. AnytimeVal measures how
efficiently the agent FOUND and KEPT good versions over its budget:

    S*_t = max_{i<=t} S(H_i on validation)          (best-so-far at iter t)
    AnytimeVal = (1 / b_iter) * sum_{t=1..b_iter} S*_t

Early termination carries the final best-so-far forward for the remaining budget.
It surfaces "early saturation / lost the best version": if the agent's FINAL
self-score is below its BEST-ever self-score, it found an improvement and then
regressed away from it (exactly what we saw: mmau stalled, jointav's best
checkpoint wasn't the frozen one).

Input: a trial workspace with `_score_<epoch>/reward.txt` dirs (the agent's
validation self-checks, ordered by the epoch in the dir name). Optionally a
b_iter budget (default = the validation-iteration cap, 20).

Usage: compute_anytime.py --workspace <ws> [--b-iter 20] [--out report.json]
"""
from __future__ import annotations
import argparse, glob, json, os, re


def collect_val_history(ws: str):
    """Return [(epoch, reward), ...] ordered by time from _score_* dirs."""
    hist = []
    for d in glob.glob(os.path.join(ws, "_score_*")):
        m = re.search(r"_score_(\d+)", os.path.basename(d))
        if not m:
            continue
        rf = os.path.join(d, "reward.txt")
        if not os.path.isfile(rf):
            continue
        try:
            r = float(open(rf).read().strip())
        except Exception:
            continue
        hist.append((int(m.group(1)), r))
    hist.sort(key=lambda x: x[0])
    return hist


def anytime(scores, b_iter: int):
    """scores: validation scores in iteration order. Returns (AnytimeVal, best_so_far_series)."""
    best = float("-inf"); series = []
    for t in range(b_iter):
        if t < len(scores):
            best = max(best, scores[t])
        # if past the run's evals, carry the final best-so-far forward
        series.append(best if best != float("-inf") else 0.0)
    return (sum(series) / b_iter if b_iter else 0.0), series


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--b-iter", type=int, default=20, help="validation-iteration budget")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    hist = collect_val_history(args.workspace)
    scores = [r for _, r in hist]
    av, series = anytime(scores, args.b_iter)
    best = max(scores) if scores else None
    final = scores[-1] if scores else None
    best_iter = (scores.index(best) + 1) if scores else None
    report = {
        "workspace": args.workspace,
        "n_val_evals": len(scores),
        "b_iter": args.b_iter,
        "anytime_validation": round(av, 4),
        "best_val": best, "best_iter": best_iter,
        "final_val": final,
        "lost_best": (best is not None and final is not None and final < best - 1e-9),
        "val_history": [round(s, 4) for s in scores],
    }
    js = json.dumps(report, indent=2)
    if args.out:
        open(args.out, "w").write(js)
    print(js)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
