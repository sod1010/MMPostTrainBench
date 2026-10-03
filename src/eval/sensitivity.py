#!/usr/bin/env python3
"""① Harness/Post-training Sensitivity of each bench (Evo-Bench Sens/Perf).

Question: which of our benches actually RESPOND to post-training quality? A bench
whose score doesn't move with the quality of the trained model is a bad target
(the agent can't win there — cf. mmar only degrading, mmau/jointav hugging base).

Given a score matrix M[variant][bench] (each variant = a differently-post-trained
Qwen3-Omni-30B, incl. the base), we compute per bench, following Evo-Bench:

    Sens(b) = Pearson corr( {M[v][b]}_v , {Q^(-b)_v}_v )
    Perf(b) = mean_v M[v][b]              (difficulty = 1 - Perf)

where Q^(-b)_v = leave-one-bench-out mean quality of variant v (its avg score
over all benches except b) — a robust estimate of "overall variant quality".
Sens(b) > 0 means the bench's score tracks variant quality (good target);
Sens(b) <= 0 means it doesn't respond (drop / deprioritize).

Input: a JSON score matrix {"<variant>": {"<bench>": score, ...}, ...}.
Usage: sensitivity.py --matrix scores.json [--out sens.json]
"""
from __future__ import annotations
import argparse, json, math


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n; my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx > 1e-12 and dy > 1e-12 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True, help="JSON {variant:{bench:score}}")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    M = json.load(open(args.matrix))
    variants = list(M)
    benches = sorted({b for v in M.values() for b in v})

    out = {}
    for b in benches:
        # variants that have a score for this bench
        vs = [v for v in variants if b in M[v]]
        if len(vs) < 2:
            out[b] = {"sens": None, "perf": None, "n_variants": len(vs), "note": "need >=2 variants"}
            continue
        task_scores = [M[v][b] for v in vs]
        # leave-one-bench-out overall quality per variant
        quality = []
        for v in vs:
            others = [M[v][bb] for bb in benches if bb != b and bb in M[v]]
            quality.append(sum(others) / len(others) if others else 0.0)
        sens = pearson(task_scores, quality)
        perf = sum(task_scores) / len(task_scores)
        out[b] = {"sens": round(sens, 4), "perf": round(perf, 4),
                  "difficulty": round(1 - perf, 4), "n_variants": len(vs),
                  "responsive": sens > 0}

    report = {"benches": out,
              "drop_candidates": [b for b, d in out.items() if d.get("sens") is not None and d["sens"] <= 0],
              "keep": [b for b, d in out.items() if d.get("responsive")],
              "variants": variants}
    js = json.dumps(report, indent=2)
    if args.out:
        open(args.out, "w").write(js)
    print(js)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
