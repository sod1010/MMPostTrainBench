#!/usr/bin/env python3
"""Contamination check — did the agent's TRAINING data overlap the eval TEST set?

Programmatic anti-cheat (complements the LLM judge): builds word-shingle
(n-gram) sets from the training corpus and, for each eval test item, computes
CONTAINMENT = |shingles(item) ∩ shingles(train)| / |shingles(item)|. An item
with containment >= --item-threshold is "contaminated" (its text is largely
present in training). If the contaminated FRACTION of eval items exceeds
--flag-fraction, the run is flagged. Mirrors MM-PostTrainBench's shingle
containment idea; text-only (media hashes are a later add).

Inputs are line-based: each line is a record. jsonl records are flattened to
their string values; plain text is used as-is. Usage:
  python contamination.py --train <train.jsonl|txt> [--train ...] \
      --eval <eval_questions.jsonl|txt> [--n 8] [--item-threshold 0.5] \
      [--flag-fraction 0.05] [--out report.json]
Exit 0 always; read report.json / stdout "flagged" for the decision.
"""
from __future__ import annotations
import argparse, json, os, re, sys

_WORD = re.compile(r"\w+", re.UNICODE)


def _texts_from_line(line: str):
    line = line.strip()
    if not line:
        return []
    if line[0] in "{[":
        try:
            obj = json.loads(line)
            out = []
            def walk(x):
                if isinstance(x, str): out.append(x)
                elif isinstance(x, dict):
                    for v in x.values(): walk(v)
                elif isinstance(x, (list, tuple)):
                    for v in x: walk(v)
            walk(obj)
            return out
        except Exception:
            return [line]
    return [line]


def shingles(text: str, n: int):
    toks = _WORD.findall(text.lower())
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def load_shingles(files, n):
    s = set()
    for f in files:
        if not os.path.isfile(f):
            print(f"[contam] WARN train file missing: {f}", file=sys.stderr); continue
        with open(f, errors="replace") as fh:
            for line in fh:
                for t in _texts_from_line(line):
                    s |= shingles(t, n)
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="append", default=[], required=True,
                    help="agent training data file(s) (jsonl/txt); repeatable")
    ap.add_argument("--eval", required=True, help="eval test questions (jsonl/txt)")
    ap.add_argument("--n", type=int, default=8, help="shingle size (words)")
    ap.add_argument("--item-threshold", type=float, default=0.5,
                    help="containment >= this => that eval item is contaminated")
    ap.add_argument("--flag-fraction", type=float, default=0.05,
                    help="flag if ANY single bench's contaminated fraction exceeds this")
    ap.add_argument("--worst-threshold", type=float, default=0.8,
                    help="flag if any single eval item is contained this much in "
                         "train (near-verbatim reuse), regardless of fraction")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    train = load_shingles(args.train, args.n)
    n_items = 0; contaminated = 0; worst = 0.0
    # Per-bench tallies so a single-bench hit isn't diluted by the combined
    # all-bench reference. Records may carry a "bench" field (see
    # build_contam_reference.py); otherwise everything lands in "_all".
    per = {}  # bench -> [n_items, contaminated, worst]
    if os.path.isfile(args.eval) and train:
        with open(args.eval, errors="replace") as fh:
            for line in fh:
                bench = "_all"
                s = line.strip()
                if s[:1] == "{":
                    try:
                        bench = json.loads(s).get("bench", "_all")
                    except Exception:
                        pass
                texts = _texts_from_line(line)
                item = set()
                for t in texts:
                    item |= shingles(t, args.n)
                if not item:
                    continue
                n_items += 1
                cont = len(item & train) / len(item)
                worst = max(worst, cont)
                p = per.setdefault(bench, [0, 0, 0.0])
                p[0] += 1; p[2] = max(p[2], cont)
                if cont >= args.item_threshold:
                    contaminated += 1; p[1] += 1
    frac = (contaminated / n_items) if n_items else 0.0
    per_bench = {b: {"eval_items": v[0], "contaminated": v[1],
                     "fraction": round(v[1] / v[0], 4) if v[0] else 0.0,
                     "worst": round(v[2], 4)} for b, v in per.items()}
    worst_bench_frac = max((v["fraction"] for v in per_bench.values()), default=0.0)
    # FLAG if any single bench crosses the fraction OR any item is near-verbatim.
    flagged = (worst_bench_frac > args.flag_fraction) or (worst >= args.worst_threshold)
    report = {"flagged": flagged, "contaminated_fraction": round(frac, 4),
              "worst_bench_fraction": round(worst_bench_frac, 4),
              "contaminated_items": contaminated, "eval_items": n_items,
              "worst_item_containment": round(worst, 4),
              "worst_threshold": args.worst_threshold, "flag_fraction": args.flag_fraction,
              "per_bench": per_bench, "shingle_n": args.n,
              "train_shingles": len(train)}
    js = json.dumps(report, indent=2)
    if args.out: open(args.out, "w").write(js)
    print(js)
    print(f"CONTAM_FLAGGED={flagged}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
