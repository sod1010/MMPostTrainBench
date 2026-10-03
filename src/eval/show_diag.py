#!/usr/bin/env python3
"""Pretty-print a val self-check diagnostic file for the agent's console.

score.sh (EVAL_SPLIT=val) calls this after a self-check finishes:
    python show_diag.py <OUT/diag.jsonl>
It prints how many val items were wrong and a small sample of the failures
(question / gold / pred) so the agent can target real weaknesses instead of
hill-climbing a scalar. Kept as a standalone file so score.sh needs only a
one-line call (no nested here-doc escaping in the launcher).

Only ever reads the val diag file; the sealed eval split never produces one.
"""
import json
import sys

K = 8  # how many failing items to sample


def main():
    if len(sys.argv) < 2:
        return
    path = sys.argv[1]
    try:
        recs = [json.loads(l) for l in open(path) if l.strip()]
    except FileNotFoundError:
        return
    if not recs:
        return
    wrong = [r for r in recs if r.get("ok") is False]
    graded = [r for r in recs if r.get("ok") is not None]
    print(f"[val-diag] {len(wrong)}/{len(graded) or len(recs)} items WRONG on the val self-check "
          f"(this is the 30% you can see; the sealed 70% is graded separately).")
    if not wrong:
        print("  (no wrong items in this run)")
        return
    print(f"  sample of up to {K} failures — target these, don't just chase the number:")
    for r in wrong[:K]:
        q = (r.get("question") or "")
        q = " ".join(str(q).split())[:160]
        gold = r.get("gold")
        pred = str(r.get("pred"))
        if len(pred) > 60:
            pred = pred[:60] + "…"
        print(f"  - gold={gold!r}  pred={pred!r}")
        if q:
            print(f"      Q: {q}")


if __name__ == "__main__":
    main()
