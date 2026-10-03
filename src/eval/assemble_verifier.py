#!/usr/bin/env python3
"""Assemble a triangulated verifier_score.json for ONE loop from its 3 completed
eval-leg out dirs ($ws/verify_eval, $ws/verify_val, $ws/verify_ood).

Operator-side: reads baselines.json for the target + probe real name, but records
the probe ONLY under its codename (agent-facing output never names the probe).
Same schema as verify_eval_split.sh's inline assembler; extracted here so the batch
packed submitter (verify_pool.sh) can reuse it instead of duplicating the heredoc.

Usage: assemble_verifier.py <ws> <target_bench> <probe_bench> <ood_codename> <eps>
  reads   <ws>/verify_{eval,val,ood}/{reward.txt,metrics.json,.job}, <ws>/anytime.json
  writes  <ws>/verifier_score.json + <ws>/verifier_score.txt
"""
import json
import os
import sys

REPO_ROOT = os.environ.get(
    "REPO_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_metrics(d):
    try:
        return json.load(open(os.path.join(d, "metrics.json")))
    except Exception:
        return {}


def read_reward(d):
    """Prefer the reward.txt scalar; fall back to metrics.json['accuracy']."""
    try:
        v = open(os.path.join(d, "reward.txt")).read().strip()
        if v:
            return v
    except Exception:
        pass
    a = load_metrics(d).get("accuracy")
    return str(a) if a is not None else None


def read_file(p):
    try:
        return open(p).read().strip() or None
    except Exception:
        return None


def delta(a, b):
    a, b = num(a), num(b)
    return round(a - b, 4) if (a is not None and b is not None) else None


def main():
    if len(sys.argv) < 6:
        print("usage: assemble_verifier.py <ws> <bench> <probe> <ood_code> <eps>",
              file=sys.stderr)
        return 2
    ws, bench, probe, ood_code, eps = sys.argv[1:6]
    eps = num(eps) or 0.02

    try:
        base = json.load(open(f"{REPO_ROOT}/src/eval/baselines.json"))["scores"]
    except Exception:
        base = {}
    eval_b, ood_b = base.get(bench), base.get(probe)

    eval_dir, val_dir, ood_dir = (f"{ws}/verify_eval", f"{ws}/verify_val",
                                  f"{ws}/verify_ood")
    eval_r, val_r, ood_r = read_reward(eval_dir), read_reward(val_dir), read_reward(ood_dir)
    eval_reward, val_reward, ood_reward = num(eval_r), num(val_r), num(ood_r)
    eval_delta, ood_delta = delta(eval_r, eval_b), delta(ood_r, ood_b)
    gap = (round(val_reward - eval_reward, 4)
           if (val_reward is not None and eval_reward is not None) else None)

    anytime = None
    try:
        anytime = json.load(open(f"{ws}/anytime.json"))
    except Exception:
        pass

    # SOFT benchmax signal: target improved but the cross-modality probe regressed.
    benchmax_flag, reason = False, None
    if (eval_delta is not None and ood_delta is not None
            and eval_delta > 0 and ood_delta < -eps):
        benchmax_flag = True
        reason = (f"target eval improved (delta={eval_delta:+.4f}) but OOD probe "
                  f"{ood_code} regressed (delta={ood_delta:+.4f}) beyond -{eps}: "
                  f"likely distributional benchmax / narrow overfit rather than a "
                  f"real capability gain. (soft flag — no auto-rollback)")

    em, vm, om = load_metrics(eval_dir), load_metrics(val_dir), load_metrics(ood_dir)
    rec = {
        "bench": bench,
        "eval": {"reward": eval_reward, "baseline": num(eval_b),
                 "delta": eval_delta, "n": em.get("n"), "job": read_file(f"{eval_dir}/.job")},
        "val": {"reward": val_reward, "n": vm.get("n"), "job": read_file(f"{val_dir}/.job")},
        "gap": gap,
        "ood": {"codename": ood_code, "reward": ood_reward, "baseline": num(ood_b),
                "delta": ood_delta, "n": om.get("n"), "job": read_file(f"{ood_dir}/.job")},
        "anytime": (anytime.get("anytime_validation") if isinstance(anytime, dict) else None),
        "anytime_detail": anytime,
        "benchmax_flag": benchmax_flag,
        "benchmax_reason": reason,
    }
    json.dump(rec, open(f"{ws}/verifier_score.json", "w"), indent=2)
    open(f"{ws}/verifier_score.txt", "w").write((eval_r or "") + "\n")
    print(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
