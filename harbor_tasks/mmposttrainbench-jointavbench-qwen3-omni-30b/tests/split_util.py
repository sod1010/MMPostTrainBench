#!/usr/bin/env python3
"""Deterministic val/eval split shared by every bench adapter/runner.

Evo-Bench principle: the agent may query only a VALIDATION subset (via score.sh);
the official held-out EVALUATION subset stays sealed (verifier only), and the two
are mutually exclusive. We assign each test item to a split purely from its
0-based index with a fixed seed, so the split is stable, needs no stored file,
and any code that can enumerate the dataset can apply it identically.

Usage in a scoring loop:
    from split_util import keep
    for idx, item in enumerate(dataset):
        if not keep(idx, os.environ.get("EVAL_SPLIT")):
            continue
        ... score item ...

EVAL_SPLIT is "val", "eval", or empty/None (=> whole set, backward compatible).
"""
from __future__ import annotations
import hashlib

SEED = 1234
VAL_FRAC = 0.30   # 30% validation (agent-visible), 70% sealed evaluation


def _unit(idx: int, seed: int = SEED) -> float:
    """Stable hash of index -> float in [0,1). Uses md5 (not Python hash, which
    is salted per process) so val/eval membership is identical everywhere."""
    h = hashlib.md5(f"{seed}:{int(idx)}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def split_of(idx: int, seed: int = SEED, val_frac: float = VAL_FRAC) -> str:
    return "val" if _unit(idx, seed) < val_frac else "eval"


def keep(idx: int, split, seed: int = SEED, val_frac: float = VAL_FRAC) -> bool:
    """True if item `idx` belongs to `split`. split None/"" /"all"/"full" => keep all."""
    if not split or split in ("all", "full", "none"):
        return True
    if split not in ("val", "eval"):
        raise ValueError("invalid evaluation split")
    return split_of(idx, seed, val_frac) == split


def resolve_split(requested, role_env: str = "MMPTB_ROLE") -> str:
    """Operator-only policy for the sealed 'eval' split (info-isolation Step 2).

    Consumers call this before launching inference. This prevents accidental
    misconfiguration; an environment variable is NOT an authorization boundary.
      * role 'verifier' (operator sets MMPTB_ROLE=verifier): honored as-is.
      * any other role (agent/default): 'eval' is refused and downgraded to 'val'
        (with an AUDIT line to stderr); a blank/whole request also defaults to 'val'.
    Returns the effective split string. Does NOT change the 30/70 membership itself
    (keep()/split_of() untouched); only which split a caller is allowed to run."""
    import os, sys
    sp = (requested or "").strip()
    if sp not in ("", "all", "full", "none", "val", "eval"):
        raise ValueError("invalid EVAL_SPLIT; expected val or eval")
    if os.environ.get(role_env, "agent") == "verifier":
        return "" if sp in ("all", "full", "none") else sp
    if sp == "eval":
        sys.stderr.write("[split_util] AUDIT: eval-split requested without verifier "
                         "role -> forcing val (sealed split is operator-only).\n")
        return "val"
    if sp in ("", "all", "full", "none"):
        return "val"
    return sp


def mmswe_config(split):
    """One dataset/split contract shared by generation and grading."""
    import os
    datasets = {os.environ[k] for k in ("SWE_DATASET", "MMSWE_DATASET") if os.environ.get(k)}
    if len(datasets) > 1:
        raise ValueError("SWE_DATASET and MMSWE_DATASET disagree")
    dataset = next(iter(datasets), "SWE-bench/SWE-bench_Multimodal")
    official = os.environ.get("MMSWE_SPLIT_BY_OFFICIAL", "1").lower() not in ("0", "", "false")
    declared = {os.environ[k] for k in ("SWE_SPLIT", "MMSWE_SPLIT") if os.environ.get(k)}
    want = {"val": "dev", "eval": "test"}.get(split) if official else None
    if len(declared) > 1 or (want and declared and declared != {want}):
        raise ValueError("MMSWE generation/grading split configuration disagrees")
    return dataset, want or next(iter(declared), "dev"), official


def recompute_ok(results, split, seed: int = SEED, val_frac: float = VAL_FRAC):
    """Recompute (accuracy, correct, n) over ONLY the items in `split`.

    `results` is an official runner's per-sample list (each a dict with a boolean
    'ok'), emitted in dataset order — so we select by 0-based index using the
    shared deterministic split. split None/""/"all"/"full"/"none" => whole list.
    Used by the mmau/mmar/jointavbench adapters (which shell out to omni runners
    that can't subset by index themselves)."""
    kept = [r for i, r in enumerate(results) if keep(i, split, seed, val_frac)]
    n = len(kept)
    correct = sum(1 for r in kept if r.get("ok"))
    return ((correct / n) if n else 0.0), correct, n


def summarize(n: int, seed: int = SEED, val_frac: float = VAL_FRAC) -> dict:
    v = sum(1 for i in range(n) if split_of(i, seed, val_frac) == "val")
    return {"n": n, "val": v, "eval": n - v, "val_frac": round(v / n, 4) if n else 0.0}


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    print(summarize(n))
