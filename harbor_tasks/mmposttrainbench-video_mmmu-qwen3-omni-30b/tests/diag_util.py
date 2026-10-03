#!/usr/bin/env python3
"""Val-only per-item diagnostics, shared by every bench adapter.

Why: the agent's self-check (score.sh, EVAL_SPLIT=val) is far more useful when it
can see WHICH items it got wrong — the question, the gold answer, and the model's
prediction — instead of only a scalar accuracy. That lets the agent target its
data synthesis at real failure modes rather than blindly hill-climbing a number
(which is what pushes it toward overfitting / benchmaxing).

Isolation contract — this is the hard gate that keeps the seam safe:
    dump_val_diag writes ONLY when split == "val".
For the sealed "eval" split (verifier) and the whole-set case it writes nothing,
so the verifier never persists per-item content out of its namespace. The 30%
val is agent-visible by design; the 70% eval is not, ever.

Usage (from an adapter, after it has the runner's per-sample list):
    from diag_util import dump_val_diag
    dump_val_diag(records, out_dir, split)   # out_dir = dir of --json-output-file
"""
from __future__ import annotations

import json
import os


def _norm(pos: int, r: dict) -> dict:
    """Normalize one heterogeneous runner record to a diag row.

    Preserves the record's OWN identifier (idx / qid / instance_id) as `idx` so a
    wrong item traces back to the exact source sample — enumerating alone loses
    that (a record with idx=37 would otherwise be re-numbered to 0). `pos` keeps
    the ordinal position within this diag file. Media references (video/audio/
    image/url) and MC options are carried through when the runner supplies them,
    so an error is traceable to the exact media (and, via the id, its timestamp).
    Diagnostics-only; never affects scoring.

    Field names differ across benches:
      self-runners (mmau/mmar/jointav): question, gold, pred|resp, ok, qid
      omnivideobench:                   question, correct_answer, model_answer, is_correct
      lmms / mmswe:                     idx (doc_id) / instance_id
    """
    gold = r.get("gold", r.get("correct_answer"))
    pred = r.get("pred", r.get("resp", r.get("model_answer")))
    ok = r.get("ok")
    if ok is None:
        ok = r.get("is_correct")
    orig_id = r.get("idx")
    if orig_id is None:
        orig_id = r.get("qid", r.get("instance_id", r.get("id", pos)))
    out = {
        "idx": orig_id,   # original sample id (NOT the re-numbered ordinal)
        "pos": pos,       # ordinal position within this diag file
        "question": r.get("question"),
        "gold": gold,
        "pred": pred,
        "ok": (bool(ok) if ok is not None else None),
    }
    for mk in ("media", "video", "audio", "image", "media_path", "url", "options"):
        v = r.get(mk)
        if v is not None:
            out[mk] = v
    return out


def dump_val_diag(records, out_dir: str, split, filename: str = "diag.jsonl"):
    """Write a per-item diag.jsonl to out_dir — ONLY when split == 'val'.

    `records` is a runner's per-sample list (already restricted to the split's
    items, since the runners/adapters subset before scoring). Returns the path
    written, or None if nothing was written (wrong split, or no records).
    """
    if split != "val":
        return None
    if not isinstance(records, list) or not records:
        return None
    out_dir = out_dir or "."
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        for i, r in enumerate(records):
            if not isinstance(r, dict):
                continue
            f.write(json.dumps(_norm(i, r), ensure_ascii=False) + "\n")
    return path
