"""Small shared helpers for fail-closed adapter outputs."""
import math
import os
from pathlib import Path


def prepare_output(path):
    """Invalidate previous scores/diagnostics before any operation can fail."""
    output = Path(path)
    for p in (output, output.parent / "diag.jsonl"):
        p.unlink(missing_ok=True)


def configure_split():
    from split_util import resolve_split
    split = resolve_split(os.environ.get("EVAL_SPLIT"))
    os.environ["EVAL_SPLIT"] = split
    return split


def validate_counts(accuracy, correct, n):
    if type(n) is not int or n <= 0 or type(correct) is not int or not 0 <= correct <= n:
        raise ValueError("invalid or empty result counts")
    if type(accuracy) not in (int, float) or not math.isfinite(accuracy):
        raise ValueError("invalid accuracy")
    if not 0 <= accuracy <= 1 or abs(accuracy - correct / n) > 1e-8:
        raise ValueError("accuracy does not match result counts")


def validate_runner_results(raw):
    validate_counts(raw.get("accuracy"), raw.get("correct"), raw.get("n"))
    rows = raw.get("results")
    if not isinstance(rows, list) or len(rows) != raw["n"]:
        raise ValueError("incomplete per-item results")
    if any(not isinstance(r, dict) or type(r.get("ok")) is not bool for r in rows):
        raise ValueError("invalid per-item result")
    if sum(r["ok"] for r in rows) != raw["correct"]:
        raise ValueError("per-item results disagree with aggregate")
