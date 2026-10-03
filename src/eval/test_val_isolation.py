#!/usr/bin/env python3
"""Check agent-visible output; optional manifest check is a separate guarantee.

PASS_OUTPUT verifies the supported output contract and known leak signatures.
It does not prove val-only inference or filesystem isolation. Use
--require-val-only-inference with an evaluator-owned manifest for that check.
"""
import argparse
import json
import math
from pathlib import Path
import re

PATTERNS = [
    r"['\"]Overall['\"]\s*:\s*\{['\"]num['\"]\s*:",
    r"SELF-CHECK recompute-over-all",
    r"over\s+\d+/\d+\s+samples",
    r"\bn_total\b",
    r"(?im)^\s*Correct predictions\s*:",
    r"(?im)^\s*Accuracy\s*:",
    r"\d+\s+total\)",
]
ALLOWED = {"accuracy", "task", "raw_metric", "raw_value", "n", "correct"}


def inspect_output(out, manifest=None, require_manifest=False):
    out = Path(out)
    failures = []
    try:
        if not (out / "run.log").read_text().strip():
            failures.append("empty run.log")
        metrics = json.loads((out / "metrics.json").read_text())
        if not isinstance(metrics, dict) or set(metrics) - ALLOWED:
            raise ValueError("unexpected metric schema")
        acc, n = metrics.get("accuracy"), metrics.get("n")
        if type(acc) not in (int, float) or not math.isfinite(acc) or not 0 <= acc <= 1:
            raise ValueError("invalid accuracy")
        if type(n) is not int or n <= 0:
            raise ValueError("missing or empty sample count")
        if "correct" in metrics:
            from eval_util import validate_counts
            validate_counts(acc, metrics["correct"], n)
        if "raw_value" in metrics and metrics["raw_value"] != acc:
            raise ValueError("raw value disagrees with val accuracy")
        if "@eval" in metrics.get("task", "") or "@eval" in metrics.get("raw_metric", ""):
            raise ValueError("sealed result in val output")
    except (OSError, ValueError, TypeError):
        failures.append("missing, malformed or incomplete output artifacts")
    for path in out.rglob("*"):
        if path.is_file() and path.suffix in (".log", ".json", ".jsonl", ".txt", ".out", ".err"):
            try:
                content = path.read_text()
            except (OSError, UnicodeError):
                failures.append(f"unreadable artifact: {path.name}")
                continue
            if any(re.search(p, content) for p in PATTERNS):
                failures.append(f"known leak signature: {path.name}")
    if manifest:
        try:
            from split_util import keep
            ids = json.loads(Path(manifest).read_text())["doc_ids"]
            if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 or not keep(i, "val") for i in ids):
                raise ValueError("manifest is not val-only")
        except (OSError, ValueError, KeyError, TypeError):
            failures.append("invalid evaluator run manifest")
    elif require_manifest:
        failures.append("val-only inference unverified: no evaluator run manifest")
    return failures


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output")
    ap.add_argument("--manifest")
    ap.add_argument("--require-val-only-inference", action="store_true")
    args = ap.parse_args()
    failures = inspect_output(args.output, args.manifest, args.require_val_only_inference)
    print("FAIL" if failures else "PASS_OUTPUT")
    for reason in failures:
        print(reason)
    if not args.manifest:
        print("Val-only inference is UNVERIFIED; this check covers output only.")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
