"""CPU-only harness fixture; deliberately emits both stdout and stderr leaks."""
import json
import os
from pathlib import Path
import sys

def arg(name):
    return sys.argv[sys.argv.index(name) + 1]

task = arg("--tasks")
metric = os.environ["LMMS_METRIC"]
out = Path(arg("--output_path"))
mode = os.environ.get("FIXTURE_MODE", "ok")
print("{'Overall': {'num': 20, 'acc': 0.95}}")
print("SEALED_SENTINEL", file=sys.stderr)
if mode == "runner_fail":
    raise SystemExit(9)
if mode == "no_output":
    raise SystemExit(0)
aggregate = {"results": {task: {metric + ",none": 0}}}
if mode == "wrong_metric":
    aggregate["results"][task] = {"unrelated,none": 0.9}
(out / "fixture_results.json").write_text(json.dumps(aggregate))
if mode == "no_samples":
    raise SystemExit(0)
from split_util import keep
rows = [{"doc_id": i, metric: {"score": 0}, "target": "A", "filtered_resps": ["B"],
         "input": "visible question" if keep(i, "val") else "SEALED_SENTINEL"} for i in range(20)]
val_row = next(r for r in rows if keep(r["doc_id"], "val"))
if mode == "missing_metric":
    del val_row[metric]
if mode == "duplicate":
    rows.append(rows[0])
if mode == "nan":
    val_row[metric] = float("nan")
name = "wrong_task" if mode == "wrong_task" else task
with (out / f"fixture_samples_{name}.jsonl").open("w") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")
    if mode == "bad_json":
        f.write("{broken\n")
