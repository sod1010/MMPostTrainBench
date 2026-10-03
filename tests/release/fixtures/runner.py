import json
import os
from pathlib import Path
import sys

def arg(name):
    return sys.argv[sys.argv.index(name) + 1]

mode = os.environ.get("FIXTURE_MODE", "ok")
print("Correct predictions: 19")
print("Accuracy: 95%")
print("SEALED_SENTINEL", file=sys.stderr)
if mode == "runner_fail":
    raise SystemExit(9)
out = Path(arg("--output_file" if "--output_file" in sys.argv else "--out"))
if mode == "no_output":
    raise SystemExit(0)
if mode == "bad_json":
    out.write_text("{broken")
    raise SystemExit(0)
from split_util import keep, mmswe_config
if "--output_file" in sys.argv:
    rows = [{"model_answer": "B", "correct_answer": "A", "is_correct": False,
             "question": "visible" if keep(i, "val") else "SEALED_SENTINEL"} for i in range(20)]
    if mode == "empty":
        rows = []
    if mode == "malformed":
        rows[0] = {}
    out.write_text(json.dumps(rows))
elif "--preds" in sys.argv:
    out.mkdir(exist_ok=True)
    ids = ["fixture__repo-1"]
    summary = {"dataset": arg("--dataset"), "split": arg("--split"), "n": 1,
               "resolved": 0, "resolved_ids": [], "instance_ids": ids}
    report = {ids[0]: {"resolved": False, "patch_successfully_applied": False,
                       "infra_failure": mode == "infra"}}
    if mode == "partial":
        report = {}
    if mode == "wrong_split":
        summary["split"] = "wrong"
    (out / "summary.json").write_text(json.dumps(summary))
    (out / "report.json").write_text(json.dumps(report))
    if mode == "grader_fail":
        raise SystemExit(8)
elif os.environ.get("FIXTURE_KIND") == "mmswe":
    dataset, split, _ = mmswe_config(os.environ["EVAL_SPLIT"])
    iid = "fixture__repo-1"
    out.write_text(json.dumps({"instance_id": iid, "model_patch": ""}) + "\n")
    meta = {"dataset": dataset, "split": split, "eval_split": os.environ["EVAL_SPLIT"],
            "n": 1, "instance_ids": [iid], "n_generation_failed": int(mode == "gen_failed")}
    (out.parent / "meta.json").write_text(json.dumps(meta))
else:
    rows = [{"idx": i, "ok": False, "question": "visible" if keep(i, "val") else "SEALED_SENTINEL"}
            for i in range(20) if keep(i, os.environ["EVAL_SPLIT"])]
    out.write_text(json.dumps({"task": "fixture", "accuracy": 0, "correct": 0,
                              "n": 0 if mode == "empty" else len(rows), "results": rows}))
