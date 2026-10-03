#!/usr/bin/env python3
"""mmswe (SWE-bench Multimodal) evaluate.py — omni seam adapter.

Contract shim between mmptb's verifier (`python3 evaluate.py --model-path ...
--limit ... --json-output-file ...`) and the two-stage mmswe eval:

  stage 1  GENERATE  run_mmswe_official.py (omni_sft python, THIS interpreter):
           omni model -> predictions.jsonl {instance_id, model_patch}.
  stage 2  GRADE      dlc_native_grade.py (swebench venv, $SWE_VENV_PY): Route B
           (registry-v2 manifest -> mirror blob + sha256 verify -> unpack ->
           unshare+chroot -> apply patch + run tests -> get_eval_report). No
           docker daemon needed; requires a Linux host with working namespace/chroot permissions.

resolved_rate (resolved / n over the selected split subset) is normalized to the
flat {"accuracy": <0-1 float>} contract test.sh / assemble_verifier read. The
runner ALREADY applied EVAL_SPLIT + --limit, so we must NOT re-split here (same
double-filter trap the mmau adapter warns about) — grade exactly the preds.

Env (all operator paths; the defaults are the in-image eval-deliver layout):
  MMSWE_RUNNER   run_mmswe_official.py     (default /opt/eval/runners/...)
  SWE_WORK       grader home: holds venv/ (swebench), hf_cache/ (offline SWE
                 dataset) and blob_cache/ (verified docker layers)
  SWE_VENV_PY    $SWE_WORK/venv/bin/python            (swebench venv)
  MMSWE_GRADER   dlc_native_grade.py       (default: next to this file)
  SWE_DATASET    SWE-bench/SWE-bench_Multimodal
  SWE_SPLIT      derived from EVAL_SPLIT: val -> dev, eval -> test
  SWE_GRADE_TIMEOUT  per-instance grade timeout sec (default 1800)
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_EVAL = os.path.dirname(os.path.dirname(HERE))       # .../src/eval
DEFAULT_RUNNER = "/opt/eval/runners/run_mmswe_official.py"
DEFAULT_SWE_WORK = "/opt/eval/mmswe"
DEFAULT_GRADER = os.path.join(HERE, "dlc_native_grade.py")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="mmswe omni eval adapter")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--json-output-file", required=True)
    ap.add_argument("--limit", type=int, default=-1)
    # tolerated-but-ignored (text-task retry ladder passes these)
    ap.add_argument("--templates-dir", default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--max-connections", type=int, default=None)
    ap.add_argument("--gpu-memory-utilization", type=float, default=None)
    return ap.parse_known_args()[0]


def validate_run(predictions, meta, summary, report, dataset, split, eval_split):
    """Require complete, matching generation and grading sets; no partial score."""
    ids = [p["instance_id"] for p in predictions]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("missing or duplicate predictions")
    if any(not isinstance(i, str) or not i or "/" in i or i in (".", "..") for i in ids):
        raise ValueError("invalid instance id")
    if meta.get("dataset") != dataset or meta.get("split") != split or meta.get("eval_split") != eval_split:
        raise ValueError("generation split/dataset mismatch")
    if meta.get("instance_ids") != ids or meta.get("n") != len(ids):
        raise ValueError("incomplete generation output")
    if type(meta.get("n_generation_failed")) is not int or meta["n_generation_failed"] != 0:
        raise ValueError("generation failures or missing generation status")
    manifest = os.environ.get("MMSWE_IMAGE_MANIFEST")
    if os.environ.get("MMSWE_REQUIRE_FROZEN_IMAGES") == "1" and not manifest:
        raise ValueError("frozen image manifest required")
    if manifest:
        with open(manifest, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        pin = os.environ.get("MMSWE_IMAGE_MANIFEST_SHA256", digest)
        if digest != pin or meta.get("image_manifest_sha256") != digest:
            raise ValueError("generation image manifest mismatch")
        for p in predictions:
            if type(p.get("n_images")) is not int or p["n_images"] < 0 or not isinstance(p.get("image_sha256"), list) or p["n_images"] != len(p["image_sha256"]):
                raise ValueError("missing per-item media audit")
    if summary.get("dataset") != dataset or summary.get("split") != split:
        raise ValueError("grading split/dataset mismatch")
    if summary.get("instance_ids") != ids or summary.get("n") != len(ids) or set(report) != set(ids):
        raise ValueError("grading does not cover exactly the generated instances")
    for iid in ids:
        r = report[iid]
        if r.get("infra_failure") is not False or r.get("error"):
            raise ValueError("infrastructure failure or missing grading status")
        if type(r.get("resolved")) is not bool or type(r.get("patch_successfully_applied")) is not bool:
            raise ValueError("missing grading outcome")
        if r["resolved"] and not r["patch_successfully_applied"]:
            raise ValueError("unapplied patch cannot be resolved")
    resolved = {iid for iid in ids if report[iid]["resolved"]}
    claimed = summary.get("resolved_ids")
    if not isinstance(claimed, list) or len(claimed) != len(set(claimed)) or set(claimed) != resolved:
        raise ValueError("inconsistent resolved ids")
    if type(summary.get("resolved")) is not int or summary["resolved"] != len(resolved):
        raise ValueError("inconsistent resolved count")
    return ids, resolved


def main() -> int:
    args = parse_args()
    sys.path.insert(0, SRC_EVAL)
    from eval_util import prepare_output, configure_split
    prepare_output(args.json_output_file)
    try:
        from split_util import mmswe_config
        eval_split = configure_split()
        dataset, split, _ = mmswe_config(eval_split)
        if os.environ.get("MMSWE_REQUIRE_FROZEN_IMAGES") == "1" and not os.path.isfile(os.environ.get("MMSWE_IMAGE_MANIFEST", "")):
            raise ValueError("frozen image manifest required before generation")
        runner = os.environ.get("MMSWE_RUNNER", DEFAULT_RUNNER)
        swe_work = os.environ.get("SWE_WORK", DEFAULT_SWE_WORK)
        swe_py = os.environ.get("SWE_VENV_PY", f"{swe_work}/venv/bin/python")
        grader = os.environ.get("MMSWE_GRADER", DEFAULT_GRADER)
        for p in (runner, swe_py, grader):
            if not os.path.isfile(p):
                raise ValueError("missing MMSWE runtime dependency")
        # Each call gets a new private directory. A failed process cannot reuse
        # old predictions, summary.json, or per-instance logs from a previous call.
        with tempfile.TemporaryDirectory(prefix="mmptb-mmswe-") as td:
            gen_dir, grade_dir = os.path.join(td, "gen"), os.path.join(td, "grade")
            os.makedirs(gen_dir)
            os.makedirs(grade_dir)
            preds = os.path.join(gen_dir, "predictions.jsonl")
            cmd = [sys.executable, runner, "--model-path", args.model_path,
                   "--limit", str(args.limit), "--out", preds]
            gcmd = [swe_py, grader, "--preds", preds, "--out", grade_dir,
                    "--dataset", dataset, "--split", split,
                    "--scratch", os.path.join(os.environ.get("SWE_SCRATCH", "/dev/shm/mmswe"), os.path.basename(td)),
                    "--timeout", os.environ.get("SWE_GRADE_TIMEOUT", "1800")]
            jobs = os.environ.get("SWE_GRADE_JOBS", os.environ.get("MMSWE_GRADE_JOBS", ""))
            if jobs.strip():
                gcmd += ["--jobs", jobs.strip()]
            for stage, command in (("generation", cmd), ("grading", gcmd)):
                print(f"[mmswe-adapter] {stage} split={split}", flush=True)
                with open(os.path.join(td, stage + ".log"), "w") as log:
                    rc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
                if rc:
                    print(f"[mmswe-adapter] {stage} failed rc={rc}; no score", file=sys.stderr)
                    return rc
            with open(preds) as f:
                predictions = [json.loads(line) for line in f if line.strip()]
            def read(path):
                with open(path) as f:
                    return json.load(f)
            meta = read(os.path.join(gen_dir, "meta.json"))
            summary = read(os.path.join(grade_dir, "summary.json"))
            report = read(os.path.join(grade_dir, "report.json"))
            ids, resolved = validate_run(predictions, meta, summary, report, dataset, split, eval_split)
        results = [{"instance_id": iid, "question": iid, "gold": "resolved",
                    "pred": "resolved" if iid in resolved else "unresolved", "ok": iid in resolved}
                   for iid in ids]
        out_dir = os.path.dirname(os.path.abspath(args.json_output_file))
        os.makedirs(out_dir, exist_ok=True)
        from diag_util import dump_val_diag
        dump_val_diag(results, out_dir, eval_split)
        acc = len(resolved) / len(ids)
        metrics = {"accuracy": acc, "task": f"swe_bench_multimodal@{eval_split or 'all'}",
                   "correct": len(resolved), "n": len(ids)}
        if eval_split == "eval":
            metrics.update(benchmark="mmswe", eval_split=eval_split, dataset=dataset, dataset_split=split)
        if meta.get("image_manifest_sha256"):
            metrics.update(image_manifest_sha256=meta["image_manifest_sha256"],
                           n_input_images=sum(p["n_images"] for p in predictions),
                           n_without_images=sum(p["n_images"] == 0 for p in predictions))
        with open(args.json_output_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[mmswe-adapter] resolved {len(resolved)}/{len(ids)} = {acc:.4f}", flush=True)
        return 0
    except (ValueError, KeyError, TypeError, AttributeError, OSError, ImportError) as e:
        # Validation messages contain protocol reasons only, not sealed records.
        print(f"[mmswe-adapter] incomplete or invalid run ({type(e).__name__}); no score", file=sys.stderr)
        return 6


if __name__ == "__main__":
    raise SystemExit(main())
