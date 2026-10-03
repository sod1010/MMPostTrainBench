#!/usr/bin/env python3
"""Build the combined contamination reference across ALL 8 benches.

The anti-cheat gate needs a single eval-questions file to check the agent's
training data against. Since an agent with open networking could download ANY of
our benches off HuggingFace, we flag overlap with EVERY eval test set, not just
the target bench. This walks each bench's cached test data and emits one text
line per eval item to `contam_ref_all.jsonl` (consumed by contamination.py).

Only TEXT is extracted (questions/choices/answers). Audio/video/image byte
columns are skipped — contamination is a text-shingle check. Run with a python
that has pyarrow (e.g. the omni-eval env):

  python build_contam_reference.py --out src/eval/contam_ref_all.jsonl

To add ONE bench to an existing reference without rebuilding the others (their
source caches may no longer be mounted), use --only + --append:

  python build_contam_reference.py --only mmswe --append --out <existing.jsonl>

Paths are env-driven (see config.env.example):
  HF_CACHE_DIR   HF hub cache root (arrow datasets live under $HF_CACHE_DIR/datasets)
  EVAL_BENCH_DIR dir holding JointAVBench/ and OmniVideoBench_local/ (default $DATA_DIR)
  SWE_HF_CACHE   HF cache holding SWE-bench_Multimodal (default $HF_CACHE_DIR)
"""
from __future__ import annotations
import argparse, glob, json, os, sys

HF = os.path.join(os.environ.get("HF_CACHE_DIR", os.path.expanduser("~/hf_cache")), "datasets")
EB = os.environ.get("EVAL_BENCH_DIR", os.path.join(os.environ.get("DATA_DIR", "data"), "evaluationbench"))

# bench -> cached HF arrow dataset dir (text columns auto-detected)
ARROW_BENCHES = {
    "mmau":        f"{HF}/lmms-lab-audio___mmau",
    "mmar":        f"{HF}/ngqtrung___mmar",
    "mmmu_pro":    f"{HF}/MMMU___mmmu_pro",
    "video_mmmu":  f"{HF}/lmms-eval___video_mmmu",
    "videomme_v2": f"{HF}/MME-Benchmarks___video-mme-v2",
}
JOINTAV_JSON = f"{EB}/JointAVBench/jointavbench.json"
OVB_JSON     = f"{EB}/OmniVideoBench_local/data.json"
# mmswe (SWE-bench Multimodal) — arrow dir in whichever HF cache prepare_data.sh
# filled; operators who keep a separate SWE cache point SWE_HF_CACHE at it.
_SWE_HF = os.path.join(os.environ.get("SWE_HF_CACHE",
                       os.environ.get("HF_CACHE_DIR", os.path.expanduser("~/hf_cache"))),
                       "datasets")
SWE_ARROW    = os.environ.get("SWE_ARROW_DIR",
                              f"{_SWE_HF}/SWE-bench___swe-bench_multimodal")

# columns that are pure media handles / ids — nothing to leak, skip the noise
_SKIP_COLS = {"audio", "video", "image", "images", "audio_path", "video_path",
              "path", "id", "index", "url", "duration"}
# a real question/choice is short; anything longer is serialized media (e.g.
# MMMU_pro stores images as giant strings) — drop that value, and cap the item.
MAX_VALUE_CHARS = 8000
MAX_ITEM_CHARS = 8000


def _read_arrow_text(ddir: str):
    """Yield per-row text strings from every *.arrow file under ddir, using only
    string/large_string columns (+ list-of-string). Skips binary media columns."""
    import pyarrow as pa
    import pyarrow.ipc as ipc
    files = sorted(glob.glob(os.path.join(ddir, "**", "*.arrow"), recursive=True))
    for fp in files:
        try:
            try:
                reader = ipc.open_file(fp)
                table = reader.read_all()
            except Exception:
                with pa.memory_map(fp, "r") as src:
                    table = ipc.open_stream(src).read_all()
        except Exception as e:
            print(f"[ref] WARN cannot read {fp}: {e}", file=sys.stderr); continue
        # pick text-ish columns
        keep = []
        for name, typ in zip(table.schema.names, table.schema.types):
            if name.lower() in _SKIP_COLS:
                continue
            t = str(typ)
            if "string" in t or "utf8" in t or (t.startswith("list<") and ("string" in t or "utf8" in t)):
                keep.append(name)
        if not keep:
            continue
        cols = {name: table.column(name).to_pylist() for name in keep}
        n = table.num_rows
        for i in range(n):
            parts = []
            for name in keep:
                v = cols[name][i]
                if v is None:
                    continue
                if isinstance(v, (list, tuple)):
                    parts.extend([str(x) for x in v if x is not None])
                else:
                    parts.append(str(v))
            parts = [p for p in parts if p and len(p) <= MAX_VALUE_CHARS]  # drop serialized media
            txt = " ".join(p.strip() for p in parts if p.strip())
            if txt:
                yield txt[:MAX_ITEM_CHARS]


def _walk_strings(x, out):
    if isinstance(x, str):
        s = x.strip()
        if s:
            out.append(s)
    elif isinstance(x, dict):
        for k, v in x.items():
            if str(k).lower() in _SKIP_COLS:
                continue
            _walk_strings(v, out)
    elif isinstance(x, (list, tuple)):
        for v in x:
            _walk_strings(v, out)


def _read_jointav(fp: str):
    data = json.load(open(fp))
    items = data if isinstance(data, list) else data.get("data", list(data.values()))
    for it in items:
        out = []; _walk_strings(it, out)
        txt = " ".join(out)
        if txt:
            yield txt


def _read_ovb(fp: str):
    # OVB: list of videos, each with a 'questions' list — one line per question
    data = json.load(open(fp))
    for vid in data:
        for q in vid.get("questions", []) or []:
            out = []; _walk_strings(q, out)
            txt = " ".join(out)
            if txt:
                yield txt


def _read_mmswe(ddir: str):
    """SWE-bench Multimodal — TWO lines per instance, both dev and test splits
    (an agent with open networking can pull either off HF).

    Split into a PROBLEM side (issue text + screenshot URLs = what a leaked
    "question" looks like) and a GOLD side (patch/test_patch/eval_script/
    FAIL_TO_PASS = the answer). Two reasons: a long gold patch must not be
    truncated away by MAX_ITEM_CHARS behind the problem statement, and the
    contamination report then distinguishes "downloaded the tasks" from the far
    worse "trained on the fixes" — on mmswe the gold patch IS the answer, so a
    single near-verbatim item is enough to void the score."""
    import pyarrow as pa
    import pyarrow.ipc as ipc
    PROBLEM = ("instance_id", "repo", "base_commit", "problem_statement",
               "hints_text", "image_assets")
    GOLD = ("instance_id", "patch", "test_patch", "eval_script",
            "FAIL_TO_PASS", "PASS_TO_PASS")
    for fp in sorted(glob.glob(os.path.join(ddir, "**", "*.arrow"), recursive=True)):
        try:
            table = ipc.open_file(fp).read_all()
        except Exception:
            with pa.memory_map(fp, "r") as src:
                table = ipc.open_stream(src).read_all()
        have = set(table.schema.names)
        cols = {c: table.column(c).to_pylist()
                for c in (set(PROBLEM) | set(GOLD)) & have}
        for i in range(table.num_rows):
            for group in (PROBLEM, GOLD):
                parts = [str(cols[c][i]) for c in group
                         if c in cols and cols[c][i] is not None]
                txt = " ".join(p.strip() for p in parts if p.strip())
                if txt:
                    yield txt[:MAX_ITEM_CHARS]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "contam_ref_all.jsonl"))
    ap.add_argument("--only", action="append", default=[],
                    help="emit ONLY these benches (repeatable); default = all 8")
    ap.add_argument("--append", action="store_true",
                    help="append to --out instead of truncating (pair with --only "
                         "to add one bench to an existing reference)")
    args = ap.parse_args()
    want = set(args.only) or None

    per_bench = {}
    n = 0
    with open(args.out, "a" if args.append else "w") as w:
        for bench, ddir in ARROW_BENCHES.items():
            if want and bench not in want:
                continue
            c = 0
            if os.path.isdir(ddir):
                for txt in _read_arrow_text(ddir):
                    w.write(json.dumps({"bench": bench, "text": txt}, ensure_ascii=False) + "\n")
                    c += 1; n += 1
            else:
                print(f"[ref] WARN missing dir for {bench}: {ddir}", file=sys.stderr)
            per_bench[bench] = c
        for bench, fp, reader in (("jointavbench", JOINTAV_JSON, _read_jointav),
                                  ("omnivideobench", OVB_JSON, _read_ovb),
                                  ("mmswe", SWE_ARROW, _read_mmswe)):
            if want and bench not in want:
                continue
            c = 0
            if os.path.exists(fp):   # mmswe's source is a dir, the others files
                for txt in reader(fp):
                    w.write(json.dumps({"bench": bench, "text": txt}, ensure_ascii=False) + "\n")
                    c += 1; n += 1
            else:
                print(f"[ref] WARN missing source for {bench}: {fp}", file=sys.stderr)
            per_bench[bench] = c

    print(json.dumps({"out": args.out, "total_items": n, "per_bench": per_bench}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
