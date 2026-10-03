#!/usr/bin/env python
"""Pre-warm the shared blob cache (MMSWE_BLOB_CACHE) for the mmswe val subset,
from a login node — pure curl + sha256, NO GPU, NO GPU job, NO unshare. The blob PULL
half of grading needs none of those; only unpack (unshare/chroot) does. So we
fetch + verify every layer of every val image into the content-addressed cache
here, and the later GPU grading job finds a warm cache (shared layers already on
the shared filesystem) and only has to unpack + run tests.

Selection mirrors run_mmswe_official.py EXACTLY: load_dataset(split=MMSWE_SPLIT)
-> keep(i, EVAL_SPLIT) so we warm precisely the instances Phase 2 will grade.

Usage:
  ./venv/bin/python prewarm_cache.py [--limit N] [--split val]
"""
import argparse, os, sys, time, urllib.parse
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dlc_native_grade as G  # reuse the verified-pull path (ensure_blob etc.)

# split_util lives in src/eval; this file sits in src/eval/tasks/mmswe/
SRC_EVAL = os.environ.get("MMPTB_SRC_EVAL") or str(HERE.parent.parent)


class _Log:
    def write(self, s):
        sys.stdout.write(s)
    def flush(self):
        sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.environ.get("MMSWE_DATASET",
                    "SWE-bench/SWE-bench_Multimodal"))
    ap.add_argument("--split", default=os.environ.get("MMSWE_SPLIT", "dev"))
    ap.add_argument("--eval-split", default=os.environ.get("EVAL_SPLIT", "val"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    if G.BLOB_CACHE is None:
        print("MMSWE_BLOB_CACHE disabled — nothing to warm", flush=True); return 2
    G.BLOB_CACHE.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from swebench.harness.utils import make_test_spec
    sys.path.insert(0, SRC_EVAL)
    from split_util import keep as _keep

    d = load_dataset(args.dataset, split=args.split)
    if args.eval_split:
        d = d.select([i for i in range(len(d)) if _keep(i, args.eval_split)])
    if args.limit > 0:
        d = d.select(range(min(args.limit, len(d))))
    print(f"[prewarm] dataset={args.dataset} split={args.split} "
          f"eval_split={args.eval_split} -> {len(d)} instances", flush=True)

    logf = _Log()
    scratch = Path("/dev/shm/mmswe_prewarm"); scratch.mkdir(parents=True, exist_ok=True)
    seen = set()
    t_all = time.time()
    warmed = 0
    for k, doc in enumerate(d):
        iid = doc["instance_id"]
        spec = make_test_spec(doc)
        repo, tag = G.parse_image_ref(spec.image)
        token = None
        if G.USE_TOKEN:
            token = G._curl_json(
                G.AUTH.format(repo=urllib.parse.quote(repo, safe="/")))["token"]
        manifest = G.get_manifest(repo, tag, token, logf)
        digests = [manifest["config"]["digest"]] + [l["digest"] for l in manifest["layers"]]
        t0 = time.time()
        n_new = 0
        for dg in digests:
            if dg in seen:
                continue
            seen.add(dg)
            cpath = G.BLOB_CACHE / dg.replace(":", "_")
            if cpath.exists():
                continue
            G.ensure_blob(repo, dg, scratch, logf)  # pull->verify->cache
            n_new += 1
        warmed += 1
        print(f"[prewarm] [{k+1}/{len(d)}] {iid} layers={len(digests)} "
              f"new={n_new} in {time.time()-t0:.0f}s (cache={len(seen)} uniq)",
              flush=True)
    print(f"[prewarm] DONE {warmed} instances, {len(seen)} unique blobs, "
          f"{time.time()-t_all:.0f}s total", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
