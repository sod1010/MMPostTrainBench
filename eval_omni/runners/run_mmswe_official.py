#!/usr/bin/env python3
"""SWE-bench Multimodal (mmswe) GENERATION runner for mmposttrainbench.

Stage-1 of the two-stage mmswe eval: load the omni model (Qwen3-Omni-30B moe,
transformers) and, for each SWE-bench MM instance in the selected split/limit,
produce a candidate git unified-diff patch from the issue text + (optional)
problem-statement screenshots. Writes `predictions.jsonl` (swebench format:
{instance_id, model_name_or_path, model_patch}) + `meta.json` (ordered
instance_ids, n, split). GRADING is a separate stage (dlc_native_grade.py,
Route B, no docker) driven by tasks/mmswe/evaluate.py — this runner never sees
the gold patch / tests, so nothing sealed crosses the model seam.

v1 = single-shot BLIND generation (no repo checkout): the model gets the problem
statement + hints + screenshots and must emit a diff. This is the honest floor
for the go/no-go gate; a localize->edit scaffold is a later iteration.

Mirrors run_mmau_official.py's model-loading / split / EVAL_SPLIT handling so
the mmswe runner is consistent with the other self-runners.

CLI:
  --model-path PATH   (required) HF dir of the omni model
  --limit N           cap instances (N<=0 = full split)
  --out PATH          predictions.jsonl to write (meta.json written alongside)
Env:
  EVAL_SPLIT          val|eval|""  (see MMSWE_SPLIT_BY_OFFICIAL)
  MMSWE_SPLIT_BY_OFFICIAL  "1" (default): val->HF dev (100, visible), eval->HF test
                      (480, sealed), no hash-keep. "0": legacy 30/70 hash split_util
                      over MMSWE_SPLIT.
  MMSWE_DATASET       default SWE-bench/SWE-bench_Multimodal
  MMSWE_SPLIT         default dev  (only used when MMSWE_SPLIT_BY_OFFICIAL=0)
  MMSWE_USE_IMAGES    "1" (default) to fetch problem-statement screenshots
  MMSWE_MAX_IMAGES    default 4
  MMSWE_MAX_NEW_TOKENS default 3072
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys, tempfile
from pathlib import Path

MODEL_NAME = "mmptb-omni"
SRC_EVAL = os.environ.get("MMPTB_SRC_EVAL") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "eval")


def _extract_patch(text: str) -> str:
    """Pull a git unified diff out of the model's free-text response.

    Prefer a ```diff ...``` (or ```patch/```) fenced block; else take from the
    first `diff --git` to the end. Returns "" if nothing diff-like is present
    (an empty patch never applies -> counts as unresolved, which is correct)."""
    if not text:
        return ""
    # fenced block
    m = re.search(r"```(?:diff|patch)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        body = m.group(1).strip("\n")
        if "diff --git" in body or body.lstrip().startswith(("--- ", "diff ")):
            return body.rstrip("\n") + "\n"
    # bare diff --git .. EOF
    i = text.find("diff --git ")
    if i != -1:
        return text[i:].rstrip("\n") + "\n"
    return ""


def _frozen_images(urls, max_images):
    manifest = Path(os.environ['MMSWE_IMAGE_MANIFEST']).resolve()
    raw = manifest.read_bytes()
    expected = os.environ.get('MMSWE_IMAGE_MANIFEST_SHA256')
    if expected and hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('image manifest digest mismatch')
    table = json.loads(raw)
    if table.get('version') != 1:
        raise ValueError('unsupported image manifest')
    paths = []
    for url in urls[:max_images]:
        entry = table['images'][url]
        if entry['status'] == 'unavailable':
            continue
        digest = entry['sha256']
        if entry['status'] != 'available' or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('invalid image manifest entry')
        path = manifest.parent / 'blobs' / (digest + '.png')
        if path.is_symlink():
            raise ValueError('image blob must not be a symlink')
        data = path.read_bytes()
        if len(data) != entry['size'] or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError('image bytes differ from manifest')
        paths.append(str(path))
    return paths


def _fetch_images(urls, max_images):
    """Best-effort download of problem-statement screenshots to temp files.

    Network to user-images.githubusercontent.com may be blocked on the eval
    pod; every failure is swallowed and simply yields fewer/zero images (the
    run degrades to text-only, logged). Uses the cluster CA bundle if present."""
    if max_images < 0:
        raise ValueError('MMSWE_MAX_IMAGES must be nonnegative')
    if os.environ.get('MMSWE_IMAGE_MANIFEST'):
        return _frozen_images(urls, max_images)
    if os.environ.get('MMSWE_REQUIRE_FROZEN_IMAGES') == '1':
        raise ValueError('frozen image manifest required')
    paths = []
    if not urls:
        return paths
    try:
        import requests
    except Exception as e:
        print(f"[mmswe] requests unavailable ({e}); text-only", flush=True)
        return paths
    ca = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("CURL_CA_BUNDLE")
    tmpd = tempfile.mkdtemp(prefix="mmswe_img_")
    for k, u in enumerate(urls[:max_images]):
        if not isinstance(u, str) or not u.startswith("http"):
            continue
        try:
            r = requests.get(u, timeout=15, verify=(ca or True))
            if r.status_code == 200 and r.content:
                ext = ".png"
                ct = r.headers.get("content-type", "")
                if "jpeg" in ct or "jpg" in ct:
                    ext = ".jpg"
                p = os.path.join(tmpd, f"img{k}{ext}")
                with open(p, "wb") as f:
                    f.write(r.content)
                # A 200 does NOT mean we got an image: these URLs also serve HTML
                # error pages, SVGs and short clips, and PIL then raises
                # UnidentifiedImageError deep inside the processor and kills the
                # whole run (that is exactly how base_val died at 15/34). Verify
                # here and drop anything Pillow cannot decode.
                try:
                    from PIL import Image
                    with Image.open(p) as im:
                        im.verify()
                except Exception as e:
                    print(f"[mmswe]   not a decodable image, dropped "
                          f"({u[:60]}...): {type(e).__name__}", flush=True)
                    try:
                        os.remove(p)
                    except OSError:
                        pass
                    continue
                paths.append(p)
        except Exception as e:
            print(f"[mmswe]   image fetch failed ({u[:60]}...): {e}", flush=True)
    return paths


def _build_prompt(doc):
    repo = doc.get("repo", "")
    base = doc.get("base_commit", "")
    ps = (doc.get("problem_statement") or "").strip()
    hints = (doc.get("hints_text") or "").strip()
    parts = [
        f"Repository: {repo} (at base commit {base})",
        "",
        "## Issue",
        ps,
    ]
    if hints:
        parts += ["", "## Additional hints", hints]
    parts += [
        "",
        "## Task",
        "Produce a single git unified diff that resolves this issue, exactly as "
        "produced by `git diff` at the repository root. File paths must use the "
        "`a/<path>` and `b/<path>` prefixes. Do not include commentary. Output "
        "ONLY the diff, wrapped in a ```diff code fence.",
    ]
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dataset = os.environ.get("MMSWE_DATASET", "SWE-bench/SWE-bench_Multimodal")
    split = os.environ.get("MMSWE_SPLIT", "dev")
    use_images = os.environ.get("MMSWE_USE_IMAGES", "1") not in ("0", "", "false")
    max_images = int(os.environ.get("MMSWE_MAX_IMAGES", "4"))
    max_new = int(os.environ.get("MMSWE_MAX_NEW_TOKENS", "3072"))

    import torch
    from transformers import (Qwen3OmniMoeForConditionalGeneration,
                              Qwen3OmniMoeProcessor)
    from qwen_omni_utils import process_mm_info
    from datasets import load_dataset

    sys.path.insert(0, SRC_EVAL)
    from split_util import keep as _keep, resolve_split, mmswe_config

    _SPLIT = resolve_split(os.environ.get("EVAL_SPLIT", ""))
    dataset, split, official = mmswe_config(_SPLIT)
    frozen_file = os.environ.get('MMSWE_FROZEN_DATASET_FILE')
    if frozen_file:
        from datasets import Dataset
        raw = Path(frozen_file).read_bytes()
        if hashlib.sha256(raw).hexdigest() != os.environ.get('MMSWE_FROZEN_DATASET_SHA256'):
            raise ValueError('frozen generation dataset bytes changed')
        d = Dataset.from_file(frozen_file)
    else:
        d = load_dataset(dataset, split=split)
    if not official and _SPLIT:
        d = d.select([i for i in range(len(d)) if _keep(i, _SPLIT)])
    if args.limit > 0:
        d = d.select(range(min(args.limit, len(d))))
    if os.environ.get('MMSWE_EXPECTED_IDS_JSON'):
        if list(d['instance_id']) != json.loads(os.environ['MMSWE_EXPECTED_IDS_JSON']):
            raise ValueError('generation IDs differ from frozen evaluation contract')
    print(f"[mmswe] dataset={dataset} split={split} -> {len(d)} instances", flush=True)

    if not len(d):
        raise ValueError("empty MMSWE split")
    print("[mmswe] loading model...", flush=True)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa").eval()
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    proc = Qwen3OmniMoeProcessor.from_pretrained(args.model_path)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    meta_path = os.path.join(os.path.dirname(out_path), "meta.json")

    instance_ids = []
    n_nonempty = 0
    n_failed = 0
    with open(out_path, "w") as fout:
        for i, doc in enumerate(d):
            iid = doc["instance_id"]
            instance_ids.append(iid)
            prompt = _build_prompt(doc)

            img_paths = []
            if use_images:
                ia = doc.get("image_assets")
                urls = []
                if isinstance(ia, dict):
                    urls = ia.get("problem_statement") or []
                elif isinstance(ia, str):
                    try:
                        urls = (json.loads(ia) or {}).get("problem_statement") or []
                    except Exception:
                        urls = []
                img_paths = _fetch_images(urls, max_images)

            # One instance must never be able to abort the whole eval: a bad
            # asset, an OOM or a processor error is recorded for diagnosis; any
            # generation failure makes the run incomplete (nonzero exit, no score).
            try:
                content = []
                for p in img_paths:
                    content.append({"type": "image", "image": p})
                content.append({"type": "text", "text": prompt})
                conv = [
                    {"role": "system", "content": [{"type": "text",
                     "text": "You are an expert software engineer who fixes bugs in "
                             "JavaScript / front-end repositories by writing precise "
                             "git unified-diff patches."}]},
                    {"role": "user", "content": content},
                ]
                text = proc.apply_chat_template(conv, add_generation_prompt=True,
                                                tokenize=False)
                audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
                inputs = proc(text=text, audio=audios, images=images, videos=videos,
                              return_tensors="pt", padding=True,
                              use_audio_in_video=False).to("cuda").to(model.dtype)
                with torch.no_grad():
                    gen = model.generate(**inputs, max_new_tokens=max_new,
                                         do_sample=False, use_audio_in_video=False,
                                         thinker_do_sample=False)
                if isinstance(gen, tuple):
                    gen = gen[0]
                full = proc.batch_decode(gen, skip_special_tokens=True)[0]
                resp = full.split("assistant\n")[-1].strip()
                patch = _extract_patch(resp)
            except Exception as e:
                patch = ""
                n_failed += 1
                print(f"  [{i+1}/{len(d)}] {iid} GENERATION FAILED "
                      f"({type(e).__name__}: {str(e)[:160]}) -> empty patch",
                      flush=True)
            n_nonempty += int(bool(patch.strip()))
            fout.write(json.dumps({"instance_id": iid,
                                   "model_name_or_path": MODEL_NAME,
                                   "n_images": len(img_paths),
                                   "image_sha256": [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in img_paths],
                                   "model_patch": patch}) + "\n")
            fout.flush()
            print(f"  [{i+1}/{len(d)}] {iid} imgs={len(img_paths)} "
                  f"patch_len={len(patch)}", flush=True)

    with open(meta_path, "w") as f:
        json.dump({"task": "swe_bench_multimodal", "dataset": dataset,
                   "split": split, "eval_split": _SPLIT, "n": len(instance_ids),
                   "n_nonempty_patch": n_nonempty,
                   "n_generation_failed": n_failed,
                   "image_manifest_sha256": (hashlib.sha256(Path(os.environ['MMSWE_IMAGE_MANIFEST']).read_bytes()).hexdigest()
                                             if os.environ.get('MMSWE_IMAGE_MANIFEST') else None),
                   "evaluation_contract_sha256": os.environ.get('MMSWE_EVALUATION_CONTRACT_SHA256'),
                   "instance_ids": instance_ids}, f, indent=2)
    print(f"[mmswe] wrote {len(instance_ids)} preds ({n_nonempty} non-empty, "
          f"{n_failed} generation failures) "
          f"-> {out_path}; meta -> {meta_path}", flush=True)
    return 5 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
