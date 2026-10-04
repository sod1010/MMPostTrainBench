#!/bin/bash
# Download the OPEN Qwen3-Omni-30B weights + the 8 eval benchmarks' data from
# their PUBLIC HuggingFace origins into MMPTB_ROOT, so the omni-eval / verifier
# containers can bind-mount them (-v $MODEL_DIR:/models, -v $HF_CACHE_DIR:/hf_cache,
# -v $DATA_DIR:/data). Mirrors PostTrainBench's containers/download_hf_cache flow:
# a declarative manifest (resources.json) + a generic downloader here.
#
# Idempotent: skips a repo whose cache/output already looks complete. Uses the HF
# mirror (set HF_ENDPOINT in config.env). Per-bench repo overridable via <ID>_REPO.
#
#   bash prepare_data.sh            # model + all 8 benches
#   bash prepare_data.sh mmau mmar  # only the named benches (model always checked)
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.env"
MANIFEST="${RESOURCES_JSON:-$HERE/resources.json}"

export HF_ENDPOINT
export HF_HUB_DISABLE_XET=1          # xet/torchcodec is deliberately absent in the omni env
TOKEN_FILE="${HF_TOKEN_FILE:-${TOKEN_FILE:-${XDG_CONFIG_HOME:-$HOME/.config}/mmposttrainbench/hf_token}}"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$TOKEN_FILE" ]; then
    HF_TOKEN="$(tr -d '[:space:]' < "$TOKEN_FILE")"
fi
export HF_TOKEN
# The HF CLI reads HF_TOKEN from the environment; keep it out of process argv.

HF_BIN="${HF_BIN:-hf}"
PY="${PREPARE_PY:-python3}"
command -v "$HF_BIN" >/dev/null || { echo "ERROR: '$HF_BIN' CLI not found"; exit 1; }
[ -f "$MANIFEST" ] || { echo "ERROR: manifest $MANIFEST not found"; exit 1; }
mkdir -p "$HF_CACHE_DIR" "$DATA_DIR"

WANT=("$@")   # optional subset of bench ids; empty = all

# ---- 1. model weights -> $MODEL_DIR (clean HF-format dir mounted at /models) --
MODEL_REPO="${MODEL_REPO:-$("$PY" -c "import json;print(json.load(open('$MANIFEST'))['model']['default_repo'])")}"
if [ -f "$MODEL_DIR/config.json" ] && ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1; then
    echo "=== model already present at $MODEL_DIR — skip ==="
else
    echo "=== downloading $MODEL_REPO -> $MODEL_DIR (HF_ENDPOINT=$HF_ENDPOINT) ==="
    mkdir -p "$MODEL_DIR"
    "$HF_BIN" download "$MODEL_REPO" --local-dir "$MODEL_DIR"
fi

# ---- 2. per-bench datasets (generic loop over resources.json) ----------------
# emit: "<id>\t<repo>\t<cache_glob>\t<post>\t<needs_token>" per bench
ROWS=()
while IFS= read -r resource_row; do ROWS+=("$resource_row"); done < <("$PY" - "$MANIFEST" <<'PY'
import json, os, sys
m = json.load(open(sys.argv[1]))
for b in m["benches"]:
    repo = os.environ.get(b["id"].upper() + "_REPO", b["repo"])
    if any("|" in str(x) or "\n" in str(x) for x in (repo, b["id"], b.get("cache_glob", ""), b.get("post", ""))):
        raise ValueError("Invalid delimiter in resource manifest")
    print("|".join([b["id"], repo, b.get("cache_glob", "*"+b["id"]+"*"),
                     b.get("post", ""), "1" if b.get("needs_token") else "0"]))
PY
)

want_bench() { [ ${#WANT[@]} -eq 0 ] && return 0; for w in "${WANT[@]}"; do [ "$w" = "$1" ] && return 0; done; return 1; }

# --- post-processing hooks ----------------------------------------------------
# omnivideobench: official repo -> our local layout via the existing converter.
omnivideobench_convert() {  # $1=snapshot_dir
    local out="${OVB_DATA:-$DATA_DIR/evaluationbench/OmniVideoBench_local/data.json}"
    local vid="${OVB_VIDEO_DIR:-$DATA_DIR/OmniVideoBench/videos_local}"
    local conv="$REPO_ROOT/eval_omni/runners/convert_omnivideobench.py"
    if [ -f "$conv" ]; then
        echo "    converting OmniVideoBench -> $out (videos -> $vid)"
        "$PY" "$conv" --src "$1" --out "$out" --video-dir "$vid"
    else
        echo "ERROR: converter missing" >&2; return 1
    fi
}

# jointavbench: 2853-row public mirror -> run_jointav_official.py wants
#   JOINTAV_DATA=<dir>/jointavbench.json + videos/<qid>.mp4. The HF repo carries the
#   QA json + video files; stage them into the expected layout.
jointavbench_convert() {  # $1=snapshot_dir
    local dst="${JOINTAV_DATA:-$DATA_DIR/evaluationbench/JointAVBench/jointavbench.json}"
    local dstdir; dstdir="$(dirname "$dst")"
    if [ -f "$dst" ]; then echo "    jointavbench already staged -> $dst"; return 0; fi
    mkdir -p "$dstdir/videos"
    [ -f "$1/jointavbench.json" ] || { echo "ERROR: missing JointAVBench annotation" >&2; return 1; }
    cp "$1/jointavbench.json" "$dst"
    "$PY" - "$1/videos" "$dstdir/videos" <<'PYMEDIA'
import os,sys
from pathlib import Path
src,dst=map(Path,sys.argv[1:]); media=list(src.glob('*.mp4'))
if not media: raise SystemExit('Missing JointAVBench videos')
for p in media:
    q=dst/p.name
    if q.exists() or q.is_symlink(): q.unlink()
    q.symlink_to(os.path.relpath(p.resolve(),dst.resolve()))
PYMEDIA
    echo "    jointavbench staged from $1 -> $dst (set JOINTAV_DATA / JOINTAV_MEDIA_ROOT if the repo layout differs)"
}

# videomme_v2: lmms-eval utils.py hard-requires the 800 source .mp4 at
#   HF_HOME/<cache_dir>/{video_id}.mp4. The dataset repo ships videos (often as
#   archives); extract them into that dir.
videomme_v2_extract() {  # $1=snapshot_dir
    local cache="${VIDEOMME_V2_VIDEO_DIR:-$HF_CACHE_DIR/videomme_v2_videos}"
    mkdir -p "$cache"
    local n; n="$(find "$cache" -name '*.mp4' 2>/dev/null | wc -l)"
    if [ "$n" -ge 800 ]; then echo "    videomme_v2: $n mp4 already present in $cache"; return 0; fi
    find "$1" -maxdepth 3 \( -iname '*.zip' -o -iname '*.tar' -o -iname '*.tar.gz' -o -iname '*.tgz' \) -print0 2>/dev/null |
        while IFS= read -r -d '' a; do
            case "$a" in *.zip) unzip -oq "$a" -d "$cache" ;; *) tar -xf "$a" -C "$cache" ;; esac
        done
    find "$1" -maxdepth 4 -iname '*.mp4' -exec ln -sf {} "$cache/" \; 2>/dev/null || true
    n="$(find "$cache" -name '*.mp4' 2>/dev/null | wc -l)"
    echo "    videomme_v2: $n mp4 in $cache (need 800; if short, the repo may host videos separately — set VIDEOMME_V2_VIDEO_DIR)"
}

for row in "${ROWS[@]}"; do
    IFS='|' read -r id repo cache_glob post needs_token <<< "$row"
    want_bench "$id" || continue
    if [ "$needs_token" = "1" ] && [ -z "${HF_TOKEN:-}" ]; then
        echo "ERROR: $id requires approved Hugging Face access and HF_TOKEN_FILE" >&2; exit 1
    fi
    case "$post" in ""|omnivideobench_convert|jointavbench_convert|videomme_v2_extract) ;; *) echo "Invalid post handler" >&2; exit 1;; esac
    echo "=== $id  <-  $repo ==="
    if [ -n "$post" ]; then
        # convert/extract benches: pull the repo into a staging dir, then post-process
        stage="$DATA_DIR/_staging/$id"; mkdir -p "$stage"
        if ! find "$stage" -maxdepth 3 -type f | grep -q .; then
            "$HF_BIN" download "$repo" --repo-type dataset --local-dir "$stage" || \
                { echo "    download failed for $repo (check token/mirror)" >&2; exit 1; }
        fi
        "$post" "$stage"
    else
        # HF-cache benches: pre-populate HF_HOME hub cache for offline container runs
        if find "$HF_CACHE_DIR" -maxdepth 4 -ipath "$cache_glob" 2>/dev/null | grep -q .; then
            echo "    already cached under $HF_CACHE_DIR — skip"
        else
            HF_HOME="$HF_CACHE_DIR" "$HF_BIN" download "$repo" --repo-type dataset || \
                { echo "    download failed for $repo (check token/mirror)" >&2; exit 1; }
        fi
    fi
done

echo ""
echo "MODEL_DIR    = $MODEL_DIR ($(ls "$MODEL_DIR"/*.safetensors 2>/dev/null | wc -l) shards)"
echo "HF_CACHE_DIR = $HF_CACHE_DIR ($(du -sh "$HF_CACHE_DIR" 2>/dev/null | cut -f1))"
echo "DATA_DIR     = $DATA_DIR ($(du -sh "$DATA_DIR" 2>/dev/null | cut -f1))"
echo "Done. (Per-bench overrides: <ID>_REPO; and the *_DATA / *_VIDEO_DIR env in config.env.)"
