#!/bin/bash
# Per-benchmark eval recipe registry — the single source of truth for HOW each
# of the 8 benches is evaluated (which env, which evaluate.py adapter, task
# env vars, GPU count). run_verifier.sh sources this and dispatches by benchmark,
# so all 8 go through the SAME isolated verifier safeguards.
#
# emit_bench_cmd <bench> <model_path> <out_dir> <limit>:
#   writes <out_dir>/cmd.sh (runs the right evaluate.py -> metrics.json ->
#   reward.txt -> DONE) and echoes the GPU count for that bench to stdout.
#
# Recipe shapes:
#   self-runner (mmau/mmar/jointavbench): OMNI_PY (omni-eval env) + run_*_official.py
#   lmms-eval  (mmmu_pro/video_mmmu/videomme_v2): LMMS_PY (lmms-eval env) + lmms_common adapter
#   official   (omnivideobench): OMNI_PY (omni-eval env) + OVB official script
#   two-stage  (mmswe): OMNI_PY generates patches -> swebench venv grades them for
#                       real (resolved rate, normalized to the same "accuracy")

# All operator-specific paths are env-driven (sourced from config.env by the
# caller — run_verifier.sh). Defaults are repo-relative placeholders.
REPO="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OMNI_PY="${OMNI_PY:-python3}"                 # python in the omni-eval env
LMMS_PY="${LMMS_PY:-python3}"                 # python in the lmms-eval env
RUNNERS=$REPO/eval_omni/runners
EVALDIR=$REPO/src/eval
HFCACHE="${HF_CACHE_DIR:-$REPO/hf_cache}"
CABUNDLE="${CABUNDLE:-}"                       # optional operator-approved CA bundle
HFTOKEN_FILE="${HF_TOKEN_FILE:-${XDG_CONFIG_HOME:-$HOME/.config}/mmposttrainbench/hf_token}"
FFMPEG_SHIM="${FFMPEG_SHIM:-}"                 # optional ffmpeg shim libdir
OMNI_LIB="${OMNI_LIB:-}"                       # optional extra libdir for omni env
OMNI_PY_BIN="$(dirname "$OMNI_PY")"
LMMS_PY_BIN="$(dirname "$LMMS_PY")"
# OmniVideoBench data/media (repo-relative defaults under DATA_DIR)
OVB_DATA="${OVB_DATA:-${DATA_DIR:-$REPO/data}/OmniVideoBench_local/data.json}"
OVB_VIDEO_DIR="${OVB_VIDEO_DIR:-${DATA_DIR:-$REPO/data}/OmniVideoBench/videos_local}"
# mmswe (SWE-bench Multimodal) grader home: holds venv/ (swebench 5.0.2),
# hf_cache/ (the offline SWE dataset) and blob_cache/ (verified docker layers).
SWE_WORK="${SWE_WORK:-${DATA_DIR:-$REPO/data}/mmptb_swe}"
SWE_VENV_PY="${SWE_VENV_PY:-$SWE_WORK/venv/bin/python}"
SWE_HF_CACHE="${SWE_HF_CACHE:-$HFCACHE}"   # where prepare_data.sh cached the SWE dataset
SWE_DATASET="${SWE_DATASET:-SWE-bench/SWE-bench_Multimodal}"
# Do not force dev here: split_util maps val -> dev and eval -> test.

# common preamble: ffmpeg on PATH (av/video) + shim libs
_common_env() {
cat <<EOF
export PATH=$OMNI_PY_BIN:\$PATH
export LD_LIBRARY_PATH=$FFMPEG_SHIM:$OMNI_LIB:\$LD_LIBRARY_PATH
export TMPDIR=/dev/shm HF_HUB_DISABLE_XET=1
export EVAL_SPLIT='${EVAL_SPLIT:-}'
EOF
}
# HF online-first (datasets pulled via huggingface.co + optional CA and token)
_hf_online() {
cat <<EOF
export HF_HOME=$HFCACHE HF_ENDPOINT=https://huggingface.co
export HF_TOKEN=\$(tr -d '[:space:]' < $HFTOKEN_FILE)
export REQUESTS_CA_BUNDLE=$CABUNDLE SSL_CERT_FILE=$CABUNDLE CURL_CA_BUNDLE=$CABUNDLE
EOF
}
_hf_offline() { echo "export HF_HOME=$HFCACHE HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1"; }

# emit_bench_cmd BENCH MODEL OUT LIMIT  -> writes OUT/cmd.sh, echoes GPU
emit_bench_cmd() {
    local bench="$1" model="$2" out="$3" limit="$4" gpu=1
    local cmd="$out/cmd.sh"
    mkdir -p "$out"
    { echo "set -e"; echo "OUT=$out"; _common_env; } > "$cmd"
    case "$bench" in
      mmau)
        _hf_online >> "$cmd"
        cat >> "$cmd" <<EOF
export MMAU_RUNNER=$RUNNERS/run_mmau_official.py
$OMNI_PY $EVALDIR/tasks/mmau/evaluate.py --model-path '$model' --limit $limit --json-output-file "\$OUT/metrics.json" > "\$OUT/run.log" 2>&1
EOF
        gpu=1 ;;
      mmar)
        _hf_online >> "$cmd"
        cat >> "$cmd" <<EOF
export MMAR_RUNNER=$RUNNERS/run_mmar_official.py
$OMNI_PY $EVALDIR/tasks/mmar/evaluate.py --model-path '$model' --limit $limit --json-output-file "\$OUT/metrics.json" > "\$OUT/run.log" 2>&1
EOF
        gpu=1 ;;
      jointavbench)
        _hf_offline >> "$cmd"
        cat >> "$cmd" <<EOF
export JOINTAV_RUNNER=$RUNNERS/run_jointav_official.py
$OMNI_PY $EVALDIR/tasks/jointavbench/evaluate.py --model-path '$model' --limit $limit --json-output-file "\$OUT/metrics.json" > "\$OUT/run.log" 2>&1
EOF
        gpu=1 ;;
      mmmu_pro)
        _hf_online >> "$cmd"
        cat >> "$cmd" <<EOF
export PATH=$LMMS_PY_BIN:\$PATH
export LMMS_TASK=mmmu_pro_standard LMMS_METRIC=mmmu_acc LMMS_DIVISOR=1
$LMMS_PY $EVALDIR/lmms_common/evaluate.py --model-path '$model' --limit $limit --json-output-file "\$OUT/metrics.json" > "\$OUT/run.log" 2>&1
EOF
        gpu=1 ;;
      video_mmmu)
        _hf_online >> "$cmd"
        cat >> "$cmd" <<EOF
export PATH=$LMMS_PY_BIN:\$PATH
export DECORD_EOF_RETRY_MAX=20480
export LMMS_TASK=video_mmmu_comprehension LMMS_METRIC=mmmu_acc LMMS_DIVISOR=1
$LMMS_PY $EVALDIR/lmms_common/evaluate.py --model-path '$model' --limit $limit --json-output-file "\$OUT/metrics.json" > "\$OUT/run.log" 2>&1
EOF
        gpu=4 ;;
      videomme_v2)
        _hf_online >> "$cmd"
        cat >> "$cmd" <<EOF
export PATH=$LMMS_PY_BIN:\$PATH
export DECORD_EOF_RETRY_MAX=20480
export LMMS_TASK=videomme_v2 LMMS_METRIC=videomme_v2_overall_acc LMMS_DIVISOR=100
$LMMS_PY $EVALDIR/lmms_common/evaluate.py --model-path '$model' --limit $limit --json-output-file "\$OUT/metrics.json" > "\$OUT/run.log" 2>&1
EOF
        gpu=4 ;;
      omnivideobench)
        _hf_offline >> "$cmd"
        cat >> "$cmd" <<EOF
export OVB_RUNNER=$REPO/eval_omni/harness_repos/OmniVideoBench/eval/qwen3_omni_eval.py
export OVB_DATA=$OVB_DATA
export OVB_VIDEO_DIR=$OVB_VIDEO_DIR
$OMNI_PY $EVALDIR/tasks/omnivideobench/evaluate.py --model-path '$model' --limit $limit --json-output-file "\$OUT/metrics.json" > "\$OUT/run.log" 2>&1
EOF
        gpu=4 ;;
      mmswe)
        # SWE-bench Multimodal, two stages in ONE cmd.sh:
        #   1) OMNI_PY runs run_mmswe_official.py -> predictions.jsonl (a candidate
        #      unified diff per instance, generated from issue text + screenshots);
        #   2) SWE_VENV_PY runs the Route B grader -> pulls each instance's official
        #      image via registry v2, sha256-verifies every blob, unpacks a rootfs
        #      into /dev/shm, unshare+chroots, applies the patch and runs the tests.
        # accuracy = resolved / n (real FAIL_TO_PASS + PASS_TO_PASS verdicts).
        #
        # HF is OFFLINE here (the SWE dataset is pre-cached under SWE_HF_CACHE) so
        # official dev/test selection is identical in generation and grading.
        #
        # Deliberately NO CABUNDLE: REQUESTS_CA_BUNDLE/SSL_CERT_FILE *replace* the
        # system trust store rather than extend it, and this bench is the one that
        # talks to MANY public hosts (the registry, the blob mirror, and each
        # instance's screenshot host). A narrower bundle would break some of those
        # TLS handshakes at random, so mmswe keeps the system roots.
        cat >> "$cmd" <<EOF
export HF_HOME=$SWE_HF_CACHE HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export SWE_WORK=$SWE_WORK
export SWE_VENV_PY=$SWE_VENV_PY
export MMSWE_RUNNER=$RUNNERS/run_mmswe_official.py
export MMSWE_GRADER=$EVALDIR/tasks/mmswe/dlc_native_grade.py
export SWE_DATASET=$SWE_DATASET
${SWE_SPLIT:+export SWE_SPLIT=$SWE_SPLIT}
$OMNI_PY $EVALDIR/tasks/mmswe/evaluate.py --model-path '$model' --limit $limit --json-output-file "\$OUT/metrics.json" > "\$OUT/run.log" 2>&1
EOF
        gpu=8 ;;
      *) echo "emit_bench_cmd: unknown bench '$bench'" >&2; return 1 ;;
    esac
    # common tail: extract scalar reward + DONE sentinel
    cat >> "$cmd" <<EOF
rc=\$?
$OMNI_PY -c "import json;print(json.load(open('\$OUT/metrics.json'))['accuracy'])" > "\$OUT/reward.txt" 2>>"\$OUT/run.log" || true
echo "rc=\$rc" >> "\$OUT/run.log"
echo DONE > "\$OUT/DONE"
EOF
    chmod +x "$cmd"
    # recommended host memory: video decode is RAM-heavy (videomme_v2 OOM-killed at
    # the 128Gi default). Written to a sidecar file so the GPU-only echo contract
    # stays backward compatible; callers read mem.txt to provision --memory.
    local mem=128Gi
    case "$bench" in
      video_mmmu|videomme_v2|omnivideobench) mem=1024Gi ;;
      mmswe) mem=1024Gi ;;   # rootfs unpack into /dev/shm (tmpfs=RAM) + 30B model
      jointavbench) mem=512Gi ;;
      mmmu_pro) mem=256Gi ;;
    esac
    echo "$mem" > "$out/mem.txt"
    echo "$gpu"
}
