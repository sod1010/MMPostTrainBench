#!/bin/bash
# Qwen3-Omni 多 benchmark 评测统一入口。把每个 benchmark 的启动方式 + 多卡策略固化于此。
#
#   entrypoint.sh <benchmark> [limit]
#     benchmark ∈ mmmu_pro | mmar | video_mmmu | videomme_v2      (lmms-eval)
#                 omnivideobench                                   (官方脚本)
#                 mmau | jointavbench                              (官方对齐 self-runner)
#     limit: 采样数(默认 8;<=0 或 all = 全量)
#
# 运行时挂载(见 README):
#   -v <model>:/models        30B 模型
#   -v <hf_cache>:/hf_cache    lmms-eval 数据 & 视频缓存(datasets 缓存 + video_mmmu/ + videomme_v2/)
#   -v <data>:/data            官方脚本/self-runner 数据(OmniVideoBench data.json+videos、JointAVBench)
set -eo pipefail

# This legacy entry point invokes raw harnesses and does not apply split scoring.
# Require an explicit operator whole-pool diagnostic request. Normal val/sealed
# evaluation must use the adapters via src/docker/run_verifier.sh.
if [ "${1:-}" != "--help" ]; then
    if [ "${MMPTB_ROLE:-agent}" != "verifier" ] || [ "${EVAL_SPLIT:-}" != "all" ]; then
        echo "Raw harness diagnostics require MMPTB_ROLE=verifier EVAL_SPLIT=all." >&2
        echo "Use the benchmark adapters/run_verifier.sh for val or sealed scores." >&2
        exit 2
    fi
fi

BENCH="${1:?usage: entrypoint.sh <benchmark> [limit]}"
LIMIT="${2:-8}"
[ "$LIMIT" = "all" ] && LIMIT=-1

MODEL="${MODEL_PATH:-/models}"
export HF_HOME="${HF_HOME:-/hf_cache}"
DATA="${EVAL_DATA_ROOT:-/data}"
OUT="${OUT_ROOT:-/opt/eval/results}"
export TMPDIR=/tmp HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
       DECORD_EOF_RETRY_MAX=20480 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NGPU="$(python -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 1)"
mkdir -p "$OUT"
R=/opt/eval/runners
LMMS_ARGS="pretrained=$MODEL,device_map=auto,attn_implementation=sdpa"

echo "[eval] benchmark=$BENCH limit=$LIMIT model=$MODEL ngpu=$NGPU"

case "$BENCH" in
  # ---- lmms-eval 图/音:数据并行(每卡整模型,快)----
  mmmu_pro|mmar)
    task=$([ "$BENCH" = mmmu_pro ] && echo mmmu_pro_standard || echo mmar)
    python -m accelerate.commands.launch --num_processes "$NGPU" --num_machines 1 --mixed_precision bf16 \
      -m lmms_eval --model qwen3_omni --model_args "pretrained=$MODEL,attn_implementation=sdpa" \
      --tasks "$task" --batch_size 1 --limit "$LIMIT" --log_samples --output_path "$OUT" ;;

  # ---- lmms-eval 视频:单进程 device_map=auto 多卡模型并行(视频帧激活大)----
  video_mmmu|videomme_v2)
    task=$([ "$BENCH" = video_mmmu ] && echo video_mmmu_comprehension || echo videomme_v2)
    python -m lmms_eval --model qwen3_omni --model_args "$LMMS_ARGS" \
      --tasks "$task" --batch_size 1 --limit "$LIMIT" --log_samples --output_path "$OUT" ;;

  # ---- OmniVideoBench:官方 qwen3_omni_eval.py(原生 Qwen3-Omni)----
  omnivideobench)
    OVB=/opt/eval/harness_repos/OmniVideoBench
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NGPU-1)))}"
    export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-512}" OVB_LIMIT="$([ "$LIMIT" -gt 0 ] 2>/dev/null && echo "$LIMIT" || echo '')"
    export PYTHONPATH="$OVB:$PYTHONPATH"
    python "$OVB/eval/qwen3_omni_eval.py" \
      --data_json_file "$DATA/OmniVideoBench_local/data.json" \
      --video_dir "$DATA/OmniVideoBench_local/videos" \
      --model_path "$MODEL" --output_file "$OUT/omnivideobench.json" --max_duration 6000 ;;

  # ---- MMAU / JointAVBench:官方对齐 self-runner ----
  mmau)
    python "$R/run_mmau_official.py" --model-path "$MODEL" --limit "$LIMIT" --out "$OUT/mmau.json" ;;
  jointavbench)
    python "$R/run_jointav_official.py" --model-path "$MODEL" --limit "$LIMIT" \
      --data "$DATA/JointAVBench/jointavbench.json" --media-root "$DATA/JointAVBench" --out "$OUT/jointav.json" ;;

  --help|*)
    sed -n '2,20p' "$0"; exit 0 ;;
esac
echo "[eval] done -> $OUT"
