#!/bin/bash
# PUBLIC megatron SFT entrypoint for Qwen3-Omni-30B-A3B (MoE) — open-source.
#
# Full-parameter Megatron-SWIFT SFT that writes HF safetensors directly
# (--save_safetensors true), so the produced checkpoint loads straight into the
# eval/verifier image (Qwen3OmniMoeForConditionalGeneration) with no mcore->HF
# conversion. Runs inside the omni-train image (train_omni/Dockerfile), which
# has the public ms-swift[megatron] stack on PATH — no internal PYTHONPATH, no
# conda env, no vendored framework forks.
#
# The mmposttrainbench loop calls this from inside the agent container
# (src/docker/run_agent.sh), which sets MODEL_PATH / DATASET_PATH / OUTPUT_DIR /
# MODEL_TYPE / MTP_NUM_LAYERS (and the TP/EP/GBS/iters knobs) by env. It also runs
# standalone on any 8-GPU node with the omni-train image.
#
# Docs: https://swift.readthedocs.io/en/latest/Megatron-SWIFT/Quick-start.html
set -eo pipefail

echo "[$(date)] SFT start; python: $(command -v python) $(python --version 2>&1)"

# ── paths (all env-overridable; defaults are container mount points) ─────────
MODEL_PATH="${MODEL_PATH:-/models}"                 # HF-format Qwen3-Omni-30B base
DATASET_PATH="${DATASET_PATH:?DATASET_PATH must be set (the agent-built dataset)}"
OUTPUT_DIR="${OUTPUT_DIR:-/output}"
mkdir -p "$OUTPUT_DIR"

# ── model family (public Qwen3-Omni-30B MoE) ─────────────────────────────────
# Defaults match the public MoE model. MTP is disabled by default; operators
# can override these values for a compatible model.
MODEL_TYPE="${MODEL_TYPE:-qwen3_omni_moe}"
MTP_NUM_LAYERS="${MTP_NUM_LAYERS:-0}"

# ── step control ─────────────────────────────────────────────────────────────
# TRAIN_ITERS and NUM_TRAIN_EPOCHS are mutually exclusive in megatron (epochs
# overwrites train_iters), so pass exactly one.
if [ -n "${TRAIN_ITERS:-}" ]; then
    ITER_ARG="--train_iters ${TRAIN_ITERS}"
else
    ITER_ARG="--num_train_epochs ${NUM_TRAIN_EPOCHS:-2}"
fi

# ── runtime env ──────────────────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export ENABLE_AUDIO_OUTPUT=0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_NVLS_ENABLE=0
export NCCL_CUMEM_ENABLE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Video-decode memory caps: a single long/high-res clip can decode to tens of GB
# in a dataloader worker and OOM-kill a rank (system RAM, not GPU). Cap frames to
# 128 (= full 2fps for <=64s clips) and per-frame pixels to ~2536^2. Both
# overridable per run.
export FPS_MAX_FRAMES=${FPS_MAX_FRAMES:-128}
export FPS_MIN_FRAMES=${FPS_MIN_FRAMES:-4}
export QWEN_OMNI_MEGATRON_MAX_PIXELS=${QWEN_OMNI_MEGATRON_MAX_PIXELS:-6422528}

# ── train ────────────────────────────────────────────────────────────────────
echo "[$(date)] model=$MODEL_PATH  type=$MODEL_TYPE  mtp=$MTP_NUM_LAYERS"
echo "[$(date)] dataset=$DATASET_PATH  output=$OUTPUT_DIR"
echo "[$(date)] starting megatron sft ..."
NNODES=${NNODES:-${WORLD_SIZE:-1}} \
NODE_RANK=${NODE_RANK:-${RANK:-0}} \
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1} \
MASTER_PORT=${MASTER_PORT:-29500} \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=${NPROC_PER_NODE:-8} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} \
megatron sft \
    --model "$MODEL_PATH" \
    --model_type "$MODEL_TYPE" \
    --save_safetensors true \
    --dataset "$DATASET_PATH" \
    --split_dataset_ratio 0.01 \
    --load_from_cache_file true \
    --tuner_type full \
    --attn_impl flash_attn \
    --tensor_model_parallel_size ${TP:-8} \
    --pipeline_model_parallel_size ${PP:-1} \
    --expert_model_parallel_size ${EP:-8} \
    --expert_tensor_parallel_size 1 \
    --sequence_parallel true \
    --mtp_num_layers "$MTP_NUM_LAYERS" \
    --packing false \
    --lazy_tokenize true \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner false \
    --micro_batch_size 1 \
    --global_batch_size ${GBS:-48} \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --optimizer_cpu_offload true \
    --optimizer_offload_fraction 1.0 \
    --finetune true \
    --lr ${LR:-3e-6} \
    --lr_warmup_fraction 0.12 \
    --min_lr 1e-7 \
    ${ITER_ARG} \
    --max_length ${MAX_LENGTH:-8192} \
    --context_parallel_size ${CP:-1} \
    --truncation_strategy ${TRUNCATION_STRATEGY:-left} \
    --save_strategy ${SAVE_STRATEGY:-epoch} \
    --save_steps ${SAVE_STEPS:-50} \
    --eval_steps 100 \
    --dataloader_num_workers ${DATALOADER_NUM_WORKERS:-1} \
    --dataloader_prefetch_factor 1 \
    --dataloader_persistent_workers false \
    --logging_steps 1 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc ${DATASET_NUM_PROC:-16} \
    --loss_scale hermes \
    --output_dir "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/train_full_$(date +%Y%m%d_%H%M%S).log"

# NOTE: if you build omni-train from a base WITHOUT NVIDIA apex, add
#   --gradient_accumulation_fusion false
# to the command above (Megatron-SWIFT runs without apex that way). The default
# ModelScope base image ships apex, so no flag is needed.

# success sentinel (only reached if the pipeline above succeeded under pipefail);
# the caller polls $OUTPUT_DIR/DONE then stages the HF checkpoint as final_model.
echo "OK $(date)" > "$OUTPUT_DIR/DONE"
echo "[$(date)] training done -> $OUTPUT_DIR (DONE written)"
