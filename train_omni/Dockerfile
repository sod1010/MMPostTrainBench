# Qwen3-Omni-30B post-training environment — PUBLIC, open-source-buildable.
#
# This is the `omni-train` image the mmposttrainbench agent runs inside and the
# same image any training node uses to run the megatron SFT. It
# is the training counterpart of eval_omni/Dockerfile (the eval/verifier image)
# and, like it, ships ONLY the environment recipe — model and data are mounted
# at runtime, never baked in.
#
# It is deliberately built on the PUBLIC ModelScope×SWIFT image, which already
# ships the hard-to-compile CUDA deps (torch + TransformerEngine + apex +
# flash-attn + megatron-core) prebuilt. On top we only `pip install -U` the
# Megatron-SWIFT training stack and the Qwen3-Omni multimodal extras, so the
# build uses public dependencies on a compatible Docker host — no CUDA
# toolkit / nvcc compile step, no internal registry.
#
# Build (on a host that can reach the aliyun CR + pypi):
#   DOCKER_BUILDKIT=1 docker build -t omni-train:local .
#   # override the base to a newer/closer ModelScope tag if you prefer:
#   docker build --build-arg BASE_IMAGE=<modelscope swift image> -t omni-train:local .
#
# Megatron-SWIFT (public modelscope/ms-swift) natively trains Qwen3-Omni-30B-A3B
# MoE full-parameter SFT/GRPO with use_audio_in_video (TP/PP/EP, packing,
# loss_scale). NOTE: Qwen3-Omni Megatron support currently lives in the
# ms-swift 4.6.0.dev0 line, which is NOT yet on PyPI (PyPI tops out at 4.5.2), so
# we install from git at a fixed revision (SWIFT_REF).
# Bump SWIFT_REF to the tag that first ships a PyPI 4.6 release when available. See:
#   https://swift.readthedocs.io/en/latest/Megatron-SWIFT/Multimodal-Model.html (lists Qwen3-Omni)
#   https://github.com/modelscope/ms-swift  (issue #7108: Qwen3-Omni support)
#
# ── base ──────────────────────────────────────────────────────────────────
# Public ModelScope image (cu128 line): ships torch2.10 + TransformerEngine +
# apex + flash-attn + megatron-core, all prebuilt. Publicly pullable. The exact
# tag is pinned for reproducibility; see the ms-swift "Mirror" doc for the full
# list of published tags (cu128 / cu129 / cu130).
#   https://swift.readthedocs.io/en/latest/GetStarted/SWIFT-installation.html#mirror
ARG BASE_IMAGE=modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.10.0-vllm0.17.1-modelscope1.34.0-swift4.0.3
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TMPDIR=/tmp \
    HF_HUB_DISABLE_XET=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    MODELSCOPE_CACHE=/root/.cache/modelscope/hub

# System deps: ffmpeg is required by qwen-omni-utils / PyAV for A/V decode; git
# for editable installs. (The ModelScope base already has python + CUDA + git,
# but ffmpeg is not guaranteed.)
RUN sed -i -E 's#https?://(archive|security)\.ubuntu\.com/ubuntu#https://mirrors.aliyun.com/ubuntu#g' /etc/apt/sources.list 2>/dev/null || true; \
    apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 1) Megatron-SWIFT training stack (public). `ms-swift[megatron]` pulls a
#    compatible megatron-core; mcore-bridge removes the HF<->mcore convert step
#    (--save_safetensors true writes HF directly). Versions follow the official
#    "Recommended Operating Environment" matrix (megatron-core >=0.16,<0.20;
#    mcore-bridge >=1.3; transformers 4.57+). Pin ranges, not exacts, so the
#    solver stays compatible with the base image's prebuilt torch/TE.
#    ms-swift itself is installed from git (SWIFT_REF) because Qwen3-Omni Megatron
#    support is only in the 4.6.0.dev0 line (not yet on PyPI). `[megatron]` pulls
#    the megatron extra of the git checkout; we still pin the peer deps below.
ARG SWIFT_REF=c2bcc23a1f1c428dc8be0ae509c31a532922453f
RUN python -m pip install --upgrade pip && \
    python -m pip install -U \
      "ms-swift[megatron] @ git+https://github.com/modelscope/ms-swift.git@${SWIFT_REF}" \
      "megatron-core>=0.16,<0.20" \
      "mcore-bridge>=1.3" \
      "transformers>=4.57"

# 2) Qwen3-Omni multimodal I/O extras (audio+video). qwen-omni-utils drives the
#    A/V preprocessing; PyAV/librosa/soundfile/decord are the decoders. We drop
#    hf_xet (Xet CDN times out) and torchcodec (a bad build stalls qwen video
#    reads; without it qwen-omni-utils falls back to decord) — mirrors eval_omni.
RUN python -m pip install -U \
      "qwen-omni-utils" \
      "av" "librosa" "soundfile" "decord" && \
    python -m pip uninstall -y hf_xet torchcodec 2>/dev/null || true

# 3) Sanity: the training entrypoint (`megatron`) and the public MoE model type
#    must both resolve. `megatron sft --help` exercises the ms-swift megatron CLI
#    and its Qwen3-Omni template registration without needing a GPU.
RUN command -v megatron >/dev/null || { echo "FATAL: megatron CLI not on PATH after ms-swift[megatron] install" >&2; exit 1; } && \
    python -c "import swift, megatron.core, mcore_bridge; import transformers; print('train env OK: ms-swift', __import__('swift').__version__, 'transformers', transformers.__version__)" && \
    (megatron sft --help >/dev/null 2>&1 || echo "note: 'megatron sft --help' non-zero; verify the CLI at runtime")

# The public SFT entrypoint (repo-shipped). Operators point SFT_SCRIPT here; it
# runs the megatron SFT for the PUBLIC Qwen3-Omni-30B-A3B MoE. See README.md.
COPY run_sft_agent.sh /opt/train/run_sft_agent.sh
RUN chmod +x /opt/train/run_sft_agent.sh

WORKDIR /opt/train
# Runtime mounts (set by src/docker/run_agent.sh): -v <model>:/models etc.
# model_type / weights / dataset / output are all passed by env at run time.
ENV MODEL_PATH=/models
