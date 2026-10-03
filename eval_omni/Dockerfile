# Qwen3-Omni-30B 多 benchmark 评测环境 —— 可交付镜像。
# 固化 2026-08 逐坑验证过的 env 配方(见 ../../agentevaluation/docs/07)。
# 镜像 = 隔离核(环境+harness+入口);模型与数据不进镜像,运行时挂载(见 README)。
#
# 构建(在有 docker registry 访问的机器上 build):
#   DOCKER_BUILDKIT=1 docker build -t omni-eval:local .
# base 可切换(若拉不到 docker.io,build 时用 --build-arg 指向任一可拉的
# CUDA 12.8 + cudnn base):
#   docker build --build-arg BASE_IMAGE=<your CUDA12.8 base> ...
ARG BASE_IMAGE=nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TMPDIR=/tmp \
    HF_HUB_DISABLE_XET=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

# 系统依赖:python + ffmpeg(qwen-omni-utils/av 需要)+ git(editable 安装)
# PY 可切换:验证过的配方用 python3.12(ACR base 自带);docker.io 的 jammy
# (ubuntu22.04)base 无 3.12,用其自带 python3(3.10) —— 评测栈 transformers
# 4.57 / torch 2.11 cu128 / datasets 3.6 均兼容 3.10。build 时 --build-arg PY=...
ARG PY=python3
# archive.ubuntu.com is not mirrored here (crawls); point apt at the aliyun
# ubuntu mirror so apt-get update + ffmpeg/python install are fast.
RUN sed -i -E 's#https?://(archive|security)\.ubuntu\.com/ubuntu#https://mirrors.aliyun.com/ubuntu#g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
        ${PY} ${PY}-dev python3-pip \
        ffmpeg git ca-certificates && \
    ln -sf /usr/bin/${PY} /usr/local/bin/python && \
    ln -sf /usr/bin/${PY} /usr/local/bin/python3 && \
    rm -rf /var/lib/apt/lists/*

# 1) torch cu128(驱动 570/CUDA12.8;从 aliyun pytorch-wheels 取 cu128 wheel,依赖走 aliyun pypi)
RUN python -m pip install --upgrade pip && \
    python -m pip install \
      --find-links https://mirrors.aliyun.com/pytorch-wheels/cu128/ \
      torch==2.11.0 torchvision==0.26.0

# 2) 评测依赖(精确 pin;关键:transformers 4.57 修多卡 device_map;datasets 3.6 音频走 soundfile)
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt && \
    python -m pip uninstall -y hf_xet torchcodec
# ↑ 显式移除 hf_xet(Xet CDN 超时)与 torchcodec(坏版本会拖垮 qwen_omni_utils 视频读取;缺它回退 decord)

# 3) harness:lmms-eval(editable)+ 官方 repo 脚本 + 官方对齐 self-runner
COPY harness_repos/ /opt/eval/harness_repos/
COPY runners/ /opt/eval/runners/
COPY entrypoint.sh /opt/eval/entrypoint.sh
RUN python -m pip install -c /tmp/requirements.txt -e /opt/eval/harness_repos/lmms-eval && \
    python -m pip uninstall -y hf_xet torchcodec && \
    chmod +x /opt/eval/entrypoint.sh

# 校验:关键类 + 无 torchcodec 回退
RUN python -c "from transformers import Qwen3OmniMoeForConditionalGeneration; import datasets,lmms_eval,qwen_omni_utils; print('env OK: transformers',__import__('transformers').__version__,'datasets',datasets.__version__)"

WORKDIR /opt/eval
# 运行时挂载:-v <model>:/models  -v <data>:/data  -v <hf_cache>:/hf_cache
ENV MODEL_PATH=/models HF_HOME=/hf_cache EVAL_DATA_ROOT=/data
ENTRYPOINT ["/opt/eval/entrypoint.sh"]
CMD ["--help"]
