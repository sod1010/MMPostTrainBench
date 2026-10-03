#!/bin/bash
# Baked into an operator-built, digest-pinned image; never mounted from agent input.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HOME=/tmp HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH=/opt/mmptb-service
case "$1" in
  train)
    export MMPTB_DATA=/input/data MMPTB_BASE_MODEL=/input/base MMPTB_OUTPUT=/output/model
    exec python3 -I /input/code/train.py
    ;;
  generate)
    exec python3 -I /opt/mmptb-service/generate.py
    ;;
  *) exit 64 ;;
esac
