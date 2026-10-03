#!/bin/bash
# Build the four images the omni loop needs, in dependency order:
#   1. omni-train:local     <- train_omni/ (public ms-swift Megatron-SWIFT env;
#                              base of the agent image and any training node)
#   2. omni-eval:local      <- eval_omni/  (eval-deliver isolation core; single
#                              source of truth for the pinned omni eval env)
#   3. mmptb-verifier:local <- <TASK_DIR>/tests        (FROM omni-eval + codex + /tests)
#   4. mmptb-agent:local    <- <TASK_DIR>/environment  (FROM omni-train + CLI agents)
#
# Run on a host with a reachable docker daemon (aliyun CR + pypi reachable).
# Usage: bash build_images.sh [--skip-train] [--skip-agent]
#   --skip-train : you already built/pushed omni-train and set OMNI_TRAIN_IMAGE
#                  to a registry tag your training nodes can pull (recommended for
#                  a shared setup; the agent build still FROMs $OMNI_TRAIN_IMAGE).
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.env"

# Check all local evaluator inputs before building any large image.
python3 "$REPO_ROOT/eval_omni/check_build_inputs.py"
python3 "$REPO_ROOT/scripts/sync_eval_bundles.py" --check

SKIP_TRAIN=0
SKIP_AGENT=0
for a in "$@"; do
    [ "$a" = "--skip-train" ] && SKIP_TRAIN=1
    [ "$a" = "--skip-agent" ] && SKIP_AGENT=1
done

# Some remote/proxied daemons have no buildx, so default to the legacy builder.
# The Dockerfiles use only standard instructions (COPY/RUN/ARG/ENV), so legacy
# is sufficient. Override with DOCKER_BUILDKIT=1 where buildx exists.
export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-0}"

if [ "$SKIP_TRAIN" -eq 0 ]; then
    echo "=== [1/4] omni-train base (public Qwen3-Omni-30B training env) ==="
    echo "    base=$OMNI_TRAIN_BASE  ->  $OMNI_TRAIN_IMAGE"
    DOCKER_BUILDKIT=$DOCKER_BUILDKIT docker build \
        --build-arg BASE_IMAGE="$OMNI_TRAIN_BASE" \
        -t "$OMNI_TRAIN_IMAGE" \
        "$REPO_ROOT/train_omni"
else
    echo "=== [1/4] omni-train build SKIPPED (--skip-train; using $OMNI_TRAIN_IMAGE as-is) ==="
fi

echo ""
echo "=== [2/4] omni-eval base (eval-deliver isolation core) ==="
echo "    base=$OMNI_EVAL_BASE  ->  $OMNI_EVAL_IMAGE"
DOCKER_BUILDKIT=$DOCKER_BUILDKIT docker build \
    --build-arg BASE_IMAGE="$OMNI_EVAL_BASE" \
    --build-arg PY="${PY:-python3}" \
    -t "$OMNI_EVAL_IMAGE" \
    "$REPO_ROOT/eval_omni"

echo ""
echo "=== [3/4] verifier (FROM $OMNI_EVAL_IMAGE) ==="
echo "    context=$TASK_DIR/tests  ->  $VERIFIER_IMAGE"
[ -d "$TASK_DIR/tests" ] || { echo "ERROR: $TASK_DIR/tests missing — run run_adapter.py first"; exit 1; }
DOCKER_BUILDKIT=$DOCKER_BUILDKIT docker build \
    --build-arg OMNI_EVAL_IMAGE="$OMNI_EVAL_IMAGE" \
    -t "$VERIFIER_IMAGE" \
    "$TASK_DIR/tests"

if [ "$SKIP_AGENT" -eq 0 ]; then
    echo ""
    echo "=== [4/4] agent (FROM $OMNI_TRAIN_IMAGE) ==="
    echo "    context=$TASK_DIR/environment  ->  $AGENT_IMAGE"
    [ -d "$TASK_DIR/environment" ] || { echo "ERROR: $TASK_DIR/environment missing"; exit 1; }
    DOCKER_BUILDKIT=$DOCKER_BUILDKIT docker build \
        --build-arg OMNI_TRAIN_IMAGE="$OMNI_TRAIN_IMAGE" \
        -t "$AGENT_IMAGE" \
        "$TASK_DIR/environment"
else
    echo ""
    echo "=== [4/4] agent build SKIPPED (--skip-agent) ==="
fi

echo ""
echo "=== build done ==="
docker images | grep -E "omni-train|omni-eval|mmptb-" || true
