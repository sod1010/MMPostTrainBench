#!/bin/bash
# Run the AGENT half of the loop: an agent produces $WORKSPACE_HOST/final_model.
#
# WORKSPACE SHADOWING NOTE: the agent image bakes the task files into
# /home/agent/workspace via `COPY .`. We bind-mount $WORKSPACE_HOST there so
# the verifier (a separate container) can read final_model from the same host
# dir — but a bind mount SHADOWS the baked files. So we first SEED
# $WORKSPACE_HOST from the bundle's environment/ (the exact set `COPY .`
# places), then mount it. The agent then works in a host-backed workspace and
# final_model persists for the verifier with no docker-cp of 68G weights.
#
# AGENT_ENGINE:
#   placeholder (default) — no multi-hour GPU run; stage final_model = base
#     model (hardlink). Proves the loop plumbing end-to-end (seed -> workspace
#     -> final_model -> verifier) cheaply. This is what pilot criterion #3 uses.
#   claude-code | codex | gemini — run the real CLI agent headless with
#     instruction.md as the prompt and the task.toml timeout as the budget.
#
# AGENT_MODEL (optional) — pin the agent's model id, e.g.
#   AGENT_ENGINE=claude-code AGENT_MODEL=claude-opus-4-5
#   AGENT_ENGINE=codex       AGENT_MODEL=gpt-5.1
# Left empty, each CLI picks its own default model. To drive a gateway-fronted
# or self-hosted model, also point the matching *_BASE_URL (see the docker run
# -e list below) at it: anything speaking the Anthropic or OpenAI wire protocol
# works, including a local vLLM/SGLang server.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.env"
if [ -n "${BENCH:-${BENCHMARK:-}}" ]; then
    selected_bench="${BENCH:-$BENCHMARK}"
    case "$selected_bench" in mmau|mmar|mmmu_pro|video_mmmu|videomme_v2|jointavbench|omnivideobench|mmswe) ;; *) echo "Invalid benchmark" >&2; exit 2;; esac
    TASK_DIR="$REPO_ROOT/harbor_tasks/mmposttrainbench-${selected_bench}-qwen3-omni-30b"
fi
python3 "$HERE/runtime_paths.py" || exit 2

AGENT_ENGINE="${AGENT_ENGINE:-placeholder}"
AGENT_MODEL="${AGENT_MODEL:-}"          # optional model-id pin; empty = CLI default
FINAL_MODEL_DST="$WORKSPACE_HOST/final_model"

# --- seed the host workspace from the bundle's environment/ ------------------
seed_workspace() {
    mkdir -p "$WORKSPACE_HOST"
    echo "=== seeding workspace from $TASK_DIR/environment ==="
    # Mirror the agent Dockerfile's `COPY .` minus the SAME strip list it applies
    # (see environment/Dockerfile), plus instruction.md (harbor feeds it as the
    # prompt; we drop it in the workspace so a headless agent can read it too).
    # Seed everything else rather than an allowlist of a few names: the bind mount
    # SHADOWS the baked workspace, so anything missed here simply does not exist at
    # runtime. An allowlist silently broke the agent's self-evaluation whenever the
    # bundle grew a file — omnivideobench's evaluate.py imports split_util.py
    # unguarded, and the lmms benches (mmmu_pro/video_mmmu/videomme_v2) delegate to
    # lmms_common/evaluate.py.
    # contamination_judge.py is deliberately NOT seeded: it is the verifier's own
    # anti-cheat probe (only tests/test.sh calls it), and the agent has no use for
    # it. The agent Dockerfile strips it for the same reason.
    local strip=(Dockerfile .dockerignore entrypoint.sh system_monitor.sh \
                 requirements-direct.txt contamination_judge.py)
    shopt -s dotglob nullglob
    for p in "$TASK_DIR/environment"/*; do
        local b skip=0
        b="$(basename "$p")"
        for s in "${strip[@]}"; do [ "$b" = "$s" ] && skip=1 && break; done
        [ "$skip" = 1 ] || cp -a "$p" "$WORKSPACE_HOST/"
    done
    shopt -u dotglob nullglob
    [ -e "$TASK_DIR/instruction.md" ] && cp -a "$TASK_DIR/instruction.md" "$WORKSPACE_HOST/"
    chmod +x "$WORKSPACE_HOST/timer.sh" 2>/dev/null || true
}

stage_base_as_final() {
    echo "=== [placeholder] staging base model as final_model ==="
    [ -f "$MODEL_DIR/config.json" ] || { echo "ERROR: $MODEL_DIR/config.json missing"; exit 1; }
    rm -rf "$FINAL_MODEL_DST"; mkdir -p "$FINAL_MODEL_DST"
    if cp -al "$MODEL_DIR"/. "$FINAL_MODEL_DST"/ 2>/dev/null; then
        echo "    staged via hardlinks (same filesystem)"
    else
        echo "    hardlink failed (cross-fs); full copy (this is ~68G, slow)..."
        cp -a "$MODEL_DIR"/. "$FINAL_MODEL_DST"/
    fi
}

# agent timeout (seconds) from task.toml, for the real-agent budget.
agent_timeout() {
    python3 - "$TASK_DIR/task.toml" <<'PY' 2>/dev/null || echo 86400
import re,sys
t=open(sys.argv[1]).read()
m=re.search(r'\[agent\][^\[]*?timeout_sec\s*=\s*([0-9.]+)', t, re.S)
print(int(float(m.group(1))) if m else 86400)
PY
}

seed_workspace

if [ "$AGENT_ENGINE" = "placeholder" ]; then
    stage_base_as_final
    echo "=== [placeholder] agent done; final_model at $FINAL_MODEL_DST ==="
    exit 0
fi

# --- real CLI agent ----------------------------------------------------------
TIMEOUT_SEC="$(agent_timeout)"
mkdir -p "$LOGS_HOST/agent"
echo "=== running real agent: engine=$AGENT_ENGINE timeout=${TIMEOUT_SEC}s ==="
echo "    agent image : $AGENT_IMAGE"
echo "    base model  : $MODEL_DIR -> /models (agent fine-tunes FROM here)"

# Per-engine headless command. The agent reads /home/agent/workspace/instruction.md,
# fine-tunes /models, and must leave weights at /home/agent/workspace/final_model.
case "$AGENT_ENGINE" in
    claude-code)
        MODEL_FLAG=""; [ -n "$AGENT_MODEL" ] && MODEL_FLAG="--model $AGENT_MODEL"
        AGENT_CMD_DEFAULT="claude $MODEL_FLAG"' -p "$(cat /home/agent/workspace/instruction.md)" --dangerously-skip-permissions' ;;
    codex)
        MODEL_FLAG=""; [ -n "$AGENT_MODEL" ] && MODEL_FLAG="-m $AGENT_MODEL"
        AGENT_CMD_DEFAULT="codex $MODEL_FLAG"' -a never exec --yolo "$(cat /home/agent/workspace/instruction.md)"' ;;
    gemini)
        MODEL_FLAG=""; [ -n "$AGENT_MODEL" ] && MODEL_FLAG="-m $AGENT_MODEL"
        AGENT_CMD_DEFAULT="gemini $MODEL_FLAG"' --yolo -p "$(cat /home/agent/workspace/instruction.md)"' ;;
    *) echo "ERROR: unknown AGENT_ENGINE=$AGENT_ENGINE"; exit 1 ;;
esac
AGENT_CMD="${AGENT_CMD:-$AGENT_CMD_DEFAULT}"

# --- DATA-SOURCE ISOLATION (anti-cheat) -------------------------------------
# The agent must NOT see the eval test sets. We deliberately do NOT mount the
# verifier's $HF_CACHE_DIR (holds mmau/mmmu_pro/mmar/video datasets) or the eval
# $DATA_DIR into the agent container. The agent gets:
#   /models   base weights (ro)              — fine-tunes FROM here
#   /home/agent/workspace                    — its scratch + must emit final_model
#   /hf_cache a SEPARATE, clean HF cache      — for model/dep pulls, NOT eval data
#   /train_data (optional) agent training data — NEVER the eval set
# The eval hf_cache/data are mounted ONLY in the verifier (run_verifier.sh).
# The agent's self-evaluation is also pinned to the VAL split (EVAL_SPLIT=val
# below); the sealed EVAL split is scored only by run_verifier.sh, which passes
# EVAL_SPLIT=eval explicitly. Note this is a default, not a secret: split_util's
# keep() is symmetric, so val and eval determine each other.
AGENT_HF_CACHE="${AGENT_HF_CACHE:-$MMPTB_ROOT/agent_hf_cache}"   # clean, agent-only
mkdir -p "$AGENT_HF_CACHE"
TRAIN_MOUNT=()
[ -n "${AGENT_TRAIN_DATA:-}" ] && TRAIN_MOUNT=(-v "$AGENT_TRAIN_DATA":/train_data:ro)

# Pass credentials by environment name, never as Docker command-line values.
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"
export ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export GEMINI_API_KEY="${GEMINI_API_KEY:-}"

docker run --rm --gpus "$GPUS" \
    --shm-size=64g \
    -e HF_HOME=/hf_cache -e HF_HUB_DISABLE_XET=1 -e TMPDIR=/tmp \
    -e BASE_MODEL_PATH=/models \
    -e EVAL_SPLIT="${AGENT_EVAL_SPLIT:-val}" \
    -e MMPTB_ROLE=agent \
    -e ANTHROPIC_API_KEY \
    -e ANTHROPIC_AUTH_TOKEN \
    -e ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-}" \
    -e ANTHROPIC_MODEL="${ANTHROPIC_MODEL:-}" \
    -e OPENAI_API_KEY \
    -e OPENAI_BASE_URL="${OPENAI_BASE_URL:-}" \
    -e GEMINI_API_KEY \
    -e GOOGLE_GEMINI_BASE_URL="${GOOGLE_GEMINI_BASE_URL:-}" \
    -v "$WORKSPACE_HOST":/home/agent/workspace \
    -v "$LOGS_HOST/agent":/logs/agent \
    -v "$MODEL_DIR":/models:ro \
    -v "$AGENT_HF_CACHE":/hf_cache \
    "${TRAIN_MOUNT[@]}" \
    --entrypoint /bin/bash \
    "$AGENT_IMAGE" -lc "date +%s > /timer_start; cd /home/agent/workspace; timeout ${TIMEOUT_SEC}s bash -lc '$AGENT_CMD' 2>&1 | tee /logs/agent/agent.txt || true"

if [ ! -f "$FINAL_MODEL_DST/config.json" ]; then
    echo "WARNING: agent did not leave a valid final_model/ (no config.json). "
    echo "         The verifier will score 0 for a missing model."
fi
echo "=== agent done; workspace at $WORKSPACE_HOST ==="
