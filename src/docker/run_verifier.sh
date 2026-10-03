#!/bin/bash
# SUCCESS CRITERION #2 — the verifier container.
# Stage a final_model into the workspace, then run the FULL verifier
# orchestrator (/tests/test.sh) inside the verifier container exactly as
# harbor's separate-verifier mode would. Confirms test.sh produces
# metrics.json + reward.txt + contamination/disallowed judgement files.
#
# By default final_model = the base Qwen3-Omni-30B (a floor score, proves the
# pipeline). Point FINAL_MODEL_SRC at an agent's trained output to score it.
#
# We bypass the image's log-streamer ENTRYPOINT with
# `--entrypoint /bin/bash ... /tests/test.sh` (harbor keeps the streamer as
# PID 1 and invokes test.sh separately; for direct docker runs we just exec it).
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.env"

FINAL_MODEL_SRC="${FINAL_MODEL_SRC:-$MODEL_DIR}"
FINAL_MODEL_DST="$WORKSPACE_HOST/final_model"

mkdir -p "$WORKSPACE_HOST" "$LOGS_HOST/verifier"

# --- stage final_model into the workspace -----------------------------------
if [ -d "$FINAL_MODEL_DST" ] && [ -f "$FINAL_MODEL_DST/config.json" ]; then
    echo "=== final_model already staged at $FINAL_MODEL_DST (reusing) ==="
else
    echo "=== staging final_model from $FINAL_MODEL_SRC ==="
    [ -f "$FINAL_MODEL_SRC/config.json" ] || { echo "ERROR: $FINAL_MODEL_SRC/config.json missing"; exit 1; }
    rm -rf "$FINAL_MODEL_DST"
    mkdir -p "$FINAL_MODEL_DST"
    # Hardlink-copy when on the same filesystem (near-free, no 68G duplication);
    # fall back to a full copy across filesystems.
    if cp -al "$FINAL_MODEL_SRC"/. "$FINAL_MODEL_DST"/ 2>/dev/null; then
        echo "    staged via hardlinks (same filesystem)"
    else
        echo "    hardlink failed (cross-fs); falling back to full copy..."
        cp -a "$FINAL_MODEL_SRC"/. "$FINAL_MODEL_DST"/
    fi
fi

# --- run test.sh in the verifier container ----------------------------------
echo ""
echo "=== running /tests/test.sh in verifier container ==="
echo "    verifier image : $VERIFIER_IMAGE"
echo "    workspace      : $WORKSPACE_HOST -> /home/agent/workspace"
echo "    logs           : $LOGS_HOST -> /logs"
echo "    judge          : $( [ -n "$CODEX_API_KEY" ] && echo 'enabled' || echo 'skipped (no CODEX_API_KEY)')"

# Pilot knobs (optional):
#   VERIFIER_LIMIT   — eval sample count; the baked test.sh hardcodes -1 (full),
#                      but a limit-parameterized variant honors this. Default -1.
#   VERIFIER_TEST_SH — host path to a test.sh to mount over /tests/test.sh
#                      (e.g. the limit-parameterized pilot copy). Unset = baked.
# BENCH — run ANY bench's bundle without rebuilding a per-bench image: mount that
#   bench's self-contained tests/ (evaluate.py + lmms_common + templates + metadata
#   + contamination_judge + test.sh) over the image's baked /tests. The mmptb-verifier
#   image (FROM omni-eval + codex) supplies the runtime (runners, lmms-eval, python,
#   judge); only /tests is swapped per bench. metadata.json in that tests/ selects the
#   benchmark for test.sh. Unset = use the image's baked bundle (mmau).
BENCH_MOUNT=()
if [ -n "${BENCH:-}" ]; then
    BENCH_TESTS="${BENCH_TASK_DIR:-$REPO_ROOT/harbor_tasks/mmposttrainbench-${BENCH}-qwen3-omni-30b/tests}"
    [ -d "$BENCH_TESTS" ] || { echo "ERROR: BENCH=$BENCH tests dir not found: $BENCH_TESTS (generate it with: python src/harbor_adapter/run_adapter.py -b $BENCH -m qwen3-omni-30b -o harbor_tasks)"; exit 1; }
    echo "    bench          : $BENCH  (/tests <- $BENCH_TESTS)"
    BENCH_MOUNT=(-v "$BENCH_TESTS":/tests:ro)
fi

TESTSH_MOUNT=()
if [ -n "${VERIFIER_TEST_SH:-}" ]; then
    [ -f "$VERIFIER_TEST_SH" ] || { echo "ERROR: VERIFIER_TEST_SH=$VERIFIER_TEST_SH not found"; exit 1; }
    echo "    test.sh        : OVERRIDE $VERIFIER_TEST_SH (VERIFIER_LIMIT=${VERIFIER_LIMIT:--1})"
    TESTSH_MOUNT=(-v "$VERIFIER_TEST_SH":/tests/test.sh:ro)
fi

# Pass credentials by environment name, never as Docker command-line values.
export CODEX_API_KEY="${CODEX_API_KEY:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"

docker run --rm --gpus "$GPUS" \
    --shm-size=32g \
    -e HF_HOME=/hf_cache \
    -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 -e HF_HUB_DISABLE_XET=1 \
    -e TMPDIR=/tmp \
    -e VERIFIER_LIMIT="${VERIFIER_LIMIT:--1}" \
    -e EVAL_SPLIT="${VERIFIER_SPLIT:-eval}" \
    -e MMPTB_ROLE=verifier \
    -e LMMS_LOG_SAMPLES=1 \
    -e CODEX_API_KEY \
    -e OPENAI_API_KEY \
    -v "$WORKSPACE_HOST":/home/agent/workspace \
    -v "$LOGS_HOST":/logs \
    -v "$HF_CACHE_DIR":/hf_cache \
    -v "$DATA_DIR":/data:ro \
    "${BENCH_MOUNT[@]}" \
    "${TESTSH_MOUNT[@]}" \
    --entrypoint /bin/bash \
    "$VERIFIER_IMAGE" /tests/test.sh

echo ""
echo "=== verifier outputs ==="
echo "--- metrics.json ---"; cat "$LOGS_HOST/verifier/metrics.json" 2>&1 || echo "(missing)"
echo ""
echo "--- reward.txt ---"; cat "$LOGS_HOST/verifier/reward.txt" 2>&1 || echo "(missing)"
echo ""
echo "--- judgement files ---"
cat "$LOGS_HOST/verifier/contamination_judgement.txt" 2>&1 || echo "(no contamination_judgement)"
cat "$LOGS_HOST/verifier/disallowed_model_judgement.txt" 2>&1 || echo "(no disallowed_model_judgement)"
