#!/bin/bash
# SUCCESS CRITERION #3 — the end-to-end loop.
# agent (produces final_model) -> verifier (scores it -> reward.txt), the two
# halves running as SEPARATE containers sharing final_model via the host
# workspace dir. Defaults to AGENT_ENGINE=placeholder so the loop plumbing is
# proven without a multi-hour training run; set AGENT_ENGINE=claude-code (etc.)
# for a real run.
#
# Prereq: images built (bash build_images.sh) and a fresh workspace.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.env"

FRESH="${FRESH:-1}"   # 1 = wipe the workspace/logs before the run
if [ "$FRESH" = "1" ]; then
    echo "=== fresh run: clearing $WORKSPACE_HOST and $LOGS_HOST ==="
    rm -rf "$WORKSPACE_HOST" "$LOGS_HOST"
fi
mkdir -p "$WORKSPACE_HOST" "$LOGS_HOST"

echo ""
echo "################ AGENT ################"
bash "$HERE/run_agent.sh"

echo ""
echo "################ VERIFIER ################"
# run_verifier reuses the already-staged $WORKSPACE_HOST/final_model.
FINAL_MODEL_SRC="$WORKSPACE_HOST/final_model" bash "$HERE/run_verifier.sh"

echo ""
echo "################ ANTI-CHEAT GATE ################"
# Run the cheat gate over the AGENT's workspace (its code) + the raw reward:
# LLM judge (+ optional contamination); if cheating is detected the reward is
# overridden to the base-model baseline. Set GATE_DISABLE=1 to skip.
if [ -f "$LOGS_HOST/verifier/reward.txt" ]; then
    RAW="$(cat "$LOGS_HOST/verifier/reward.txt")"
    rm -f "$LOGS_HOST/verifier/reward_final.txt"
    if ! FINAL="$(GATE_METRICS="$LOGS_HOST/verifier/metrics.json" \
        GATE_REPORT="$LOGS_HOST/verifier/gate_report.json" \
        BENCHMARK="${BENCHMARK:-mmau}" bash "$HERE/cheat_gate.sh" \
        "${BENCHMARK:-mmau}" "$WORKSPACE_HOST" "$RAW" "${AGENT_TRAIN_DATA:-}")"; then
        echo "LOOP FAILED: gate or baseline protocol validation failed; no final reward" >&2
        exit 1
    fi
    echo "$FINAL" > "$LOGS_HOST/verifier/reward_final.txt"
    echo "raw=$RAW  final(after gate)=$FINAL"
    [ -f "$WORKSPACE_HOST/../gate_report.json" ] && echo "gate: $(cat "$WORKSPACE_HOST/../gate_report.json" 2>/dev/null | tr -d '\n')"
fi

echo ""
echo "################ LOOP RESULT ################"
if [ -f "$LOGS_HOST/verifier/reward_final.txt" ]; then
    echo "reward (post-gate): $(cat "$LOGS_HOST/verifier/reward_final.txt")"
    echo "LOOP OK: agent -> verifier -> anti-cheat gate -> reward end-to-end."
elif [ -f "$LOGS_HOST/verifier/reward.txt" ]; then
    echo "reward: $(cat "$LOGS_HOST/verifier/reward.txt") (gate skipped)"
    echo "LOOP OK: agent -> verifier -> reward.txt end-to-end."
else
    echo "LOOP FAILED: no reward.txt at $LOGS_HOST/verifier/"
    exit 1
fi
