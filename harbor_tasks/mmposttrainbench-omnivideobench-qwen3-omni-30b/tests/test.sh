#!/bin/bash
set -e

# Operator-owned verifier entry point. Role variables prevent configuration
# mistakes; separate containers and mounts provide the actual isolation.
export MMPTB_ROLE=verifier
export EVAL_SPLIT="${EVAL_SPLIT:-eval}"

# PostTrainBench verification script
# Records an explicit unknown integrity status and runs 3-phase evaluation.
# Matches the original run_task.sh evaluation pipeline.
#
# Tamper-resistance design (harbor 0.7.0 separate-verifier mode):
#   - This script runs in a SEPARATE container from the agent (see
#     [verifier].environment_mode = "separate" in task.toml). The agent
#     never has shell or filesystem access to this container, so it
#     can't tamper with evaluate.py, templates/, the Python interpreter,
#     installed packages (vllm, inspect_evals, transformers), or this
#     script itself.
#   - All verifier-side files (evaluate.py, templates/, contamination_judge.py,
#     metadata.json, evaluation_code/, bfcl_evaluation_code.py) are
#     BAKED INTO the verifier image at build time (see tests/Dockerfile)
#     and live at /tests/.
#   - The agent's workspace at /home/agent/workspace is transferred from
#     the agent container by harbor as a configured artifact and
#     contains the agent's training scripts + final_model. The
#     operator reviews source in a separate static audit boundary;
#     evaluate.py reads /home/agent/workspace/final_model.
#   - The agent's final_model is the only file the verifier executes
#     code against (via vllm). Bad weights are penalized by the eval
#     score, not by tampering.

TESTS="/tests"
WORKSPACE="/home/agent/workspace"
LOGS_DIR="/logs/verifier"

mkdir -p "$LOGS_DIR"
# A new verification must never accept a previous run's score or reward.
rm -f "$LOGS_DIR/metrics.json" "$LOGS_DIR/reward.txt" "$LOGS_DIR/diag.jsonl"

echo "=== PostTrainBench Verifier ==="
echo "Tests dir: $TESTS"
echo "Workspace: $WORKSPACE"
echo "Logs dir: $LOGS_DIR"

# Check GPU availability
echo ""
echo "=== GPU Check ==="
nvidia-smi 2>&1 | tee "$LOGS_DIR/gpu_check.txt" || echo "nvidia-smi failed"

# Check if final_model exists in agent's workspace
echo ""
echo "=== Checking final_model ==="
if [ ! -d "$WORKSPACE/final_model" ]; then
    echo "ERROR: final_model directory not found"
    ls -la "$WORKSPACE" > "$LOGS_DIR/workspace_listing.txt" 2>&1
    echo '{"error": "final_model not found", "accuracy": 0}' > "$LOGS_DIR/metrics.json"
    echo "0" > "$LOGS_DIR/reward.txt"
    exit 0
fi

# Check if final_model has required files
echo "Contents of final_model:"
ls -la "$WORKSPACE/final_model" | tee "$LOGS_DIR/final_model_listing.txt"

if [ ! -f "$WORKSPACE/final_model/config.json" ]; then
    echo "ERROR: final_model/config.json not found - not a valid model"
    echo '{"error": "invalid model - no config.json", "accuracy": 0}' > "$LOGS_DIR/metrics.json"
    echo "0" > "$LOGS_DIR/reward.txt"
    exit 0
fi

# Show model config
echo ""
echo "=== Model config.json ==="
cat "$WORKSPACE/final_model/config.json" | head -50 | tee "$LOGS_DIR/model_config.txt"

# Check for tokenizer
echo ""
echo "=== Checking tokenizer files ==="
ls -la "$WORKSPACE/final_model/"*token* 2>/dev/null || echo "No tokenizer files found with 'token' in name"
ls -la "$WORKSPACE/final_model/"*.json 2>/dev/null || echo "No json files found"

# ============================================================
# Read metadata for benchmark and model info — from /tests, NOT workspace,
# so the agent can't redirect the verifier by overwriting metadata.json.
# ============================================================
BENCHMARK_ID=""
BENCHMARK_NAME=""
MODEL_ID=""

if [ -f "$TESTS/metadata.json" ]; then
    BENCHMARK_ID=$(python3 -c "import json; print(json.load(open('$TESTS/metadata.json'))['benchmark_id'])" 2>/dev/null || echo "")
    BENCHMARK_NAME=$(python3 -c "import json; print(json.load(open('$TESTS/metadata.json'))['benchmark_name'])" 2>/dev/null || echo "Unknown")
    MODEL_ID=$(python3 -c "import json; print(json.load(open('$TESTS/metadata.json'))['model_id'])" 2>/dev/null || echo "Unknown")
    echo "Benchmark ID: $BENCHMARK_ID"
    echo "Benchmark Name: $BENCHMARK_NAME"
    echo "Model: $MODEL_ID"
fi

# ============================================================
# Integrity audit status
# The legacy live Codex judge is disabled: running tools from an agent-controlled
# workspace with verifier credentials/data crosses the sealed evaluation boundary.
# Numeric evaluation is separate from integrity certification. The operator must
# perform the reviewed static workspace audit in a separate boundary before accepting
# this run; see docs/disclosure-security.md. Submitted verdict files are never read.
# ============================================================
echo ""
echo "=== Integrity audit requires operator review ==="
rm -f "$LOGS_DIR/judge_output.json"
printf '%s\n' 'unknown (legacy workspace judge disabled; operator audit required)' > "$LOGS_DIR/contamination_judgement.txt"
printf '%s\n' 'unknown (legacy workspace judge disabled; operator audit required)' > "$LOGS_DIR/disallowed_model_judgement.txt"
printf '%s\n' '{"verdict":"unknown","cheated":null,"reason":"operator_static_audit_required","certified":false}' > "$LOGS_DIR/integrity_status.json"

# ============================================================
# Evaluation with 3-phase retry logic
# Matches run_task.sh evaluation pipeline.
#
# evaluate.py is run from /tests (untamperable). Some evaluate.py scripts
# (arenahardwriting, healthbench) `from evaluation_code.X import Y`, so
# /tests must be cwd for the import to resolve. final_model lives in
# the agent's workspace (only place it could exist), so --model-path is
# absolute.
# ============================================================
echo ""
echo "=== Running evaluation on final_model ==="

cd "$TESTS"

EVAL_COUNTER=0

kill_gpu_processes() {
    echo "Killing GPU processes..."
    # Kill GPU-holding processes EXCEPT PID 1 (container init / dumb-init).
    # In Docker/Modal, the agent's vLLM process can get reparented to PID 1,
    # which still holds GPU memory when the verifier starts. Killing PID 1
    # would destroy the entire container.
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | grep -v '^$' \
        | while read pid; do
            if [ "$pid" -gt 1 ] 2>/dev/null; then
                kill -9 "$pid" 2>/dev/null || true
            fi
        done
    sleep 5
}

run_evaluation() {
    local max_tokens_arg="$1"
    local eval_num="$2"

    kill_gpu_processes

    set +e
    python3 "$TESTS/evaluate.py" \
        --model-path "$WORKSPACE/final_model" \
        --json-output-file "$LOGS_DIR/metrics.json" \
        --templates-dir "$TESTS/templates" \
        --limit "${VERIFIER_LIMIT:--1}" \
        ${max_tokens_arg} \
        2>&1 | tee "$LOGS_DIR/final_eval_${eval_num}.txt"
    local exit_code=${PIPESTATUS[0]}
    set -e
    return $exit_code
}

run_evaluation_with_retry() {
    local max_retries="$1"
    local max_tokens_arg="$2"

    for ((attempt=1; attempt<=max_retries; attempt++)); do
        sleep 5
        if [ -f "$LOGS_DIR/metrics.json" ]; then
            return 0
        fi

        EVAL_COUNTER=$((EVAL_COUNTER + 1))
        echo "Evaluation attempt $EVAL_COUNTER (phase attempt $attempt of $max_retries)"

        if run_evaluation "$max_tokens_arg" "$EVAL_COUNTER"; then
            if [ -f "$LOGS_DIR/metrics.json" ]; then
                return 0
            fi
        else
            rm -f "$LOGS_DIR/metrics.json"
        fi
    done

    return 1
}

# Determine token limit args per benchmark for phase 2 and 3
get_phase2_tokens() {
    case "$BENCHMARK_ID" in
        aime2025)    echo "--max-tokens 12000" ;;
        arenahardwriting) echo "--max-new-tokens 12288" ;;
        bfcl)        echo "--max-tokens 12000" ;;
        gpqamain)    echo "--max-tokens 12000" ;;
        gsm8k)       echo "--max-tokens 3000" ;;
        healthbench) echo "--max-new-tokens 12288" ;;
        humaneval)   echo "--max-tokens 3000" ;;
        *)           echo "" ;;
    esac
}

get_phase3_tokens() {
    case "$BENCHMARK_ID" in
        aime2025)    echo "--max-tokens 8000" ;;
        arenahardwriting) echo "--max-new-tokens 8192" ;;
        bfcl)        echo "--max-tokens 8000" ;;
        gpqamain)    echo "--max-tokens 8000" ;;
        gsm8k)       echo "--max-tokens 2000" ;;
        healthbench) echo "--max-new-tokens 8192" ;;
        humaneval)   echo "--max-tokens 2000" ;;
        *)           echo "" ;;
    esac
}

# Phase 1: up to 4 attempts with default tokens
echo ""
echo "--- Phase 1: default token limits (up to 4 attempts) ---"
run_evaluation_with_retry 4 "" || true

# Phase 2: up to 3 attempts with reduced tokens
PHASE2_TOKENS=$(get_phase2_tokens)
echo ""
echo "--- Phase 2: reduced tokens [${PHASE2_TOKENS}] (up to 3 attempts) ---"
run_evaluation_with_retry 3 "$PHASE2_TOKENS" || true

# Phase 3: up to 2 attempts with further reduced tokens
PHASE3_TOKENS=$(get_phase3_tokens)
echo ""
echo "--- Phase 3: further reduced tokens [${PHASE3_TOKENS}] (up to 2 attempts) ---"
run_evaluation_with_retry 2 "$PHASE3_TOKENS" || true

# ============================================================
# Extract accuracy and write reward
# ============================================================
echo ""
echo "=== Evaluation complete (${EVAL_COUNTER} total attempts) ==="

if [ -f "$LOGS_DIR/metrics.json" ]; then
    echo "metrics.json contents:"
    cat "$LOGS_DIR/metrics.json"

    # Accept only the named finite accuracy; malformed results are infra failures.
    if ! ACCURACY=$(python3 - "$LOGS_DIR/metrics.json" <<'PY_SCORE'
import json, math, sys
with open(sys.argv[1]) as f:
    metrics = json.load(f)
a = metrics.get("accuracy")
if metrics.get("error") or type(a) not in (int, float) or not math.isfinite(a) or not 0 <= a <= 1:
    raise SystemExit("invalid evaluation accuracy")
print(a)
PY_SCORE
    ); then
        rm -f "$LOGS_DIR/reward.txt" "$LOGS_DIR/metrics.json"
        echo "ERROR: invalid evaluation result; no reward emitted"
        exit 1
    fi
    echo "Accuracy: $ACCURACY"
    echo "$ACCURACY" > "$LOGS_DIR/reward.txt"
else
    echo "ERROR: metrics.json not created after all evaluation attempts; no reward emitted"
    rm -f "$LOGS_DIR/reward.txt"
    exit 1
fi

echo ""
echo "=== Verification complete ==="
echo "Results in $LOGS_DIR/"
ls -la "$LOGS_DIR/"
