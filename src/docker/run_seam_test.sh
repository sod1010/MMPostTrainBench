#!/bin/bash
# SUCCESS CRITERION #1 — the seam.
# Run the verifier image's /tests/evaluate.py DIRECTLY (bypassing test.sh)
# against the BASE Qwen3-Omni-30B, and confirm it emits {"accuracy": ...}.
# A limit-8 run should land near the eval-deliver smoke value (mmau ~0.375),
# proving the harbor evaluate.py contract is wired to the omni runner.
#
# This is the cheapest proof that "our harbor eval plugged in": no agent, no
# test.sh, just evaluate.py -> run_mmau_official.py -> normalized JSON.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.env"

OUT_HOST="${LOGS_HOST}/seam"
mkdir -p "$OUT_HOST"

echo "=== seam test: evaluate.py on base model, limit=$SEAM_LIMIT ==="
echo "    verifier image : $VERIFIER_IMAGE"
echo "    model          : $MODEL_DIR"
echo "    hf_cache       : $HF_CACHE_DIR"
echo "    out            : $OUT_HOST/seam_metrics.json"

docker run --rm --gpus "$GPUS" \
    --shm-size=32g \
    -e HF_HOME=/hf_cache \
    -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 -e HF_HUB_DISABLE_XET=1 \
    -e TMPDIR=/tmp \
    -v "$MODEL_DIR":/models:ro \
    -v "$HF_CACHE_DIR":/hf_cache \
    -v "$DATA_DIR":/data:ro \
    -v "$OUT_HOST":/out \
    --entrypoint /bin/bash \
    "$VERIFIER_IMAGE" -lc "\
        python3 /tests/evaluate.py \
          --model-path /models \
          --limit $SEAM_LIMIT \
          --json-output-file /out/seam_metrics.json"

echo ""
echo "=== seam_metrics.json ==="
cat "$OUT_HOST/seam_metrics.json"
echo ""
python3 - "$OUT_HOST/seam_metrics.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
acc = m.get("accuracy")
assert isinstance(acc, (int, float)), f"no numeric accuracy in {m}"
print(f"\nSEAM OK: accuracy={acc:.4f} (n={m.get('n')}, correct={m.get('correct')})")
print("Expected ~0.375 at limit=8 per eval-deliver smoke; a numeric value here "
      "means the harbor evaluate.py contract is wired to the omni runner.")
PY
