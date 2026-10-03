#!/bin/bash
# One-time ONLINE load_dataset warmup, run INSIDE the omni-eval container, to
# build the `datasets` arrow cache so the offline verifier can resolve the MMAU
# dataset. Needed because:
#   - `hf download --repo-type dataset` only populates the HUB cache; datasets'
#     dataset_module_factory still needs one online build to create the arrow
#     cache under $HF_HOME/datasets, after which HF_HUB_OFFLINE=1 works.
#
# Usage (from src/docker):  bash warmup_dataset.sh
# Prereq: $HF_CACHE_DIR already holds the hub-cached dataset.
# Public HTTPS uses the image's normal trust store. For a managed proxy, set
# CA (or CABUNDLE) to an operator-approved CA certificate; verification stays on.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.env"

CA="${CA:-${CABUNDLE:-}}"
CA_MOUNT=()
if [ -n "$CA" ]; then
    [ -f "$CA" ] || { echo "ERROR: configured CA certificate is not a file"; exit 1; }
    CA_MOUNT=(-v "$CA":/ca.crt:ro)
fi

# The in-container warmup body.
cat > "$MMPTB_ROOT/_warmup_body.sh" <<'EOF'
set -e
if [ -f /ca.crt ]; then
    cp /ca.crt /usr/local/share/ca-certificates/operator-root.crt
    cat /ca.crt >> "$(python -c 'import certifi; print(certifi.where())')"
    update-ca-certificates >/dev/null 2>&1 || true
fi
export HF_ENDPOINT=https://huggingface.co HF_HUB_DISABLE_XET=1 HF_HOME=/hf_cache
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE
python - <<'PY'
from datasets import load_dataset
d = load_dataset("lmms-lab-audio/mmau", split="test_mini")
print("WARMUP_OK rows:", d.num_rows, "cols:", d.column_names)
PY
echo "WARMUP_DONE"
EOF

echo "=== warming datasets cache in $OMNI_EVAL_IMAGE (online) ==="
docker run --rm \
    -v "$HF_CACHE_DIR":/hf_cache \
    "${CA_MOUNT[@]}" \
    -v "$MMPTB_ROOT/_warmup_body.sh":/warmup_body.sh:ro \
    --entrypoint bash "$OMNI_EVAL_IMAGE" /warmup_body.sh
