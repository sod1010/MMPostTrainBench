#!/bin/bash
# Oracle reference solution (harbor `-a oracle`): submit the BASE model UNCHANGED as
# final_model. A post-training task has no trivial reference patch — the meaningful oracle
# is "no-op train = base baseline": it validates the whole build -> generate -> grade chain
# and reproduces the pinned base baseline WITHOUT spending any model budget. This mirrors
# src/docker/run_agent.sh:stage_base_as_final (the self-hosted docker-runner equivalent).
set -euo pipefail
# Resolve the provided base Qwen3-Omni weights (present locally in the task env; see
# instruction.md "base weights are already present locally"). Prefer explicit env, then
# the broker mount, then a couple of conventional locations.
BASE=""
for c in "${MMPTB_BASE_MODEL:-}" "${MODEL_DIR:-}" /input/base /models "$HOME/models/Qwen3-Omni-30B-A3B-Instruct"; do
  [ -n "$c" ] && [ -f "$c/config.json" ] && { BASE="$c"; break; }
done
[ -n "$BASE" ] || { echo "oracle: cannot locate base Qwen3-Omni weights (set MMPTB_BASE_MODEL or MODEL_DIR)"; exit 1; }
mkdir -p final_model
cp -a "$BASE"/. final_model/
echo "oracle: staged base model ($BASE) as final_model -> reproduces the base baseline (no training)"
