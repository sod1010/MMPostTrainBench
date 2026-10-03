#!/bin/bash
# mmposttrainbench — unified entrypoint. Run ONE bench with ONE agent.
#
#   bash run_task.sh <bench> <agent> [model]
#     bash run_task.sh mmau  claude-code anthropic.claude-opus-4-8
#     bash run_task.sh mmswe codex       gpt-5.1
#     bash run_task.sh mmau  oracle                 # self-test: NO model, must reproduce baseline
#
# Thin positional wrapper over src/docker/run_loop.sh (pure `docker run`, single
# GPU host, no scheduler). It maps the args onto the env contract the docker layer
# already reads — BENCHMARK / AGENT_ENGINE / AGENT_MODEL — and sources config.env.
#
#   <agent> = oracle  ->  AGENT_ENGINE=placeholder (no-op "train" = base model as
#             final_model) run through the SAME verifier + anti-cheat gate, then
#             assert the reward reproduces the suite baseline. This validates the
#             whole build->generate->grade->gate chain WITHOUT spending model budget
#             (the equivalent of harbor's `-a oracle` / a task's solution/solve.sh).
#
# Backend: `docker` is the supported backend in this distribution.
# It requires a GPU host with a Docker daemon.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage(){ echo "usage: run_task.sh <bench> <agent|oracle> [model]"; echo "  benches: $VALID"; exit 2; }
VALID="mmau mmar mmmu_pro video_mmmu videomme_v2 jointavbench omnivideobench mmswe"

BENCH="${1:-}"; AGENT="${2:-}"; MODEL="${3:-}"
[ -n "$BENCH" ] && [ -n "$AGENT" ] || usage
grep -qw "$BENCH" <<<"$VALID" || { echo "unknown bench: $BENCH"; usage; }

DOCKER_DIR="$HERE/src/docker"
[ -f "$DOCKER_DIR/config.env" ] || { echo "FATAL: missing $DOCKER_DIR/config.env (copy from config.env.example and fill in)"; exit 3; }

# Preflight: refuse to enter the (destructive, FRESH=1) loop on a half-filled config.
# The loop wipes WORKSPACE_HOST/LOGS_HOST and mounts MODEL_DIR/OMNI_PY; if any are
# unset or still the config.env.example placeholders, stop with a clear message
# instead of a confusing run_loop failure. (guarded subshell so it can't leak vars)
( set +e; source "$DOCKER_DIR/config.env" 2>/dev/null
  bad=""
  for v in WORKSPACE_HOST LOGS_HOST MODEL_DIR OMNI_PY; do
    val="$(eval "printf '%s' \"\${$v:-}\"")"
    { [ -z "$val" ] || case "$val" in */path/to/*|/path/to/*) true;; *) false;; esac; } && bad="$bad $v"
  done
  [ -z "$bad" ] || { echo "FATAL: config.env not fully filled — unset/placeholder:$bad" >&2; echo "  edit $DOCKER_DIR/config.env before running (see config.env.example)." >&2; exit 5; }
) || exit 5

# oracle => placeholder engine + post-run baseline assertion
ORACLE=0
if [ "$AGENT" = "oracle" ]; then ORACLE=1; export AGENT_ENGINE="placeholder"; else export AGENT_ENGINE="$AGENT"; fi
export BENCHMARK="$BENCH"
# run_verifier.sh selects the mounted task bundle via BENCH.
export BENCH
export AGENT_MODEL="$MODEL"

BACKEND="${BACKEND:-docker}"
case "$BACKEND" in
  docker)
    echo "=== run_task: bench=$BENCH agent=$AGENT_ENGINE model=${MODEL:-<cli-default>} backend=docker ==="
    bash "$DOCKER_DIR/run_loop.sh"
    ;;
  dlc)
    echo "FATAL: BACKEND=dlc is not implemented in this distribution; use BACKEND=docker." >&2
    exit 4
    ;;
  *)
    echo "unknown BACKEND: $BACKEND (docker|dlc)" >&2; exit 2 ;;
esac

# ── oracle self-test: the chain must reproduce the pinned baseline (no model) ──
if [ "$ORACLE" = "1" ]; then
  # shellcheck disable=SC1090
  source "$DOCKER_DIR/config.env"
  # Use raw metrics: a gate reset must not make a broken oracle appear to pass.
  python3 "$HERE/src/eval/baseline_util.py" --bench "$BENCH" \
    --metrics "$LOGS_HOST/verifier/metrics.json" \
    --tolerance "${ORACLE_TOL:-0.00005}"
fi
