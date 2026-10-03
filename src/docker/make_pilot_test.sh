#!/bin/bash
# Produce a limit-parameterized copy of the baked verifier test.sh so the pilot
# can run a small MMAU sample count (loop-plumbing proof) instead of the full
# 1000. The only change is the eval limit line:
#   --limit -1   ->   --limit "${VERIFIER_LIMIT:--1}"
# so the copy defaults to full (-1) exactly like the baked one, but honors
# VERIFIER_LIMIT when set. Mount it over /tests/test.sh via run_verifier.sh's
# VERIFIER_TEST_SH knob.
#
# Usage (from src/docker):
#   bash make_pilot_test.sh
#   VERIFIER_TEST_SH=$MMPTB_ROOT/test_pilot.sh VERIFIER_LIMIT=8 bash run_verifier.sh
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.env"

SRC="$REPO_ROOT/src/harbor_adapter/template/tests/test.sh"
DST="${1:-$MMPTB_ROOT/test_pilot.sh}"
sed 's#--limit -1 \\#--limit "${VERIFIER_LIMIT:--1}" \\#' "$SRC" > "$DST"
chmod +x "$DST"
echo "wrote $DST"
grep -n "VERIFIER_LIMIT" "$DST" || { echo "ERROR: limit line not parameterized"; exit 1; }
