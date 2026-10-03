#!/bin/bash
# Runs within the isolated instance. Changes no global host trust store.
set -euo pipefail
root=/opt/mmptb-offline
ready=$(mktemp /tmp/mmptb-offline-ready.XXXXXX)
rm -f "$ready"
node "$root/offline_server.js" "$root" 443 "$ready" > /tmp/offline_resources.log 2>&1 &
server_pid=$!
cleanup() { kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; rm -f "$ready"; }
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
for ((attempt=0; attempt<100; attempt++)); do
  [ -s "$ready" ] && break
  kill -0 "$server_pid" 2>/dev/null || { echo MMPTB_OFFLINE_START_FAILED >> /tmp/preflight.log; exit 89; }
  sleep 0.1
done
[ -s "$ready" ] || { echo MMPTB_OFFLINE_START_TIMEOUT >> /tmp/preflight.log; exit 89; }
echo MMPTB_OFFLINE_ACTIVE >> /tmp/preflight.log
export NODE_EXTRA_CA_CERTS="$root/ca.pem"
"$@"
