#!/usr/bin/env bash
# Clean up stale sentinel files and shared memory left by previous GenieSim
# sessions.  Run this before starting a new collect / train session to avoid
# the new sim manager picking up outdated state.
#
# Usage:
#   bash rlinf/envs/geniesim/scripts/cleanup_stale.sh [GENIESIM_ROOT]

set -eo pipefail

GENIESIM_ROOT="${1:-${GENIESIM_ROOT:-/geniesim/main}}"

echo "[cleanup] Removing stale sentinel files under ${GENIESIM_ROOT} ..."
rm -f "${GENIESIM_ROOT}/.geniesim_ready" \
      "${GENIESIM_ROOT}/.geniesim_progress" \
      "${GENIESIM_ROOT}/.geniesim_idle" \
      "${GENIESIM_ROOT}/.geniesim_start" \
      "${GENIESIM_ROOT}/.geniesim_stop" \
      "${GENIESIM_ROOT}/.geniesim_error" \
      "${GENIESIM_ROOT}/.sim_server_config.json" 2>/dev/null || true

echo "[cleanup] Removing stale shared memory segments ..."
rm -f /dev/shm/geniesim* 2>/dev/null || true

echo "[cleanup] Done."
