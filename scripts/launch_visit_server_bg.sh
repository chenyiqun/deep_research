#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/tidal-alsh01/usr/chenyiqun/research_project/Deep_Research/deep_research}"
VISIT_LOG_DIR="${VISIT_LOG_DIR:-${REPO_DIR}/outputs/visit_server_logs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
VISIT_LOG_FILE="${VISIT_LOG_FILE:-${VISIT_LOG_DIR}/visit_${RUN_ID}.log}"

mkdir -p "${VISIT_LOG_DIR}"
cd "${REPO_DIR}"

nohup bash scripts/start_visit_server.sh > "${VISIT_LOG_FILE}" 2>&1 &
PID=$!

echo "Started visit server in background."
echo "PID: ${PID}"
echo "Log: ${VISIT_LOG_FILE}"
echo "Health check: curl http://127.0.0.1:${VISIT_PORT:-8765}/health"
