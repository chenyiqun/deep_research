#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/tidal-alsh01/usr/chenyiqun/research_project/Deep_Research/deep_research}"
OUT_DIR="${OUT_DIR:-${REPO_DIR}/outputs/qwen3_8b_smoke}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/run_${RUN_ID}.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/run_${RUN_ID}.pid}"

mkdir -p "${LOG_DIR}"

cd "${REPO_DIR}"

nohup env \
  REPO_DIR="${REPO_DIR}" \
  OUT_DIR="${OUT_DIR}" \
  LOG_DIR="${LOG_DIR}" \
  RUN_ID="${RUN_ID}" \
  LOG_FILE="${LOG_FILE}" \
  LIMIT="${LIMIT:-2}" \
  DATA_DIR="${DATA_DIR:-/mnt/tidal-alsh01/usr/chenyiqun/datasets/DeepResearch/deep_research_bench_data}" \
  MODEL_PATH="${MODEL_PATH:-/mnt/tidal-alsh01/usr/chenyiqun/base_models/Qwen/Qwen3-8B}" \
  bash "${REPO_DIR}/scripts/run_qwen3_8b_smoke.sh" \
  >/dev/null 2>&1 &

PID="$!"
echo "${PID}" > "${PID_FILE}"

echo "Started background run."
echo "PID: ${PID}"
echo "PID file: ${PID_FILE}"
echo "Log file: ${LOG_FILE}"
echo
echo "Follow log:"
echo "tail -f ${LOG_FILE}"
echo
echo "Check process:"
echo "ps -p ${PID} -f"

