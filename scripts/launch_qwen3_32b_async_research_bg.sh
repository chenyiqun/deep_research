#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/tidal-alsh01/usr/chenyiqun/research_project/Deep_Research/deep_research}"
OUT_DIR="${OUT_DIR:-${REPO_DIR}/outputs/qwen3_32b_async_research}"
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
  VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}" \
  VLLM_MODEL="${VLLM_MODEL:-qwen3-32b}" \
  VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}" \
  WEB_SEARCH_API_KEY="${WEB_SEARCH_API_KEY:-}" \
  WEB_SEARCH_ENDPOINT="${WEB_SEARCH_ENDPOINT:-http://edithai.devops.xiaohongshu.com/ext-tools/zhipu-web-search-vip}" \
  MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-4}" \
  MAX_CONCURRENT_LLM_CALLS="${MAX_CONCURRENT_LLM_CALLS:-16}" \
  MAX_CONCURRENT_SEARCHES="${MAX_CONCURRENT_SEARCHES:-8}" \
  MAX_CONCURRENT_READERS="${MAX_CONCURRENT_READERS:-12}" \
  MAX_ROUNDS="${MAX_ROUNDS:-3}" \
  MAX_SEARCH_QUERIES_PER_ROUND="${MAX_SEARCH_QUERIES_PER_ROUND:-3}" \
  SEARCH_TOP_K="${SEARCH_TOP_K:-5}" \
  SEARCH_COUNT="${SEARCH_COUNT:-15}" \
  REPORT_MAX_TOKENS="${REPORT_MAX_TOKENS:-8192}" \
  JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-8192}" \
  VLLM_WAIT_RETRIES="${VLLM_WAIT_RETRIES:-120}" \
  VLLM_WAIT_SLEEP="${VLLM_WAIT_SLEEP:-5}" \
  bash "${REPO_DIR}/scripts/run_qwen3_32b_async_research.sh" \
  >/dev/null 2>&1 &

PID="$!"
echo "${PID}" > "${PID_FILE}"

echo "Started async deep research background run."
echo "PID: ${PID}"
echo "PID file: ${PID_FILE}"
echo "Log file: ${LOG_FILE}"
echo
echo "Follow log:"
echo "tail -f ${LOG_FILE}"
echo
echo "Check process:"
echo "ps -p ${PID} -f"
