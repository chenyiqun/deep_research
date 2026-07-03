#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/tidal-alsh01/usr/chenyiqun/research_project/Deep_Research/deep_research}"
DATA_DIR="${DATA_DIR:-/mnt/tidal-alsh01/usr/chenyiqun/datasets/DeepResearch/deep_research_bench_data}"
OUT_DIR="${OUT_DIR:-${REPO_DIR}/outputs/qwen3_32b_async_research}"
LIMIT="${LIMIT:-2}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
VLLM_MODEL="${VLLM_MODEL:-qwen3-32b}"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
WEB_SEARCH_ENDPOINT="${WEB_SEARCH_ENDPOINT:-http://edithai.devops.xiaohongshu.com/ext-tools/zhipu-web-search-vip}"
MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-4}"
MAX_CONCURRENT_LLM_CALLS="${MAX_CONCURRENT_LLM_CALLS:-16}"
MAX_CONCURRENT_SEARCHES="${MAX_CONCURRENT_SEARCHES:-8}"
MAX_CONCURRENT_READERS="${MAX_CONCURRENT_READERS:-12}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"
MAX_SEARCH_QUERIES_PER_ROUND="${MAX_SEARCH_QUERIES_PER_ROUND:-3}"
SEARCH_TOP_K="${SEARCH_TOP_K:-5}"
SEARCH_COUNT="${SEARCH_COUNT:-15}"
REPORT_MAX_TOKENS="${REPORT_MAX_TOKENS:-8192}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-8192}"
VLLM_WAIT_RETRIES="${VLLM_WAIT_RETRIES:-120}"
VLLM_WAIT_SLEEP="${VLLM_WAIT_SLEEP:-5}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/run_${RUN_ID}.log}"

cd "${REPO_DIR}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${OUT_DIR}/traces"

exec > >(tee -a "${LOG_FILE}") 2>&1

if [[ -z "${WEB_SEARCH_API_KEY:-}" ]]; then
  echo "ERROR: WEB_SEARCH_API_KEY is required. Export it before running this script."
  exit 2
fi

echo "Run ID: ${RUN_ID}"
echo "Log: ${LOG_FILE}"
echo "Started at: $(date -Iseconds)"
echo "Repo: ${REPO_DIR}"
echo "Data: ${DATA_DIR}"
echo "Output: ${OUT_DIR}"
echo "Limit: ${LIMIT}"
echo "vLLM base URL: ${VLLM_BASE_URL}"
echo "vLLM model: ${VLLM_MODEL}"
echo "Web search endpoint: ${WEB_SEARCH_ENDPOINT}"
echo "Max concurrent tasks: ${MAX_CONCURRENT_TASKS}"
echo "Max concurrent LLM calls: ${MAX_CONCURRENT_LLM_CALLS}"
echo "Max rounds: ${MAX_ROUNDS}"
echo "Search queries per round: ${MAX_SEARCH_QUERIES_PER_ROUND}"
echo "Search top-k: ${SEARCH_TOP_K}"

echo "Waiting for vLLM server: ${VLLM_BASE_URL}/models"
VLLM_READY=0
for ((attempt=1; attempt<=VLLM_WAIT_RETRIES; attempt++)); do
  if curl -fsS "${VLLM_BASE_URL%/}/models" >/dev/null 2>&1; then
    VLLM_READY=1
    break
  fi
  echo "vLLM is not ready yet (${attempt}/${VLLM_WAIT_RETRIES}); sleeping ${VLLM_WAIT_SLEEP}s"
  sleep "${VLLM_WAIT_SLEEP}"
done
if [[ "${VLLM_READY}" != "1" ]]; then
  echo "ERROR: vLLM server did not become ready at ${VLLM_BASE_URL}/models"
  exit 2
fi
echo "vLLM server is ready."

REPORT_FILE="${OUT_DIR}/qwen3_32b_async_research_reports.jsonl"

PYTHONPATH="${REPO_DIR}" python -m drb_qwen.generate_reports_async_research \
  --query-file "${DATA_DIR}/query.jsonl" \
  --output-file "${REPORT_FILE}" \
  --trace-dir "${OUT_DIR}/traces" \
  --limit "${LIMIT}" \
  --resume \
  --llm-base-url "${VLLM_BASE_URL}" \
  --llm-model "${VLLM_MODEL}" \
  --llm-api-key "${VLLM_API_KEY}" \
  --max-concurrent-tasks "${MAX_CONCURRENT_TASKS}" \
  --max-concurrent-llm-calls "${MAX_CONCURRENT_LLM_CALLS}" \
  --web-search-endpoint "${WEB_SEARCH_ENDPOINT}" \
  --search-count "${SEARCH_COUNT}" \
  --search-top-k "${SEARCH_TOP_K}" \
  --max-concurrent-searches "${MAX_CONCURRENT_SEARCHES}" \
  --max-concurrent-readers "${MAX_CONCURRENT_READERS}" \
  --max-rounds "${MAX_ROUNDS}" \
  --max-search-queries-per-round "${MAX_SEARCH_QUERIES_PER_ROUND}" \
  --report-max-tokens "${REPORT_MAX_TOKENS}"

PYTHONPATH="${REPO_DIR}" python -m drb_qwen.evaluate_race_async \
  --query-file "${DATA_DIR}/query.jsonl" \
  --criteria-file "${DATA_DIR}/criteria.jsonl" \
  --target-file "${REPORT_FILE}" \
  --reference-file "${DATA_DIR}/reference.jsonl" \
  --output-file "${OUT_DIR}/race_raw_results.jsonl" \
  --summary-file "${OUT_DIR}/race_summary.json" \
  --judge-model "${VLLM_MODEL}" \
  --limit "${LIMIT}" \
  --resume \
  --save-judge-output \
  --llm-base-url "${VLLM_BASE_URL}" \
  --llm-api-key "${VLLM_API_KEY}" \
  --max-concurrent-tasks "${MAX_CONCURRENT_TASKS}" \
  --max-concurrent-llm-calls "${MAX_CONCURRENT_LLM_CALLS}" \
  --max-tokens "${JUDGE_MAX_TOKENS}" \
  --temperature 0.0 \
  --top-p 1.0

echo "Done. Summary:"
cat "${OUT_DIR}/race_summary.json"
echo
echo "Finished at: $(date -Iseconds)"
echo "Log saved to: ${LOG_FILE}"
