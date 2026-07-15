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
SEARCH_ENGINE="${SEARCH_ENGINE:-search_live}"
MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-4}"
MAX_CONCURRENT_LLM_CALLS="${MAX_CONCURRENT_LLM_CALLS:-16}"
MAX_CONCURRENT_CONTROL_CALLS="${MAX_CONCURRENT_CONTROL_CALLS:-8}"
MAX_CONCURRENT_LONG_CALLS="${MAX_CONCURRENT_LONG_CALLS:-2}"
MAX_CONCURRENT_LLM_CALLS_PER_RUN="${MAX_CONCURRENT_LLM_CALLS_PER_RUN:-12}"
MAX_INFLIGHT_LLM_TOKENS="${MAX_INFLIGHT_LLM_TOKENS:-262144}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
CONTEXT_SAFETY_TOKENS="${CONTEXT_SAFETY_TOKENS:-512}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
FORWARD_VLLM_PRIORITY="${FORWARD_VLLM_PRIORITY:-1}"
STRUCTURED_OUTPUTS_ENABLED="${STRUCTURED_OUTPUTS_ENABLED:-1}"
MAX_CONCURRENT_SEARCHES="${MAX_CONCURRENT_SEARCHES:-8}"
MAX_CONCURRENT_READERS="${MAX_CONCURRENT_READERS:-12}"
URL_FETCH_MODE="${URL_FETCH_MODE:-auto}"
URL_FETCH_ENABLED="${URL_FETCH_ENABLED:-}"
URL_VISIT_ENDPOINT="${URL_VISIT_ENDPOINT:-}"
URL_VISIT_TIMEOUT_S="${URL_VISIT_TIMEOUT_S:-60}"
URL_VISIT_FALLBACK_ENABLED="${URL_VISIT_FALLBACK_ENABLED:-1}"
MAX_CONCURRENT_URL_FETCHES="${MAX_CONCURRENT_URL_FETCHES:-16}"
URL_FETCH_TIMEOUT_S="${URL_FETCH_TIMEOUT_S:-30}"
URL_FETCH_MAX_RETRIES="${URL_FETCH_MAX_RETRIES:-2}"
URL_FETCH_MAX_BYTES="${URL_FETCH_MAX_BYTES:-2000000}"
URL_FETCH_MAX_EXTRACTED_CHARS="${URL_FETCH_MAX_EXTRACTED_CHARS:-50000}"
URL_FETCH_CACHE_DIR="${URL_FETCH_CACHE_DIR:-${OUT_DIR}/url_cache}"
URL_FETCH_CACHE_ERRORS="${URL_FETCH_CACHE_ERRORS:-0}"
MIN_FETCHED_CONTENT_CHARS="${MIN_FETCHED_CONTENT_CHARS:-500}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"
MAX_SEARCH_QUERIES_PER_ROUND="${MAX_SEARCH_QUERIES_PER_ROUND:-3}"
MAX_INITIAL_TASKS="${MAX_INITIAL_TASKS:-4}"
MAX_RESEARCHERS="${MAX_RESEARCHERS:-4}"
MAX_SUBTASKS="${MAX_SUBTASKS:-16}"
MAX_NEW_TASKS_PER_ROUND="${MAX_NEW_TASKS_PER_ROUND:-3}"
MAX_REACT_STEPS="${MAX_REACT_STEPS:-3}"
MAX_TOOL_CALLS_PER_SUBTASK="${MAX_TOOL_CALLS_PER_SUBTASK:-18}"
MAX_TOTAL_TOOL_CALLS="${MAX_TOTAL_TOOL_CALLS:-160}"
MAX_TOTAL_SEARCHES="${MAX_TOTAL_SEARCHES:-30}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-1000000}"
MAX_RUN_SECONDS="${MAX_RUN_SECONDS:-3600}"
MIN_TOTAL_CLAIMS="${MIN_TOTAL_CLAIMS:-3}"
MIN_COVERAGE_RATIO="${MIN_COVERAGE_RATIO:-0.6}"
MAX_AUDIT_ROUNDS="${MAX_AUDIT_ROUNDS:-2}"
MAX_REPAIR_TASKS="${MAX_REPAIR_TASKS:-3}"
AUDITOR_MAX_TOKENS="${AUDITOR_MAX_TOKENS:-4096}"
RUN_STATE_DIR="${RUN_STATE_DIR:-${OUT_DIR}/run_state}"
SEARCH_TOP_K="${SEARCH_TOP_K:-5}"
SEARCH_COUNT="${SEARCH_COUNT:-15}"
REPORT_MAX_TOKENS="${REPORT_MAX_TOKENS:-8192}"
SOURCE_CONTENT_MAX_CHARS="${SOURCE_CONTENT_MAX_CHARS:-12000}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-4096}"
JUDGE_CONTEXT_RETRY_ATTEMPTS="${JUDGE_CONTEXT_RETRY_ATTEMPTS:-2}"
JUDGE_CONTEXT_SAFETY_TOKENS="${JUDGE_CONTEXT_SAFETY_TOKENS:-256}"
JUDGE_MIN_RETRY_MAX_TOKENS="${JUDGE_MIN_RETRY_MAX_TOKENS:-1024}"
VLLM_WAIT_RETRIES="${VLLM_WAIT_RETRIES:-120}"
VLLM_WAIT_SLEEP="${VLLM_WAIT_SLEEP:-5}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/run_${RUN_ID}.log}"

case "${URL_FETCH_MODE}" in
  auto|always|never)
    ;;
  *)
    echo "ERROR: URL_FETCH_MODE must be auto, always, or never; got ${URL_FETCH_MODE}"
    exit 2
    ;;
esac
if [[ -n "${URL_FETCH_ENABLED}" ]]; then
  case "${URL_FETCH_ENABLED}" in
    0|false|False|FALSE|no|No|NO)
      URL_FETCH_MODE="never"
      ;;
    1|true|True|TRUE|yes|Yes|YES)
      URL_FETCH_MODE="always"
      ;;
    *)
      echo "ERROR: legacy URL_FETCH_ENABLED must be a boolean; got ${URL_FETCH_ENABLED}"
      exit 2
      ;;
  esac
fi

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
echo "Search engine: ${SEARCH_ENGINE}"
echo "Max concurrent tasks: ${MAX_CONCURRENT_TASKS}"
echo "Max concurrent LLM calls: ${MAX_CONCURRENT_LLM_CALLS}"
echo "Max concurrent control calls: ${MAX_CONCURRENT_CONTROL_CALLS}"
echo "Max concurrent long-output calls: ${MAX_CONCURRENT_LONG_CALLS}"
echo "Max in-flight LLM tokens: ${MAX_INFLIGHT_LLM_TOKENS}"
echo "Model context length: ${MAX_MODEL_LEN}"
echo "Tokenizer path: ${TOKENIZER_PATH:-conservative estimator}"
echo "URL fetch mode: ${URL_FETCH_MODE}"
echo "URL visit endpoint: ${URL_VISIT_ENDPOINT}"
echo "Max concurrent URL fetches: ${MAX_CONCURRENT_URL_FETCHES}"
echo "URL fetch timeout seconds: ${URL_FETCH_TIMEOUT_S}"
echo "URL fetch max bytes: ${URL_FETCH_MAX_BYTES}"
echo "URL fetch max extracted chars: ${URL_FETCH_MAX_EXTRACTED_CHARS}"
echo "URL fetch cache dir: ${URL_FETCH_CACHE_DIR}"
echo "URL fetch cache errors: ${URL_FETCH_CACHE_ERRORS}"
echo "Min fetched content chars: ${MIN_FETCHED_CONTENT_CHARS}"
echo "Max concurrent readers: ${MAX_CONCURRENT_READERS}"
echo "Max concurrent searches: ${MAX_CONCURRENT_SEARCHES}"
echo "Max rounds: ${MAX_ROUNDS}"
echo "Max researchers per run: ${MAX_RESEARCHERS}"
echo "Max subtasks per run: ${MAX_SUBTASKS}"
echo "Max ReAct steps per subtask: ${MAX_REACT_STEPS}"
echo "Max total tool calls per run: ${MAX_TOTAL_TOOL_CALLS}"
echo "Max total tokens per run: ${MAX_TOTAL_TOKENS}"
echo "Run state dir: ${RUN_STATE_DIR}"
echo "Search queries per round: ${MAX_SEARCH_QUERIES_PER_ROUND}"
echo "Search top-k: ${SEARCH_TOP_K}"
echo "Search count: ${SEARCH_COUNT}"
echo "Judge max tokens: ${JUDGE_MAX_TOKENS}"
echo "Source content max chars: ${SOURCE_CONTENT_MAX_CHARS}"

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

URL_FETCH_ARGS=()
INFERENCE_ARGS=()
URL_FETCH_ARGS+=(--url-fetch-mode "${URL_FETCH_MODE}")
if [[ -n "${URL_VISIT_ENDPOINT}" ]]; then
  URL_FETCH_ARGS+=(--url-visit-endpoint "${URL_VISIT_ENDPOINT}")
  URL_FETCH_ARGS+=(--url-visit-timeout-s "${URL_VISIT_TIMEOUT_S}")
fi
case "${URL_VISIT_FALLBACK_ENABLED}" in
  0|false|False|FALSE|no|No|NO)
    URL_FETCH_ARGS+=(--disable-url-visit-fallback)
    ;;
esac
case "${URL_FETCH_CACHE_ERRORS}" in
  1|true|True|TRUE|yes|Yes|YES)
    URL_FETCH_ARGS+=(--url-fetch-cache-errors)
    ;;
esac
case "${FORWARD_VLLM_PRIORITY}" in
  1|true|True|TRUE|yes|Yes|YES)
    INFERENCE_ARGS+=(--forward-vllm-priority)
    ;;
esac
case "${STRUCTURED_OUTPUTS_ENABLED}" in
  0|false|False|FALSE|no|No|NO)
    INFERENCE_ARGS+=(--disable-structured-outputs)
    ;;
esac

PYTHONPATH="${REPO_DIR}" python -m drb_qwen.generate_reports_async_research \
  --query-file "${DATA_DIR}/query.jsonl" \
  --output-file "${REPORT_FILE}" \
  --trace-dir "${OUT_DIR}/traces" \
  --run-state-dir "${RUN_STATE_DIR}" \
  --resume-runs \
  --limit "${LIMIT}" \
  --resume \
  --llm-base-url "${VLLM_BASE_URL}" \
  --llm-model "${VLLM_MODEL}" \
  --llm-api-key "${VLLM_API_KEY}" \
  --max-concurrent-tasks "${MAX_CONCURRENT_TASKS}" \
  --max-concurrent-llm-calls "${MAX_CONCURRENT_LLM_CALLS}" \
  --max-concurrent-control-calls "${MAX_CONCURRENT_CONTROL_CALLS}" \
  --max-concurrent-long-calls "${MAX_CONCURRENT_LONG_CALLS}" \
  --max-concurrent-llm-calls-per-run "${MAX_CONCURRENT_LLM_CALLS_PER_RUN}" \
  --max-inflight-llm-tokens "${MAX_INFLIGHT_LLM_TOKENS}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --context-safety-tokens "${CONTEXT_SAFETY_TOKENS}" \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --web-search-endpoint "${WEB_SEARCH_ENDPOINT}" \
  --search-engine "${SEARCH_ENGINE}" \
  --search-count "${SEARCH_COUNT}" \
  --search-top-k "${SEARCH_TOP_K}" \
  --max-concurrent-searches "${MAX_CONCURRENT_SEARCHES}" \
  --max-concurrent-url-fetches "${MAX_CONCURRENT_URL_FETCHES}" \
  --url-fetch-timeout-s "${URL_FETCH_TIMEOUT_S}" \
  --url-fetch-max-retries "${URL_FETCH_MAX_RETRIES}" \
  --url-fetch-max-bytes "${URL_FETCH_MAX_BYTES}" \
  --url-fetch-max-extracted-chars "${URL_FETCH_MAX_EXTRACTED_CHARS}" \
  --url-fetch-cache-dir "${URL_FETCH_CACHE_DIR}" \
  --min-fetched-content-chars "${MIN_FETCHED_CONTENT_CHARS}" \
  --max-concurrent-readers "${MAX_CONCURRENT_READERS}" \
  --max-rounds "${MAX_ROUNDS}" \
  --max-search-queries-per-round "${MAX_SEARCH_QUERIES_PER_ROUND}" \
  --max-initial-tasks "${MAX_INITIAL_TASKS}" \
  --max-researchers "${MAX_RESEARCHERS}" \
  --max-subtasks "${MAX_SUBTASKS}" \
  --max-new-tasks-per-round "${MAX_NEW_TASKS_PER_ROUND}" \
  --max-react-steps "${MAX_REACT_STEPS}" \
  --max-tool-calls-per-subtask "${MAX_TOOL_CALLS_PER_SUBTASK}" \
  --max-total-tool-calls "${MAX_TOTAL_TOOL_CALLS}" \
  --max-total-searches "${MAX_TOTAL_SEARCHES}" \
  --max-total-tokens "${MAX_TOTAL_TOKENS}" \
  --max-run-seconds "${MAX_RUN_SECONDS}" \
  --min-total-claims "${MIN_TOTAL_CLAIMS}" \
  --min-coverage-ratio "${MIN_COVERAGE_RATIO}" \
  --max-audit-rounds "${MAX_AUDIT_ROUNDS}" \
  --max-repair-tasks "${MAX_REPAIR_TASKS}" \
  --auditor-max-tokens "${AUDITOR_MAX_TOKENS}" \
  --source-content-max-chars "${SOURCE_CONTENT_MAX_CHARS}" \
  --report-max-tokens "${REPORT_MAX_TOKENS}" \
  "${INFERENCE_ARGS[@]}" \
  "${URL_FETCH_ARGS[@]}"

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
  --context-retry-attempts "${JUDGE_CONTEXT_RETRY_ATTEMPTS}" \
  --context-safety-tokens "${JUDGE_CONTEXT_SAFETY_TOKENS}" \
  --min-retry-max-tokens "${JUDGE_MIN_RETRY_MAX_TOKENS}" \
  --temperature 0.0 \
  --top-p 1.0

echo "Done. Summary:"
cat "${OUT_DIR}/race_summary.json"
echo
echo "Finished at: $(date -Iseconds)"
echo "Log saved to: ${LOG_FILE}"
