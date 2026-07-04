#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/tidal-alsh01/usr/chenyiqun/research_project/Deep_Research/deep_research}"
VISIT_HOST="${VISIT_HOST:-0.0.0.0}"
VISIT_PORT="${VISIT_PORT:-8765}"
VISIT_CACHE_DIR="${VISIT_CACHE_DIR:-${REPO_DIR}/outputs/url_visit_cache}"
VISIT_FETCH_TIMEOUT_S="${VISIT_FETCH_TIMEOUT_S:-45}"
VISIT_MAX_CONCURRENT_FETCHES="${VISIT_MAX_CONCURRENT_FETCHES:-32}"
VISIT_FETCH_MAX_RETRIES="${VISIT_FETCH_MAX_RETRIES:-2}"
VISIT_FETCH_MAX_BYTES="${VISIT_FETCH_MAX_BYTES:-4000000}"
VISIT_FETCH_MAX_EXTRACTED_CHARS="${VISIT_FETCH_MAX_EXTRACTED_CHARS:-80000}"
VISIT_CONTENT_LENGTH="${VISIT_CONTENT_LENGTH:-12000}"
VISIT_MIN_CONTENT_CHARS="${VISIT_MIN_CONTENT_CHARS:-500}"
VISIT_ENABLE_CRAWL4AI="${VISIT_ENABLE_CRAWL4AI:-0}"
VISIT_CRAWL4AI_TIMEOUT_S="${VISIT_CRAWL4AI_TIMEOUT_S:-45}"
VISIT_HTML_FETCH_MODE="${VISIT_HTML_FETCH_MODE:-crawl4ai_first}"
VISIT_HTML_DIRECT_FALLBACK="${VISIT_HTML_DIRECT_FALLBACK:-0}"
VISIT_CRAWL4AI_WAIT_UNTIL="${VISIT_CRAWL4AI_WAIT_UNTIL:-domcontentloaded}"
VISIT_CRAWL4AI_MAX_RETRIES="${VISIT_CRAWL4AI_MAX_RETRIES:-2}"
VISIT_CACHE_ERRORS="${VISIT_CACHE_ERRORS:-0}"
VISIT_SUMMARY_PROVIDER="${VISIT_SUMMARY_PROVIDER:-local_vllm}"
VISIT_SUMMARY_BASE_URL="${VISIT_SUMMARY_BASE_URL:-http://127.0.0.1:8000/v1}"
VISIT_SUMMARY_MODEL="${VISIT_SUMMARY_MODEL:-qwen3-32b}"
VISIT_SUMMARY_API_KEY="${VISIT_SUMMARY_API_KEY:-EMPTY}"
VISIT_SUMMARY_TIMEOUT_S="${VISIT_SUMMARY_TIMEOUT_S:-300}"
VISIT_SUMMARY_MAX_CONCURRENT_REQUESTS="${VISIT_SUMMARY_MAX_CONCURRENT_REQUESTS:-4}"
VISIT_SUMMARY_INPUT_MAX_CHARS="${VISIT_SUMMARY_INPUT_MAX_CHARS:-60000}"
VISIT_SUMMARY_CHUNK_CHARS="${VISIT_SUMMARY_CHUNK_CHARS:-20000}"
VISIT_SUMMARY_MAX_TOKENS="${VISIT_SUMMARY_MAX_TOKENS:-2048}"
VISIT_SUMMARY_MERGE_MAX_TOKENS="${VISIT_SUMMARY_MERGE_MAX_TOKENS:-3072}"

cd "${REPO_DIR}"
mkdir -p "${VISIT_CACHE_DIR}"

ARGS=()
case "${VISIT_ENABLE_CRAWL4AI}" in
  1|true|True|TRUE|yes|Yes|YES)
    ARGS+=(--enable-crawl4ai)
    ;;
esac
case "${VISIT_CACHE_ERRORS}" in
  1|true|True|TRUE|yes|Yes|YES)
    ARGS+=(--cache-errors)
    ;;
esac
case "${VISIT_HTML_DIRECT_FALLBACK}" in
  1|true|True|TRUE|yes|Yes|YES)
    ARGS+=(--html-direct-fallback)
    ;;
esac

echo "Starting DeepResearch visit server"
echo "Repo: ${REPO_DIR}"
echo "Bind: ${VISIT_HOST}:${VISIT_PORT}"
echo "Cache dir: ${VISIT_CACHE_DIR}"
echo "crawl4ai enabled: ${VISIT_ENABLE_CRAWL4AI}"
echo "html fetch mode: ${VISIT_HTML_FETCH_MODE}"
echo "html direct fallback: ${VISIT_HTML_DIRECT_FALLBACK}"
echo "crawl4ai wait until: ${VISIT_CRAWL4AI_WAIT_UNTIL}"
echo "crawl4ai max retries: ${VISIT_CRAWL4AI_MAX_RETRIES}"
echo "summary provider: ${VISIT_SUMMARY_PROVIDER}"
echo "summary base URL: ${VISIT_SUMMARY_BASE_URL}"
echo "summary model: ${VISIT_SUMMARY_MODEL}"
echo "summary max concurrent requests: ${VISIT_SUMMARY_MAX_CONCURRENT_REQUESTS}"

PYTHONPATH="${REPO_DIR}" python -m drb_qwen.visit_server \
  --host "${VISIT_HOST}" \
  --port "${VISIT_PORT}" \
  --fetch-timeout-s "${VISIT_FETCH_TIMEOUT_S}" \
  --max-concurrent-fetches "${VISIT_MAX_CONCURRENT_FETCHES}" \
  --fetch-max-retries "${VISIT_FETCH_MAX_RETRIES}" \
  --fetch-max-bytes "${VISIT_FETCH_MAX_BYTES}" \
  --fetch-max-extracted-chars "${VISIT_FETCH_MAX_EXTRACTED_CHARS}" \
  --cache-dir "${VISIT_CACHE_DIR}" \
  --content-length "${VISIT_CONTENT_LENGTH}" \
  --min-content-chars "${VISIT_MIN_CONTENT_CHARS}" \
  --crawl4ai-timeout-s "${VISIT_CRAWL4AI_TIMEOUT_S}" \
  --html-fetch-mode "${VISIT_HTML_FETCH_MODE}" \
  --crawl4ai-wait-until "${VISIT_CRAWL4AI_WAIT_UNTIL}" \
  --crawl4ai-max-retries "${VISIT_CRAWL4AI_MAX_RETRIES}" \
  --summary-provider "${VISIT_SUMMARY_PROVIDER}" \
  --summary-base-url "${VISIT_SUMMARY_BASE_URL}" \
  --summary-model "${VISIT_SUMMARY_MODEL}" \
  --summary-api-key "${VISIT_SUMMARY_API_KEY}" \
  --summary-timeout-s "${VISIT_SUMMARY_TIMEOUT_S}" \
  --summary-max-concurrent-requests "${VISIT_SUMMARY_MAX_CONCURRENT_REQUESTS}" \
  --summary-input-max-chars "${VISIT_SUMMARY_INPUT_MAX_CHARS}" \
  --summary-chunk-chars "${VISIT_SUMMARY_CHUNK_CHARS}" \
  --summary-max-tokens "${VISIT_SUMMARY_MAX_TOKENS}" \
  --summary-merge-max-tokens "${VISIT_SUMMARY_MERGE_MAX_TOKENS}" \
  "${ARGS[@]}"
