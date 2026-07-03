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
VISIT_CACHE_ERRORS="${VISIT_CACHE_ERRORS:-0}"

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

echo "Starting DeepResearch visit server"
echo "Repo: ${REPO_DIR}"
echo "Bind: ${VISIT_HOST}:${VISIT_PORT}"
echo "Cache dir: ${VISIT_CACHE_DIR}"
echo "crawl4ai enabled: ${VISIT_ENABLE_CRAWL4AI}"

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
  "${ARGS[@]}"
