#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
cd "${REPO_DIR}"

echo "[1/6] Checking shell scripts"
bash -n scripts/run_qwen3_8b_smoke.sh
bash -n scripts/launch_qwen3_8b_smoke_bg.sh
bash -n scripts/start_qwen3_32b_vllm_server.sh
bash -n scripts/run_qwen3_32b_async_research.sh
bash -n scripts/launch_qwen3_32b_async_research_bg.sh

echo "[2/6] Compiling Python modules"
PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/drb_qwen_pycache}" \
  "${PYTHON_BIN}" -m py_compile \
    drb_qwen/*.py \
    scripts/download_drb_data.py \
    scripts/test_search_url_fetch.py \
    tests/smoke_test_scoring.py \
    tests/smoke_test_url_fetcher.py \
    tests/smoke_test_async_workflow.py

echo "[3/6] Checking CLI entrypoints"
PYTHONPATH="${REPO_DIR}" "${PYTHON_BIN}" -m drb_qwen.generate_reports --help >/dev/null
PYTHONPATH="${REPO_DIR}" "${PYTHON_BIN}" -m drb_qwen.evaluate_race --help >/dev/null
PYTHONPATH="${REPO_DIR}" "${PYTHON_BIN}" -m drb_qwen.generate_reports_async_research --help >/dev/null
PYTHONPATH="${REPO_DIR}" "${PYTHON_BIN}" -m drb_qwen.evaluate_race_async --help >/dev/null
PYTHONPATH="${REPO_DIR}" "${PYTHON_BIN}" scripts/test_search_url_fetch.py --help >/dev/null

echo "[4/6] Running scoring smoke test"
PYTHONPATH="${REPO_DIR}" "${PYTHON_BIN}" tests/smoke_test_scoring.py

echo "[5/6] Running URL fetcher smoke test"
PYTHONPATH="${REPO_DIR}" "${PYTHON_BIN}" tests/smoke_test_url_fetcher.py

echo "[6/6] Running async workflow smoke test"
PYTHONPATH="${REPO_DIR}" "${PYTHON_BIN}" tests/smoke_test_async_workflow.py

echo "All static pipeline checks passed."
