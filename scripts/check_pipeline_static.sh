#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
cd "${REPO_DIR}"

echo "[1/9] Checking shell scripts"
bash -n scripts/run_qwen3_8b_smoke.sh
bash -n scripts/launch_qwen3_8b_smoke_bg.sh
bash -n scripts/start_qwen3_32b_vllm_server.sh
bash -n scripts/run_qwen3_32b_async_research.sh
bash -n scripts/launch_qwen3_32b_async_research_bg.sh

echo "[2/9] Compiling Python modules"
PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/drb_qwen_pycache}" \
  "${PYTHON_BIN}" -m py_compile \
    drb_qwen/*.py \
    drb_qwen/multi_agent/*.py \
    scripts/download_drb_data.py \
    scripts/test_all_search_engines.py \
    scripts/test_web_search.py \
    scripts/test_search_url_fetch.py \
    scripts/compare_race_runs.py \
    tests/smoke_test_scoring.py \
    tests/smoke_test_url_fetcher.py \
    tests/smoke_test_async_workflow.py \
    tests/smoke_test_multi_agent_core.py \
    tests/smoke_test_dynamic_replan.py \
    tests/smoke_test_agent_inference.py

echo "[3/9] Checking CLI entrypoints"
PYTHONPATH="${REPO_DIR}" "${PYTHON_BIN}" -m drb_qwen.generate_reports --help >/dev/null
PYTHONPATH="${REPO_DIR}" "${PYTHON_BIN}" -m drb_qwen.evaluate_race --help >/dev/null
PYTHONPATH="${REPO_DIR}" "${PYTHON_BIN}" -m drb_qwen.generate_reports_async_research --help >/dev/null
PYTHONPATH="${REPO_DIR}" "${PYTHON_BIN}" -m drb_qwen.run_multi_agent_research --help >/dev/null
PYTHONPATH="${REPO_DIR}" "${PYTHON_BIN}" -m drb_qwen.evaluate_race_async --help >/dev/null
PYTHONPATH="${REPO_DIR}" "${PYTHON_BIN}" scripts/test_all_search_engines.py --help >/dev/null
PYTHONPATH="${REPO_DIR}" "${PYTHON_BIN}" scripts/test_web_search.py --help >/dev/null
PYTHONPATH="${REPO_DIR}" "${PYTHON_BIN}" scripts/test_search_url_fetch.py --help >/dev/null
PYTHONPATH="${REPO_DIR}" "${PYTHON_BIN}" scripts/compare_race_runs.py --help >/dev/null

echo "[4/9] Running scoring smoke test"
PYTHONPATH="${REPO_DIR}" "${PYTHON_BIN}" tests/smoke_test_scoring.py

echo "[5/9] Running URL fetcher smoke test"
PYTHONPATH="${REPO_DIR}" "${PYTHON_BIN}" tests/smoke_test_url_fetcher.py

echo "[6/9] Running async workflow smoke test"
PYTHONPATH="${REPO_DIR}" "${PYTHON_BIN}" tests/smoke_test_async_workflow.py

echo "[7/9] Running multi-agent core smoke test"
PYTHONPATH="${REPO_DIR}" "${PYTHON_BIN}" tests/smoke_test_multi_agent_core.py

echo "[8/9] Running dynamic replan and audit-repair smoke test"
PYTHONPATH="${REPO_DIR}" "${PYTHON_BIN}" tests/smoke_test_dynamic_replan.py

echo "[9/9] Running Agent inference gateway smoke test"
PYTHONPATH="${REPO_DIR}" "${PYTHON_BIN}" tests/smoke_test_agent_inference.py

echo "All static pipeline checks passed."
