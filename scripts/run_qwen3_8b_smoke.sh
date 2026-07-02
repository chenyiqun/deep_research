#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/tidal-alsh01/usr/chenyiqun/research_project/Deep_Research/deep_research}"
DATA_DIR="${DATA_DIR:-/mnt/tidal-alsh01/usr/chenyiqun/datasets/DeepResearch/deep_research_bench_data}"
MODEL_PATH="${MODEL_PATH:-/mnt/tidal-alsh01/usr/chenyiqun/base_models/Qwen/Qwen3-8B}"
OUT_DIR="${OUT_DIR:-${REPO_DIR}/outputs/qwen3_8b_smoke}"
LIMIT="${LIMIT:-2}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/run_${RUN_ID}.log}"

cd "${REPO_DIR}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Run ID: ${RUN_ID}"
echo "Log: ${LOG_FILE}"
echo "Started at: $(date -Iseconds)"

echo "Repo: ${REPO_DIR}"
echo "Data: ${DATA_DIR}"
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUT_DIR}"
echo "Limit: ${LIMIT}"

PYTHONPATH="${REPO_DIR}" python -m drb_qwen.generate_reports \
  --query-file "${DATA_DIR}/query.jsonl" \
  --output-file "${OUT_DIR}/qwen3_8b_reports.jsonl" \
  --model "${MODEL_PATH}" \
  --limit "${LIMIT}" \
  --batch-size 1 \
  --max-model-len 32768 \
  --max-tokens 8192 \
  --temperature 0.2 \
  --top-p 0.95 \
  --gpu-memory-utilization 0.90 \
  --resume

PYTHONPATH="${REPO_DIR}" python -m drb_qwen.evaluate_race \
  --query-file "${DATA_DIR}/query.jsonl" \
  --criteria-file "${DATA_DIR}/criteria.jsonl" \
  --target-file "${OUT_DIR}/qwen3_8b_reports.jsonl" \
  --reference-file "${DATA_DIR}/reference.jsonl" \
  --output-file "${OUT_DIR}/race_raw_results.jsonl" \
  --summary-file "${OUT_DIR}/race_summary.json" \
  --judge-model "${MODEL_PATH}" \
  --limit "${LIMIT}" \
  --batch-size 1 \
  --max-model-len 32768 \
  --max-tokens 8192 \
  --temperature 0.0 \
  --top-p 1.0 \
  --gpu-memory-utilization 0.90 \
  --resume \
  --save-judge-output

echo "Done. Summary:"
cat "${OUT_DIR}/race_summary.json"
echo
echo "Finished at: $(date -Iseconds)"
echo "Log saved to: ${LOG_FILE}"
