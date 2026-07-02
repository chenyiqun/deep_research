#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/tidal-alsh01/usr/chenyiqun/research_project/Deep_Research/deep_research}"
DATA_DIR="${DATA_DIR:-/mnt/tidal-alsh01/usr/chenyiqun/datasets/DeepResearch/deep_research_bench_data}"
MODEL_PATH="${MODEL_PATH:-/mnt/tidal-alsh01/usr/chenyiqun/base_models/Qwen/Qwen3-8B}"
OUT_DIR="${OUT_DIR:-${REPO_DIR}/outputs/qwen3_8b_smoke}"
LIMIT="${LIMIT:-2}"
GPU_DEVICES="${GPU_DEVICES:-0,1,2,3,4,5,6,7}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
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
echo "GPU devices: ${GPU_DEVICES}"
echo "Tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
echo "Batch size: ${BATCH_SIZE}"
echo "Max model len: ${MAX_MODEL_LEN}"
echo "Max tokens: ${MAX_TOKENS}"
echo "GPU memory utilization: ${GPU_MEMORY_UTILIZATION}"

IFS=',' read -r -a GPU_DEVICE_LIST <<< "${GPU_DEVICES}"
if (( ${#GPU_DEVICE_LIST[@]} < TENSOR_PARALLEL_SIZE )); then
  echo "ERROR: TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE} needs at least that many visible GPUs, but GPU_DEVICES=${GPU_DEVICES} only has ${#GPU_DEVICE_LIST[@]} entries."
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_DEVICES}"

VLLM_ARGS=(
  --gpu-devices "${GPU_DEVICES}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
)

PYTHONPATH="${REPO_DIR}" python -m drb_qwen.generate_reports \
  --query-file "${DATA_DIR}/query.jsonl" \
  --output-file "${OUT_DIR}/qwen3_8b_reports.jsonl" \
  --model "${MODEL_PATH}" \
  --limit "${LIMIT}" \
  --batch-size "${BATCH_SIZE}" \
  --max-tokens "${MAX_TOKENS}" \
  --temperature 0.2 \
  --top-p 0.95 \
  "${VLLM_ARGS[@]}" \
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
  --batch-size "${BATCH_SIZE}" \
  --max-tokens "${MAX_TOKENS}" \
  --temperature 0.0 \
  --top-p 1.0 \
  "${VLLM_ARGS[@]}" \
  --resume \
  --save-judge-output

echo "Done. Summary:"
cat "${OUT_DIR}/race_summary.json"
echo
echo "Finished at: $(date -Iseconds)"
echo "Log saved to: ${LOG_FILE}"
