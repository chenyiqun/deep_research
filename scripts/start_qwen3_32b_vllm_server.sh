#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/mnt/tidal-alsh01/usr/chenyiqun/research_project/Deep_Research/deep_research}"
MODEL_PATH="${MODEL_PATH:-/mnt/tidal-alsh01/usr/chenyiqun/base_models/Qwen/Qwen3-32B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-32b}"
GPU_DEVICES="${GPU_DEVICES:-0,1,2,3,4,5,6,7}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"
SCHEDULING_POLICY="${SCHEDULING_POLICY:-priority}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
OUT_DIR="${OUT_DIR:-${REPO_DIR}/outputs/vllm_qwen3_32b_server}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/server_${RUN_ID}.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/server_${RUN_ID}.pid}"

mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"

IFS=',' read -r -a GPU_DEVICE_LIST <<< "${GPU_DEVICES}"
if (( ${#GPU_DEVICE_LIST[@]} < TENSOR_PARALLEL_SIZE )); then
  echo "ERROR: TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE} needs at least that many visible GPUs, but GPU_DEVICES=${GPU_DEVICES} only has ${#GPU_DEVICE_LIST[@]} entries."
  exit 2
fi

echo "Starting vLLM server in background."
echo "Model path: ${MODEL_PATH}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "GPU devices: ${GPU_DEVICES}"
echo "Tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
echo "Base URL: http://${VLLM_HOST}:${VLLM_PORT}/v1"
echo "Prefix caching: ${ENABLE_PREFIX_CACHING}"
echo "Chunked prefill: ${ENABLE_CHUNKED_PREFILL}"
echo "Scheduling policy: ${SCHEDULING_POLICY}"
echo "Log file: ${LOG_FILE}"

VLLM_AGENT_ARGS=(--scheduling-policy "${SCHEDULING_POLICY}")
case "${ENABLE_PREFIX_CACHING}" in
  1|true|True|TRUE|yes|Yes|YES)
    VLLM_AGENT_ARGS+=(--enable-prefix-caching)
    ;;
esac
case "${ENABLE_CHUNKED_PREFILL}" in
  1|true|True|TRUE|yes|Yes|YES)
    VLLM_AGENT_ARGS+=(--enable-chunked-prefill)
    ;;
esac

nohup env CUDA_VISIBLE_DEVICES="${GPU_DEVICES}" \
  vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --trust-remote-code \
    "${VLLM_AGENT_ARGS[@]}" \
  >"${LOG_FILE}" 2>&1 &

PID="$!"
echo "${PID}" > "${PID_FILE}"

echo "Started vLLM server."
echo "PID: ${PID}"
echo "PID file: ${PID_FILE}"
echo "Log file: ${LOG_FILE}"
echo
echo "Follow log:"
echo "tail -f ${LOG_FILE}"
echo
echo "Health check after it finishes loading:"
echo "curl http://127.0.0.1:${VLLM_PORT}/v1/models"
