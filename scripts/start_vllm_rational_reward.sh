#!/usr/bin/env bash
# Start vLLM OpenAI-compatible server for Rational Rewards judge weights on Hugging Face:
#   - T2I weights: TIGER-Lab/RationalRewards-8B-T2I  → served OpenAI model id: RationalRewards-8B-T2I
#   - Edit weights: TIGER-Lab/RationalRewards-8B-Edit → served OpenAI model id: RationalRewards-8B-Edit
# Training YAML: api_base_url=http://<host>:<port>/v1 and vlm_model must equal --served-model-name.
#
# Usage (2 GPUs; data-parallel-size defaults to len(CUDA_VISIBLE_DEVICES)):
#   export CUDA_VISIBLE_DEVICES=0,1
#   export MODEL_PATH="TIGER-Lab/RationalRewards-8B-T2I"
#   ./scripts/start_vllm_rational_reward.sh --max-model-len 8192
#
# Edit judge:
#   export MODEL_PATH="TIGER-Lab/RationalRewards-8B-Edit"
#   ./scripts/start_vllm_rational_reward.sh --max-model-len 8192
#
# Runtime settings are loaded from .env. Leave the served-model name empty to
# infer it from RATIONAL_REWARD_MODEL_PATH; leave data parallelism empty to
# infer it from CUDA_VISIBLE_DEVICES.
#   Any extra arguments are forwarded to `vllm serve` (e.g. --max-model-len 8192).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/lib/load_env.sh"
flowfactory_load_env "${FLOWFACTORY_ENV_FILE:-${REPO_ROOT}/.env}"
flowfactory_require_env VLLM_BIN VLLM_HOST VLLM_PORT \
  VLLM_GPU_MEMORY_UTILIZATION RATIONAL_REWARD_MODEL_PATH \
  RATIONAL_REWARD_TENSOR_PARALLEL_SIZE

SERVED_MODEL_NAME="${RATIONAL_REWARD_SERVED_MODEL_NAME}"
if [[ -n "${SERVED_MODEL_NAME}" ]]; then
  :
elif [[ "${RATIONAL_REWARD_MODEL_PATH}" == *"RationalRewards-8B-Edit"* ]]; then
  SERVED_MODEL_NAME="RationalRewards-8B-Edit"
else
  SERVED_MODEL_NAME="RationalRewards-8B-T2I"
fi

DATA_PARALLEL_SIZE="${RATIONAL_REWARD_DATA_PARALLEL_SIZE}"
if [[ -z "${DATA_PARALLEL_SIZE}" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      DATA_PARALLEL_SIZE="$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')"
    else
      DATA_PARALLEL_SIZE=1
    fi
fi

exec "${VLLM_BIN}" serve "${RATIONAL_REWARD_MODEL_PATH}" \
  --tensor-parallel-size "${RATIONAL_REWARD_TENSOR_PARALLEL_SIZE}" \
  --data-parallel-size "${DATA_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  "$@"
