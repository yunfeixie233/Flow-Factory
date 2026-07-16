#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server for the Qwen-Image-Bench judge ("Q-Judger").
#   - Weights:        Qwen/Qwen-Image-Bench  (a fine-tuned ~27B Qwen3-VL)
#   - Served model id: Qwen-Image-Bench       (must equal the YAML `vlm_model` key)
# Training YAML: set api_base_url=http://<host>:<port>/v1 and vlm_model=<--served-model-name>.
#
# Usage (multi-GPU; tensor-parallel-size defaults to len(CUDA_VISIBLE_DEVICES)):
#   export CUDA_VISIBLE_DEVICES=0,1,2,3
#   ./scripts/start_vllm_qwen_image_bench.sh --max-model-len 32768
#
# The judge emits a long <think> ... </think> section before the JSON scores, so
# keep --max-model-len well above (image tokens + max_tokens). The reward defaults
# to max_tokens=4096; 32768 model length is a safe starting point.
#
# Runtime settings are loaded from .env. QWEN_IMAGE_BENCH_TENSOR_PARALLEL_SIZE
# may be left empty to infer it from CUDA_VISIBLE_DEVICES.
#   Any extra arguments are forwarded to `vllm serve` (e.g. --max-model-len 32768).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/lib/load_env.sh"
flowfactory_load_env "${FLOWFACTORY_ENV_FILE:-${REPO_ROOT}/.env}"
flowfactory_require_env VLLM_BIN VLLM_HOST VLLM_PORT \
  VLLM_GPU_MEMORY_UTILIZATION QWEN_IMAGE_BENCH_MODEL_PATH \
  QWEN_IMAGE_BENCH_SERVED_MODEL_NAME

TENSOR_PARALLEL_SIZE="${QWEN_IMAGE_BENCH_TENSOR_PARALLEL_SIZE}"
if [[ -z "${TENSOR_PARALLEL_SIZE}" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      TENSOR_PARALLEL_SIZE="$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')"
    else
      TENSOR_PARALLEL_SIZE=1
    fi
fi

exec "${VLLM_BIN}" serve "${QWEN_IMAGE_BENCH_MODEL_PATH}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --served-model-name "${QWEN_IMAGE_BENCH_SERVED_MODEL_NAME}" \
  --limit-mm-per-prompt '{"image": 1}' \
  --trust-remote-code \
  "$@"
