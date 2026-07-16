#!/usr/bin/env bash
# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_CHECKOUT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/lib/load_env.sh"
flowfactory_load_env "${FLOWFACTORY_ENV_FILE:-${SOURCE_CHECKOUT}/.env}"
flowfactory_require_env RUNTIME_ROOT REPO_ROOT CONDA_ENV PYTHON_OVERLAY \
  LOCAL_HOME LOCAL_CACHE LOCAL_TMP FLOWFACTORY_HF_HOME CUDA_LIBRARY_PATH \
  WANDB_MODE HF_HUB_OFFLINE

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

if [[ "${1:-}" == -- ]]; then
  shift
fi
(( $# > 0 )) || die "usage: $0 -- COMMAND [ARG ...]"
[[ -f "${RUNTIME_ROOT}/.ready" ]] || \
  die "runtime is not staged; run scripts/prepare_flowfactory_runtime.sh first"
[[ -d "${REPO_ROOT}" ]] || die "repository not found: ${REPO_ROOT}"
[[ -x "${CONDA_ENV}/bin/python" ]] || die "Python not found in ${CONDA_ENV}"
[[ -d "${FLOWFACTORY_HF_HOME}" ]] || \
  die "Hugging Face cache not found: ${FLOWFACTORY_HF_HOME}"

mkdir -p "${RUNTIME_ROOT}/logs" "${RUNTIME_ROOT}/runs" "${LOCAL_HOME}" \
  "${LOCAL_CACHE}/triton" "${LOCAL_CACHE}/torch" "${LOCAL_CACHE}/wandb" \
  "${LOCAL_TMP}"

export PYTHONUNBUFFERED=1
export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONPATH="${PYTHON_OVERLAY}:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${CUDA_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HOME="${LOCAL_HOME}"
export XDG_CACHE_HOME="${LOCAL_CACHE}"
export TRITON_CACHE_DIR="${LOCAL_CACHE}/triton"
export TORCH_HOME="${LOCAL_CACHE}/torch"
export TMPDIR="${LOCAL_TMP}"
export WANDB_CACHE_DIR="${LOCAL_CACHE}/wandb"
export WANDB_DIR="${RUNTIME_ROOT}/runs/wandb"
export HF_HOME="${FLOWFACTORY_HF_HOME}"
export HF_HUB_OFFLINE
export WANDB_MODE

cd "${REPO_ROOT}"
exec "$@"
