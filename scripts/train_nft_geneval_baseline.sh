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
flowfactory_require_env RUNTIME_ROOT REPO_ROOT CONDA_ENV FLOWFACTORY_HF_HOME CONFIG \
  CHECKPOINT_RETENTION CHECKPOINT_UPLOAD_TOOL \
  CHECKPOINT_UPLOAD_CONCURRENCY WANDB_MODE

CONFIG_STEM="$(basename "${CONFIG%.yaml}")"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${RUNTIME_ROOT}/logs/nft_${CONFIG_STEM}_${RUN_STAMP}.log}"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

[[ -d "${REPO_ROOT}" ]] || die "repository not found: ${REPO_ROOT}"
[[ -x "${CONDA_ENV}/bin/python" ]] || die "Python not found in ${CONDA_ENV}"
[[ -f "${REPO_ROOT}/${CONFIG}" ]] || die "config not found: ${REPO_ROOT}/${CONFIG}"
[[ -f "${RUNTIME_ROOT}/.ready" ]] || die "runtime is not staged; run scripts/prepare_flowfactory_runtime.sh first"
[[ -d "${FLOWFACTORY_HF_HOME}" ]] || die "Hugging Face cache not found: ${FLOWFACTORY_HF_HOME}"
[[ "${CHECKPOINT_RETENTION}" =~ ^[0-9]+$ ]] || \
  die "CHECKPOINT_RETENTION must be a non-negative integer"
[[ "${CHECKPOINT_UPLOAD_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || \
  die "CHECKPOINT_UPLOAD_CONCURRENCY must be a positive integer"
export FLOWFACTORY_CHECKPOINT_RETENTION="${CHECKPOINT_RETENTION}"

# Both matched variants use all eight visible GPUs. Do not let either launcher
# contend with any Flow-Factory trainer already active on the same machine.
if pgrep -f '[f]low_factory.train' >/dev/null; then
  die "a Flow-Factory training process is already running on this machine"
fi

mkdir -p "${RUNTIME_ROOT}/logs"
if [[ -n "${CHECKPOINT_S3_URI}" ]]; then
  [[ "${CHECKPOINT_S3_URI}" == s3://* ]] || die "CHECKPOINT_S3_URI must start with s3://"
  [[ "${CHECKPOINT_UPLOAD_TOOL}" == s5cmd ]] || \
    die "CHECKPOINT_UPLOAD_TOOL must be s5cmd"
  command -v "${CHECKPOINT_UPLOAD_TOOL}" >/dev/null || \
    die "${CHECKPOINT_UPLOAD_TOOL} is required when CHECKPOINT_S3_URI is set"
  export FLOWFACTORY_CHECKPOINT_S3_URI="${CHECKPOINT_S3_URI%/}"
  export FLOWFACTORY_CHECKPOINT_UPLOAD_TOOL="${CHECKPOINT_UPLOAD_TOOL}"
  export FLOWFACTORY_CHECKPOINT_UPLOAD_CONCURRENCY="${CHECKPOINT_UPLOAD_CONCURRENCY}"
fi

printf 'Repository: %s\n' "${REPO_ROOT}"
printf 'Config:     %s\n' "${CONFIG}"
printf 'W&B mode:   %s\n' "${WANDB_MODE}"
printf 'Log:        %s\n' "${LOG_FILE}"
printf 'S3:         %s\n' "${CHECKPOINT_S3_URI:-disabled}"
printf 'Upload:     %s (concurrency=%s)\n' \
  "${CHECKPOINT_UPLOAD_TOOL}" "${CHECKPOINT_UPLOAD_CONCURRENCY}"
printf 'Retention:  %s local checkpoint(s)\n' "${CHECKPOINT_RETENTION}"
if [[ -z "${CHECKPOINT_S3_URI}" ]]; then
  printf '%s\n' 'WARNING: checkpoints are local only and will be lost with the node.'
fi
printf '%s\n' 'Stopping policy: controlled by train.max_epochs in the selected YAML.'

set +e
"${SCRIPT_DIR}/run_in_flowfactory_runtime.sh" -- \
  python -m flow_factory.cli "${CONFIG}" 2>&1 | tee "${LOG_FILE}"
status=${PIPESTATUS[0]}
set -e

exit "${status}"
