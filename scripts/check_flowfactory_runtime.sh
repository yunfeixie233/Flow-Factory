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
flowfactory_require_env RUNTIME_ROOT REPO_ROOT CONDA_ENV FLOWFACTORY_HF_HOME \
  CUDA_LIBRARY_PATH MODEL_CACHE CLIP_CACHE PICKSCORE_PROCESSOR_CACHE \
  PICKSCORE_MODEL_CACHE CLIPSCORE_CACHE DATASET_NAME PICKAPIC_DATASET_NAME \
  PICKAPIC_REWRITE_DATASET_NAME \
  HPSV2_CHECKPOINT

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

[[ -f "${RUNTIME_ROOT}/.ready" ]] || die "runtime readiness marker is absent"
[[ -x "${CONDA_ENV}/bin/python" ]] || die "local Python is absent"
[[ -d "${REPO_ROOT}" ]] || die "staged repository is absent"
[[ -d "${FLOWFACTORY_HF_HOME}/${MODEL_CACHE}" ]] || die "SD3.5 cache is absent"
[[ -d "${FLOWFACTORY_HF_HOME}/${CLIP_CACHE}" ]] || die "OpenCLIP cache is absent"
[[ -d "${FLOWFACTORY_HF_HOME}/${PICKSCORE_PROCESSOR_CACHE}" ]] || \
  die "PickScore processor cache is absent"
[[ -d "${FLOWFACTORY_HF_HOME}/${PICKSCORE_MODEL_CACHE}" ]] || \
  die "PickScore model cache is absent"
[[ -d "${FLOWFACTORY_HF_HOME}/${CLIPSCORE_CACHE}" ]] || die "CLIPScore cache is absent"
[[ -f "${RUNTIME_ROOT}/data/${DATASET_NAME}/train.jsonl" ]] || \
  die "staged training dataset is absent"
[[ -f "${RUNTIME_ROOT}/data/${PICKAPIC_DATASET_NAME}/train.txt" ]] || \
  die "staged Pick-a-Pic training split is absent"
[[ -f "${RUNTIME_ROOT}/data/${PICKAPIC_DATASET_NAME}/test.txt" ]] || \
  die "staged Pick-a-Pic test split is absent"
[[ -f "${RUNTIME_ROOT}/data/${PICKAPIC_REWRITE_DATASET_NAME}/train.jsonl" ]] || \
  die "staged Pick-a-Pic rewrite training split is absent"
[[ -f "${RUNTIME_ROOT}/data/${PICKAPIC_REWRITE_DATASET_NAME}/test.txt" ]] || \
  die "staged Pick-a-Pic rewrite test split is absent"
[[ -f "${RUNTIME_ROOT}/data/${PICKAPIC_REWRITE_DATASET_NAME}/manifest.json" ]] || \
  die "staged Pick-a-Pic rewrite manifest is absent"
[[ -f "${HPSV2_CHECKPOINT}" ]] || die "HPSv2.1 checkpoint is absent"

"${CONDA_ENV}/bin/python" - "${RUNTIME_ROOT}" <<'PY'
import os
import sys
import sysconfig

root = os.path.realpath(sys.argv[1])
paths = {
    "prefix": sys.prefix,
    "base_prefix": sys.base_prefix,
    "stdlib": sysconfig.get_path("stdlib"),
}
for label, path in paths.items():
    resolved = os.path.realpath(path)
    if os.path.commonpath([root, resolved]) != root:
        raise SystemExit(f"error: Python {label} is not local: {resolved}")
    print(f"Python {label}: {resolved}")
PY

"${CONDA_ENV}/bin/python" -c \
  'import hpsv2; from hpsv2.src.open_clip import get_tokenizer; get_tokenizer("ViT-H-14")'

printf 'Runtime ready: %s\n' "${RUNTIME_ROOT}"
printf 'Staged code:   %s\n' "${REPO_ROOT}"
printf 'CUDA path:    %s\n' "${CUDA_LIBRARY_PATH}"
printf '%s\n' 'Local footprints:'
du -sh "${CONDA_ENV}" "${FLOWFACTORY_HF_HOME}" \
  "${RUNTIME_ROOT}/cache/flow_factory/datasets" "${RUNTIME_ROOT}/data" 2>/dev/null || true
