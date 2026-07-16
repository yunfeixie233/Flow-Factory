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

# Cold-node bootstrap. Expensive copies happen here once, never in the launcher.
# A production Pluto job should replace this bootstrap with a versioned Docker
# image and SOURCE_HF with an approved artifact prefix. This fallback installs
# its own Python on local SSD so runtime imports never depend on Sensei FS.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_CHECKOUT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/lib/load_env.sh"
FLOWFACTORY_ENV_PATH="${FLOWFACTORY_ENV_FILE:-${SOURCE_CHECKOUT}/.env}"
flowfactory_load_env "${FLOWFACTORY_ENV_PATH}"
flowfactory_require_env SOURCE_REPO SOURCE_HF RUNTIME_ROOT \
  MODEL_CACHE CLIP_CACHE PICKSCORE_PROCESSOR_CACHE PICKSCORE_MODEL_CACHE \
  CLIPSCORE_CACHE PICKSCORE_PROCESSOR_REPO PICKSCORE_PROCESSOR_REVISION \
  PICKSCORE_MODEL_REPO PICKSCORE_MODEL_REVISION CLIPSCORE_REPO \
  CLIPSCORE_REVISION DATASET_SOURCE DATASET_NAME NO_REWRITE_DATASET_NAME \
  PICKAPIC_DATASET_SOURCE PICKAPIC_DATASET_NAME \
  PICKAPIC_REWRITE_DATASET_SOURCE PICKAPIC_REWRITE_DATASET_NAME \
  MMCV_WHEEL CUDA_ROOT MAX_JOBS TORCH_CUDA_ARCH_LIST MMCV_WITH_OPS FORCE_CUDA \
  GENEVAL_BUILD_DIR GENEVAL_MMCV_VERSION GENEVAL_DETECTOR_URL \
  GENEVAL_DETECTOR_SHA256 MINIFORGE_URL MINIFORGE_SHA256 \
  RUNTIME_PYTHON_VERSION NETRC_SOURCE HPSV2_CHECKPOINT_URL \
  HPSV2_CHECKPOINT_SHA256 HPSV2_CHECKPOINT HPSV2_BPE_URL HPSV2_BPE_SHA256

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
command -v curl >/dev/null || die "curl is required for cold-node preparation"
[[ -d "${SOURCE_HF}/${MODEL_CACHE}" ]] || die "SD3.5 cache not found under ${SOURCE_HF}"
[[ -d "${SOURCE_HF}/${CLIP_CACHE}" ]] || die "OpenCLIP cache not found under ${SOURCE_HF}"
[[ -f "${DATASET_SOURCE}/train.jsonl" ]] || die "dataset not found: ${DATASET_SOURCE}"
[[ -f "${DATASET_SOURCE}/test.jsonl" ]] || die "dataset test split not found: ${DATASET_SOURCE}"
[[ -f "${PICKAPIC_DATASET_SOURCE}/train.txt" ]] || \
  die "Pick-a-Pic training split not found: ${PICKAPIC_DATASET_SOURCE}"
[[ -f "${PICKAPIC_DATASET_SOURCE}/test.txt" ]] || \
  die "Pick-a-Pic test split not found: ${PICKAPIC_DATASET_SOURCE}"
for rewrite_file in manifest.json records.jsonl train.txt original_train.txt; do
  [[ -f "${PICKAPIC_REWRITE_DATASET_SOURCE}/${rewrite_file}" ]] || \
    die "Pick-a-Pic rewrite artifact is missing: ${PICKAPIC_REWRITE_DATASET_SOURCE}/${rewrite_file}"
done
if pgrep -f '[f]low_factory.train' >/dev/null; then
  die "refusing to modify the staged runtime while Flow-Factory training is active"
fi

mkdir -p "${RUNTIME_ROOT}" "${RUNTIME_ROOT}/cache/huggingface/hub" \
  "${RUNTIME_ROOT}/cache/flow_factory/datasets" "${RUNTIME_ROOT}/cache/torch" \
  "${RUNTIME_ROOT}/cache/pip" \
  "${RUNTIME_ROOT}/cache/bootstrap" "${RUNTIME_ROOT}/cache/conda/pkgs" \
  "${RUNTIME_ROOT}/data/${DATASET_NAME}" "${RUNTIME_ROOT}/python-overlay" \
  "${RUNTIME_ROOT}/data/${NO_REWRITE_DATASET_NAME}" \
  "${RUNTIME_ROOT}/data/${PICKAPIC_DATASET_NAME}" \
  "${RUNTIME_ROOT}/data/${PICKAPIC_REWRITE_DATASET_NAME}" \
  "${RUNTIME_ROOT}/runs" "${RUNTIME_ROOT}/runs/wandb" "${RUNTIME_ROOT}/logs" \
  "${RUNTIME_ROOT}/tmp" "${RUNTIME_ROOT}/home" "${RUNTIME_ROOT}/code"
rm -f "${RUNTIME_ROOT}/.ready"

# Preparation itself must not populate or reuse a user-home cache on Sensei.
# Keeping these paths below RUNTIME_ROOT also makes an empty-root benchmark a
# genuine cold install while allowing intentional reuse on repeated prepares.
export HOME="${RUNTIME_ROOT}/home"
export XDG_CACHE_HOME="${RUNTIME_ROOT}/cache"
export PIP_CACHE_DIR="${RUNTIME_ROOT}/cache/pip"

# Valid fingerprinted caches are intentionally reused. Only abandoned build
# directories from a process that died more than an hour ago are removed.
find "${RUNTIME_ROOT}/cache/flow_factory/datasets" -mindepth 1 -maxdepth 1 \
  -type d -name '*.tmp' -mmin +60 -print -exec rm -rf {} +

# HOME is deliberately local so libraries never write to the quota-limited
# container home. Preserve online W&B authentication without exposing the key.
if [[ -f "${NETRC_SOURCE}" ]]; then
  install -m 600 "${NETRC_SOURCE}" "${RUNTIME_ROOT}/home/.netrc"
fi

stage_code() {
  local destination="${RUNTIME_ROOT}/code/Flow-Factory"
  local temporary="${destination}.tmp"
  rm -rf "${temporary}"
  mkdir -p "${temporary}"
  tar -C "${SOURCE_REPO}" \
    --exclude=.git --exclude=.scratch --exclude=saves --exclude=wandb \
    --exclude='*.pyc' --exclude=__pycache__ -cf - . | tar -C "${temporary}" -xf -
  install -m 600 "${FLOWFACTORY_ENV_PATH}" "${temporary}/.env"
  rm -rf "${destination}"
  mv "${temporary}" "${destination}"
  printf 'ready: local code checkout\n'
}

stage_code

download_verified() {
  local url=$1 expected_sha256=$2 destination=$3 label=$4
  local actual temporary
  if [[ -f "${destination}" ]]; then
    actual="$(sha256sum "${destination}" | awk '{print $1}')"
    if [[ "${actual}" == "${expected_sha256}" ]]; then
      printf 'ready: %s\n' "${label}"
      return
    fi
    rm -f "${destination}"
  fi
  printf 'downloading: %s\n' "${label}"
  mkdir -p "$(dirname "${destination}")"
  temporary="${destination}.tmp.$$"
  rm -f "${temporary}"
  curl -fL --retry 3 --retry-delay 2 -o "${temporary}" "${url}"
  actual="$(sha256sum "${temporary}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected_sha256}" ]]; then
    rm -f "${temporary}"
    die "${label} SHA-256 mismatch: expected ${expected_sha256}, got ${actual}"
  fi
  mv "${temporary}" "${destination}"
}

stage_tree() {
  local source=$1 destination=$2 sentinel=$3 label=$4
  if [[ -f "${sentinel}" && -d "${destination}" ]]; then
    printf 'ready: %s\n' "${label}"
    return
  fi
  printf 'staging: %s\n' "${label}"
  mkdir -p "${destination}"
  # GNU cp preserves the Hugging Face symlink layout and is available in the
  # minimal review image (rsync is not). A sentinel prevents repeat copies.
  cp -a "${source}/." "${destination}/"
  date -Ins > "${sentinel}"
}

stage_hf_snapshot() {
  local repo_id=$1 revision=$2 destination=$3 sentinel=$4 label=$5 profile=${6:-full}
  if [[ -f "${sentinel}" && -d "${destination}/snapshots/${revision}" ]]; then
    printf 'ready: %s\n' "${label}"
    return
  fi
  printf 'downloading: %s at pinned revision %s\n' "${label}" "${revision}"
  HF_HUB_OFFLINE=0 "${RUNTIME_ROOT}/env/bin/python" - \
    "${repo_id}" "${revision}" "${RUNTIME_ROOT}/cache/huggingface/hub" \
    "${profile}" <<'PY'
import os
import sys

from huggingface_hub import snapshot_download

repo_id, revision, cache_dir, profile = sys.argv[1:]
allow_patterns = None
if profile == "processor":
    allow_patterns = ["*.json", "*.txt", "*.model", "*.jinja"]
elif profile != "full":
    raise SystemExit(f"unknown Hugging Face snapshot profile: {profile}")
path = snapshot_download(
    repo_id=repo_id,
    revision=revision,
    cache_dir=cache_dir,
    allow_patterns=allow_patterns,
)
expected = os.path.join("snapshots", revision)
if expected not in os.path.realpath(path):
    raise SystemExit(f"unexpected snapshot path: {path}")
repo_cache = os.path.dirname(os.path.dirname(path))
refs = os.path.join(repo_cache, "refs")
os.makedirs(refs, exist_ok=True)
with open(os.path.join(refs, "main"), "w", encoding="utf-8") as handle:
    handle.write(revision)
PY
  date -Ins > "${sentinel}"
}

stage_dataset() {
  local source=$1 destination=$2 sentinel=$3
  local expected current temporary
  expected="$(sha256sum "${source}/train.jsonl" "${source}/test.jsonl" | awk '{print $1}')"
  current="$(cat "${sentinel}" 2>/dev/null || true)"
  if [[ "${current}" == "${expected}" && -f "${destination}/train.jsonl" && \
        -f "${destination}/test.jsonl" ]]; then
    printf 'ready: GenEval dataset\n'
    return
  fi

  printf 'staging: GenEval dataset\n'
  rm -rf "${destination}"
  mkdir -p "${destination}"
  cp -a "${source}/." "${destination}/"
  temporary="${sentinel}.tmp.$$"
  printf '%s\n' "${expected}" > "${temporary}"
  mv "${temporary}" "${sentinel}"
}

stage_pickapic_dataset() {
  local source=$1 destination=$2 sentinel=$3
  local expected current temporary
  expected="$(sha256sum "${source}/train.txt" "${source}/test.txt" | sha256sum | awk '{print $1}')"
  current="$(cat "${sentinel}" 2>/dev/null || true)"
  if [[ "${current}" == "${expected}" && -f "${destination}/train.txt" && \
        -f "${destination}/test.txt" ]]; then
    printf 'ready: AdvantageFlow Pick-a-Pic dataset\n'
    return
  fi

  printf 'staging: AdvantageFlow Pick-a-Pic dataset\n'
  rm -rf "${destination}"
  mkdir -p "${destination}"
  cp -a "${source}/train.txt" "${source}/test.txt" "${destination}/"
  temporary="${sentinel}.tmp.$$"
  printf '%s\n' "${expected}" > "${temporary}"
  mv "${temporary}" "${sentinel}"
}

prepare_env() {
  local env="${RUNTIME_ROOT}/env"
  local conda_root="${RUNTIME_ROOT}/miniforge"
  local installer="${RUNTIME_ROOT}/cache/bootstrap/miniforge-${MINIFORGE_SHA256}.sh"
  local conda_expected conda_current expected current temporary
  conda_expected="$(printf '%s\n' "${MINIFORGE_URL}" "${MINIFORGE_SHA256}" \
    | sha256sum | awk '{print $1}')"
  conda_current="$(cat "${RUNTIME_ROOT}/.conda-ready" 2>/dev/null || true)"
  if [[ "${conda_current}" != "${conda_expected}" || \
        ! -x "${conda_root}/bin/conda" ]]; then
    download_verified "${MINIFORGE_URL}" "${MINIFORGE_SHA256}" \
      "${installer}" "pinned Miniforge installer"
    printf 'staging: local Miniforge bootstrap\n'
    rm -rf "${conda_root}" "${RUNTIME_ROOT}/.conda-ready"
    bash "${installer}" -b -p "${conda_root}"
    temporary="${RUNTIME_ROOT}/.conda-ready.tmp.$$"
    printf '%s\n' "${conda_expected}" > "${temporary}"
    mv "${temporary}" "${RUNTIME_ROOT}/.conda-ready"
  else
    printf 'ready: local Miniforge bootstrap\n'
  fi
  expected="$({
    sha256sum \
      "${SOURCE_REPO}/pyproject.toml" \
      "${SOURCE_REPO}/config/runtime/geneval-h100-requirements.txt" \
      "${SOURCE_REPO}/config/runtime/geneval-reward-requirements.txt" \
      "${SOURCE_REPO}/config/runtime/pickapic-reward-requirements.txt" \
      "${SOURCE_REPO}/config/runtime/pickapic-reward-no-deps-requirements.txt"
    printf '%s\n' "${conda_expected}" "${RUNTIME_PYTHON_VERSION}"
  } | sha256sum | awk '{print $1}')"
  current="$(cat "${RUNTIME_ROOT}/.env-ready" 2>/dev/null || true)"
  if [[ "${current}" == "${expected}" ]] && "${env}/bin/python" -c \
      'import os, sys, sysconfig, torch, diffusers, transformers
root = os.path.realpath(sys.argv[1])
paths = [sys.prefix, sys.base_prefix, sysconfig.get_path("stdlib")]
assert all(os.path.commonpath([root, os.path.realpath(path)]) == root for path in paths)
assert torch.__version__.startswith("2.12.1")' "${RUNTIME_ROOT}" >/dev/null 2>&1; then
    printf 'ready: Python environment\n'
    return
  fi
  printf 'staging: self-contained Python %s environment on local SSD\n' \
    "${RUNTIME_PYTHON_VERSION}"
  rm -rf "${env}"
  CONDA_PKGS_DIRS="${RUNTIME_ROOT}/cache/conda/pkgs" \
    "${conda_root}/bin/conda" create -y -q -p "${env}" \
      "python=${RUNTIME_PYTHON_VERSION}" pip
  PIP_DISABLE_PIP_VERSION_CHECK=1 "${env}/bin/python" -m pip install \
    -e "${RUNTIME_ROOT}/code/Flow-Factory[deepspeed,wandb]" \
    -r "${RUNTIME_ROOT}/code/Flow-Factory/config/runtime/geneval-h100-requirements.txt" \
    -r "${RUNTIME_ROOT}/code/Flow-Factory/config/runtime/pickapic-reward-requirements.txt"
  PIP_DISABLE_PIP_VERSION_CHECK=1 "${env}/bin/python" -m pip install --no-deps \
    -r "${RUNTIME_ROOT}/code/Flow-Factory/config/runtime/pickapic-reward-no-deps-requirements.txt"
  temporary="${RUNTIME_ROOT}/.env-ready.tmp.$$"
  printf '%s\n' "${expected}" > "${temporary}"
  mv "${temporary}" "${RUNTIME_ROOT}/.env-ready"
}

prepare_hpsv2_bpe() {
  local purelib destination
  purelib="$("${RUNTIME_ROOT}/env/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
  destination="${purelib}/hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz"
  download_verified "${HPSV2_BPE_URL}" "${HPSV2_BPE_SHA256}" \
    "${destination}" "HPSv2 OpenCLIP BPE vocabulary"
  "${RUNTIME_ROOT}/env/bin/python" -c 'import hpsv2; from hpsv2.src.open_clip import get_tokenizer; get_tokenizer("ViT-H-14")'
  printf 'ready: HPSv2 Python package and tokenizer\n'
}

validate_geneval_cuda() {
  local env="${RUNTIME_ROOT}/env"
  LD_LIBRARY_PATH="${env}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "${env}/bin/python" - <<'PY'
import torch
from mmcv.ops import nms
from mmdet.apis import inference_detector, init_detector  # noqa: F401

assert torch.cuda.is_available(), "CUDA is unavailable"
boxes = torch.tensor(
    [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0], [20.0, 20.0, 30.0, 30.0]],
    device="cuda",
)
scores = torch.tensor([0.9, 0.8, 0.7], device="cuda")
_, keep = nms(boxes, scores, 0.5)
torch.cuda.synchronize()
assert keep.cpu().tolist() == [0, 2], keep
print(
    f"ready: MMCV CUDA ops on {torch.cuda.get_device_name()} "
    f"sm{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}"
)
PY
}

prepare_geneval() {
  local env="${RUNTIME_ROOT}/env"
  local cuda_target="${CUDA_ROOT}/targets/x86_64-linux"
  if [[ -f "${RUNTIME_ROOT}/.geneval-ready" ]] && \
      validate_geneval_cuda >/dev/null 2>&1; then
    printf 'ready: GenEval CUDA extensions\n'
    return
  fi
  printf 'staging: GenEval dependencies and CUDA extension\n'
  if [[ -f "${MMCV_WHEEL}" ]]; then
    printf 'staging: prebuilt MMCV wheel\n'
    PIP_DISABLE_PIP_VERSION_CHECK=1 "${env}/bin/python" -m pip install \
      --no-deps "${MMCV_WHEEL}"
  else
    printf 'building: MMCV wheel (no compatible artifact was supplied)\n'
    mkdir -p "${GENEVAL_BUILD_DIR}"
    if [[ ! -d "${GENEVAL_BUILD_DIR}/mmcv/.git" ]]; then
      git clone --depth 1 --branch "v${GENEVAL_MMCV_VERSION}" \
        https://github.com/open-mmlab/mmcv.git "${GENEVAL_BUILD_DIR}/mmcv"
    fi
    CPATH="${cuda_target}/include" LIBRARY_PATH="${cuda_target}/lib" \
      MMCV_WITH_OPS="${MMCV_WITH_OPS}" FORCE_CUDA="${FORCE_CUDA}" \
      TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" MAX_JOBS="${MAX_JOBS}" \
      CUDA_HOME="${CUDA_ROOT}" "${env}/bin/python" -m pip install \
        "${GENEVAL_BUILD_DIR}/mmcv" --no-build-isolation
  fi
  PIP_DISABLE_PIP_VERSION_CHECK=1 "${env}/bin/python" -m pip install \
    --no-deps -r "${RUNTIME_ROOT}/code/Flow-Factory/config/runtime/geneval-mmdet-requirement.txt"
  validate_geneval_cuda
  date -Ins > "${RUNTIME_ROOT}/.geneval-ready"
}

stage_geneval_detector() {
  local filename="${GENEVAL_DETECTOR_URL##*/}"
  download_verified "${GENEVAL_DETECTOR_URL}" "${GENEVAL_DETECTOR_SHA256}" \
    "${RUNTIME_ROOT}/cache/torch/hub/checkpoints/${filename}" \
    "GenEval Mask2Former checkpoint"
}

validate_no_rewrite_manifest() {
  local manifest=$1 source=$2
  "${RUNTIME_ROOT}/env/bin/python" - "${manifest}" "${source}" <<'PY'
import hashlib
import json
import sys


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

with open(sys.argv[1], encoding="utf-8") as handle:
    manifest = json.load(handle)
source = sys.argv[2]
assert manifest["variant"] == "no_rewrite", manifest
assert manifest["deduplicated"] is False, manifest
assert manifest["train_rows"] == 50_000, manifest
assert manifest["train_duplicate_rows_retained"] == 16_801, manifest
assert manifest["test_rows"] == 2_212, manifest
assert manifest["source_train_sha256"] == sha256(f"{source}/train.jsonl"), manifest
assert manifest["source_test_sha256"] == sha256(f"{source}/test.jsonl"), manifest
PY
}

prepare_no_rewrite_dataset() {
  local source="${RUNTIME_ROOT}/data/${DATASET_NAME}"
  local destination="${RUNTIME_ROOT}/data/${NO_REWRITE_DATASET_NAME}"
  local sentinel="${RUNTIME_ROOT}/.dataset-no-rewrite-ready"
  local manifest="${destination}/manifest.json"

  if [[ -f "${sentinel}" && -f "${destination}/train.jsonl" && \
        -f "${destination}/test.jsonl" && -f "${manifest}" ]] && \
      validate_no_rewrite_manifest "${manifest}" "${source}"; then
    printf 'ready: GenEval no-rewrite dataset\n'
    return
  fi

  printf 'building: row-preserving GenEval no-rewrite dataset\n'
  rm -rf "${destination}"
  "${RUNTIME_ROOT}/env/bin/python" \
    "${RUNTIME_ROOT}/code/Flow-Factory/scripts/build_geneval_no_rewrite_dataset.py" \
    --source-dir "${source}" \
    --output-dir "${destination}"
  validate_no_rewrite_manifest "${manifest}" "${source}"
  date -Ins > "${sentinel}"
}

prepare_pickapic_rewrite_dataset() {
  local source="${PICKAPIC_REWRITE_DATASET_SOURCE}"
  local destination="${RUNTIME_ROOT}/data/${PICKAPIC_REWRITE_DATASET_NAME}"
  local baseline_test="${RUNTIME_ROOT}/data/${PICKAPIC_DATASET_NAME}/test.txt"
  local sentinel="${RUNTIME_ROOT}/.pickapic-rewrite-dataset-ready"
  local expected current temporary

  expected="$({
    sha256sum \
      "${source}/manifest.json" \
      "${source}/records.jsonl" \
      "${source}/train.txt" \
      "${source}/original_train.txt" \
      "${baseline_test}"
  } | sha256sum | awk '{print $1}')"
  current="$(cat "${sentinel}" 2>/dev/null || true)"
  if [[ "${current}" == "${expected}" && -f "${destination}/train.jsonl" && \
        -f "${destination}/test.txt" && -f "${destination}/manifest.json" ]]; then
    printf 'ready: Pick-a-Pic balanced-v0 rewrite dataset\n'
    return
  fi

  printf 'building: Pick-a-Pic balanced-v0 conditioning/reward prompt pairs\n'
  rm -rf "${destination}"
  "${RUNTIME_ROOT}/env/bin/python" \
    "${RUNTIME_ROOT}/code/Flow-Factory/scripts/build_pickapic_rewrite_dataset.py" \
    --source-dir "${source}" \
    --baseline-test "${baseline_test}" \
    --output-dir "${destination}"
  temporary="${sentinel}.tmp.$$"
  printf '%s\n' "${expected}" > "${temporary}"
  mv "${temporary}" "${sentinel}"
}

prepare_env & env_pid=$!
stage_tree "${SOURCE_HF}/${MODEL_CACHE}" \
  "${RUNTIME_ROOT}/cache/huggingface/${MODEL_CACHE}" \
  "${RUNTIME_ROOT}/.sd35-ready" "SD3.5 model snapshot" & sd35_pid=$!
stage_tree "${SOURCE_HF}/${CLIP_CACHE}" \
  "${RUNTIME_ROOT}/cache/huggingface/${CLIP_CACHE}" \
  "${RUNTIME_ROOT}/.openclip-ready" "OpenCLIP ViT-L/14 snapshot" & clip_pid=$!
stage_dataset "${DATASET_SOURCE}" "${RUNTIME_ROOT}/data/${DATASET_NAME}" \
  "${RUNTIME_ROOT}/.dataset-ready" & dataset_pid=$!
stage_pickapic_dataset "${PICKAPIC_DATASET_SOURCE}" \
  "${RUNTIME_ROOT}/data/${PICKAPIC_DATASET_NAME}" \
  "${RUNTIME_ROOT}/.pickapic-dataset-ready" & pickapic_dataset_pid=$!
stage_geneval_detector & detector_pid=$!
download_verified "${HPSV2_CHECKPOINT_URL}" "${HPSV2_CHECKPOINT_SHA256}" \
  "${HPSV2_CHECKPOINT}" "HPSv2.1 checkpoint" & hpsv2_checkpoint_pid=$!

parallel_status=0
for job in \
  "environment:${env_pid}" \
  "SD3.5:${sd35_pid}" \
  "OpenCLIP:${clip_pid}" \
  "dataset:${dataset_pid}" \
  "Pick-a-Pic-dataset:${pickapic_dataset_pid}" \
  "detector:${detector_pid}" \
  "HPSv2.1-checkpoint:${hpsv2_checkpoint_pid}"; do
  label=${job%%:*}
  pid=${job##*:}
  if ! wait "${pid}"; then
    printf 'error: parallel preparation task failed: %s\n' "${label}" >&2
    parallel_status=1
  fi
done
(( parallel_status == 0 )) || die "cold-node parallel preparation failed"

stage_hf_snapshot "${PICKSCORE_PROCESSOR_REPO}" "${PICKSCORE_PROCESSOR_REVISION}" \
  "${RUNTIME_ROOT}/cache/huggingface/${PICKSCORE_PROCESSOR_CACHE}" \
  "${RUNTIME_ROOT}/.pickscore-processor-ready" "PickScore processor snapshot" \
  processor & pick_processor_pid=$!
stage_hf_snapshot "${PICKSCORE_MODEL_REPO}" "${PICKSCORE_MODEL_REVISION}" \
  "${RUNTIME_ROOT}/cache/huggingface/${PICKSCORE_MODEL_CACHE}" \
  "${RUNTIME_ROOT}/.pickscore-model-ready" "PickScore model snapshot" & pick_model_pid=$!
stage_hf_snapshot "${CLIPSCORE_REPO}" "${CLIPSCORE_REVISION}" \
  "${RUNTIME_ROOT}/cache/huggingface/${CLIPSCORE_CACHE}" \
  "${RUNTIME_ROOT}/.clipscore-ready" "CLIPScore model snapshot" & clipscore_pid=$!

reward_status=0
for job in \
  "PickScore-processor:${pick_processor_pid}" \
  "PickScore-model:${pick_model_pid}" \
  "CLIPScore:${clipscore_pid}"; do
  label=${job%%:*}
  pid=${job##*:}
  if ! wait "${pid}"; then
    printf 'error: reward snapshot preparation failed: %s\n' "${label}" >&2
    reward_status=1
  fi
done
(( reward_status == 0 )) || die "reward snapshot preparation failed"

prepare_geneval
prepare_hpsv2_bpe
prepare_no_rewrite_dataset
prepare_pickapic_rewrite_dataset

date -Ins > "${RUNTIME_ROOT}/.ready"
printf 'runtime ready: %s\n' "${RUNTIME_ROOT}"
