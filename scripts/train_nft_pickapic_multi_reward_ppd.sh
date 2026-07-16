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

# Pick-a-Pic three-reward DiffusionNFT + privileged-prompt distillation (PPD)
# launcher. Operating procedure (see guidance/storage_and_training.md and the
# PPD section of guidance/algorithms.md):
#
#   ./scripts/prepare_flowfactory_runtime.sh          # stage runtime on SSD
#   ./scripts/check_flowfactory_runtime.sh            # verify the staged runtime
#   ./scripts/prepare_ppd_records.sh                  # stage PPD records on SSD
#   ./scripts/train_nft_pickapic_multi_reward_ppd.sh  # auxiliary arm (rho=7)
#
# The matched control arm differs only in rho and run identity:
#   CONFIG=examples/nft/lora/sd3_5/pickapic_multi_reward_ppd_control.yaml \
#     ./scripts/train_nft_pickapic_multi_reward_ppd.sh
# Its ppd/control_zero metric must log exactly 0 at every step.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_CHECKOUT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export CONFIG="${CONFIG:-examples/nft/lora/sd3_5/pickapic_multi_reward_ppd.yaml}"

# PPD records are staged separately from the main preparer; fail fast with the
# staging command instead of failing later inside trainer initialization.
source "${SCRIPT_DIR}/lib/load_env.sh"
flowfactory_load_env "${FLOWFACTORY_ENV_FILE:-${SOURCE_CHECKOUT}/.env}"
flowfactory_require_env RUNTIME_ROOT PICKAPIC_REWRITE_DATASET_NAME
if [[ ! -f "${RUNTIME_ROOT}/data/${PICKAPIC_REWRITE_DATASET_NAME}/records.jsonl" ]]; then
  printf 'error: %s\n' \
    "PPD records are not staged; run scripts/prepare_ppd_records.sh first" >&2
  exit 1
fi

"${SCRIPT_DIR}/check_flowfactory_runtime.sh"
exec "${SCRIPT_DIR}/train_nft_geneval_baseline.sh"
