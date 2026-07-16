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

# =============================================================================
# Pick-a-Pic 3-reward DiffusionNFT + PPD: self-contained cold-node runner
# =============================================================================
#
# Takes a fresh Pluto node from nothing to a running experiment arm. Safe to
# copy to a new machine on its own; every other input comes from the shared
# Sensei filesystem and the Git remote.
#
# WHAT THIS ARM IS
#   Baseline recipe: examples/nft/lora/sd3_5/pickapic_multi_reward.yaml — the
#   ORIGINAL AdvantageFlow Pick-a-Pic split (25,432 train / 2,048 test original
#   prompts, no pre-rewriting), PickScore + HPSv2.1 + CLIPScore at equal weight,
#   fp16, 1001 epochs (~6 h on 8xH100).
#   W&B baseline: i2r-ali/Flow-Factory/6nyi9wvw (finished at step 1000).
#   The PPD arms differ from that baseline ONLY by run_name and the ppd block
#   (balanced_v0 rewrites enter through the auxiliary loss alone; conditioning
#   and all three rewards stay on original prompts).
#
# PREREQUISITES (new machine)
#   - /sensei-fs-3/users/yunfeix mounted (durable models, datasets, records).
#   - 8 idle GPUs; ~200 GB free on /mnt/localssd.
#   - git; network access to github.com (first run clones the fork).
#   - W&B (optional): a ~/.netrc with api.wandb.ai credentials on the node, or
#     WANDB_MODE=offline (the .env.example default) and sync later.
#
# USAGE
#   ./run_pickapic_ppd_experiment.sh                   # auxiliary arm (rho=8)
#   PPD_ARM=control ./run_pickapic_ppd_experiment.sh   # matched rho=0 control
#
#   Environment overrides (all optional):
#     FLOWFACTORY_CHECKOUT  durable source checkout (default below; cloned if absent)
#     WANDB_ENTITY          default i2r-ali (beside baseline 6nyi9wvw)
#     WANDB_MODE            online|offline (caller value wins over .env)
#     CHECKPOINT_S3_URI     s3://... run prefix; EMPTY means checkpoints are
#                           local-only and DIE WITH THE NODE (retention 2).
#     CONFIG                full config override (advanced; replaces PPD_ARM)
#
# MATCHED-PAIR DISCIPLINE
#   Run BOTH arms with the same seed (42, from the YAMLs). The causal estimate
#   is aux minus control at matched steps. On the control, train/ppd/control_zero
#   must log exactly 0 at every step — if it ever isn't, stop and investigate.
#
# STOPPING POLICY
#   train.max_epochs is 1001 (epochs 0..1000), matching baseline 6nyi9wvw:
#   the run ends on its own with eval every 50 epochs and saves every 100.
#
# HEALTHY FIRST-EPOCH METRICS (8xH100, measured 2026-07-16)
#   ppd/data_coverage_rate = 1.0        ppd/data_active_rate ~ 0.87
#   ppd/to_native_abs_loss ~ 0.0097 (aux; rho=8)   ppd/control_zero = 0 (control)
#   train/policy_loss ~ 2.0             reward_mean (3-score sum) ~ 1.19 rising
#
# MONITORING
#   tail -f "${RUNTIME_ROOT}/logs"/nft_*.log        # console metrics every step
#   nvidia-smi                                       # 8 ranks, ~29-30 GiB each
# =============================================================================

REPO_URL="${REPO_URL:-https://github.com/yunfeixie233/Flow-Factory.git}"
FLOWFACTORY_CHECKOUT="${FLOWFACTORY_CHECKOUT:-/sensei-fs-3/users/yunfeix/origin_flowfactory/Flow-Factory}"
PPD_ARM="${PPD_ARM:-aux}"
export WANDB_ENTITY="${WANDB_ENTITY:-i2r-ali}"

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

case "${PPD_ARM}" in
  aux) arm_config="examples/nft/lora/sd3_5/pickapic_multi_reward_ppd.yaml" ;;
  control) arm_config="examples/nft/lora/sd3_5/pickapic_multi_reward_ppd_control.yaml" ;;
  *) die "PPD_ARM must be 'aux' or 'control', got '${PPD_ARM}'" ;;
esac
export CONFIG="${CONFIG:-${arm_config}}"

# --- 1. Durable checkout ------------------------------------------------
if [[ ! -d "${FLOWFACTORY_CHECKOUT}/.git" ]]; then
  printf 'cloning %s -> %s\n' "${REPO_URL}" "${FLOWFACTORY_CHECKOUT}"
  git clone "${REPO_URL}" "${FLOWFACTORY_CHECKOUT}"
fi
cd "${FLOWFACTORY_CHECKOUT}"
[[ -x scripts/prepare_flowfactory_runtime.sh ]] || \
  die "checkout at ${FLOWFACTORY_CHECKOUT} is missing the runtime scripts; pull the latest main"

# --- 2. Machine-local .env ----------------------------------------------
if [[ ! -f .env ]]; then
  printf 'creating .env from .env.example (WANDB_MODE defaults to offline there)\n'
  cp .env.example .env
fi

# --- 3. Stage the node-local runtime (idempotent prewarm barrier) --------
./scripts/prepare_flowfactory_runtime.sh
./scripts/check_flowfactory_runtime.sh
./scripts/prepare_ppd_records.sh

# --- 4. Launch -----------------------------------------------------------
printf '\n=============== Pick-a-Pic PPD experiment ===============\n'
printf 'Arm:          %s\n' "${PPD_ARM}"
printf 'Config:       %s\n' "${CONFIG}"
printf 'W&B entity:   %s (baseline 6nyi9wvw lives in i2r-ali)\n' "${WANDB_ENTITY}"
printf 'Reminder:     control must log ppd/control_zero == 0 every step\n'
printf '=========================================================\n\n'
exec ./scripts/train_nft_pickapic_multi_reward_ppd.sh
