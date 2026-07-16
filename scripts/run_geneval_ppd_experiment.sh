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
# Stock-GenEval DiffusionNFT + PPD: self-contained cold-node experiment runner
# =============================================================================
#
# Takes a fresh Pluto node from nothing to a running experiment arm. Safe to
# copy to a new machine on its own; every other input comes from the shared
# Sensei filesystem and the Git remote.
#
# WHAT THIS ARM IS
#   Baseline recipe: examples/nft/lora/sd3_5/geneval_stock.yaml — the stock
#   Flow-Factory GenEval dataset (dataset/geneval, 33,199 original prompts).
#   W&B baseline: vlaa-med/Flow-Factory/m0aunm6h (finished at step 319).
#   The PPD arms differ from that baseline ONLY by run_name and the ppd block
#   (knowledge-intrinsic rewrites enter through the auxiliary loss alone;
#   rollouts and the GenEval reward stay on original prompts).
#
# PREREQUISITES (new machine)
#   - /sensei-fs-3/users/yunfeix mounted (durable models, datasets, records).
#   - 8 idle GPUs; ~200 GB free on /mnt/localssd.
#   - git; network access to github.com (first run clones the fork).
#   - W&B (optional): a ~/.netrc with api.wandb.ai credentials on the node, or
#     WANDB_MODE=offline (the .env.example default) and sync later.
#
# USAGE
#   ./run_geneval_ppd_experiment.sh                    # auxiliary arm (rho=100)
#   PPD_ARM=control ./run_geneval_ppd_experiment.sh    # matched rho=0 control
#
#   Environment overrides (all optional):
#     FLOWFACTORY_CHECKOUT  durable source checkout (default below; cloned if absent)
#     WANDB_ENTITY          default vlaa-med (beside baseline m0aunm6h)
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
#   geneval_stock.yaml sets no train.max_epochs: the run continues until you
#   stop it (checkpoints every 20 epochs, eval every 20). For a finite matched
#   pair, add the same train.max_epochs to BOTH arm YAMLs before launching.
#
# HEALTHY FIRST-EPOCH METRICS (8xH100, measured 2026-07-16)
#   ppd/data_coverage_rate = 1.0        ppd/data_active_rate ~ 0.38
#   ppd/to_native_abs_loss ~ 0.0093 (aux; rho=100)   ppd/control_zero = 0 (control)
#   train/policy_loss ~ 24              reward_geneval_mean rising from ~0.19
#
# MONITORING
#   tail -f "${RUNTIME_ROOT}/logs"/nft_*.log        # console metrics every step
#   nvidia-smi                                       # 8 ranks, ~17-30 GiB each
# =============================================================================

REPO_URL="${REPO_URL:-https://github.com/yunfeixie233/Flow-Factory.git}"
FLOWFACTORY_CHECKOUT="${FLOWFACTORY_CHECKOUT:-/sensei-fs-3/users/yunfeix/origin_flowfactory/Flow-Factory}"
PPD_ARM="${PPD_ARM:-aux}"
export WANDB_ENTITY="${WANDB_ENTITY:-vlaa-med}"

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

case "${PPD_ARM}" in
  aux) arm_config="examples/nft/lora/sd3_5/geneval_stock_ppd.yaml" ;;
  control) arm_config="examples/nft/lora/sd3_5/geneval_stock_ppd_control.yaml" ;;
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
printf '\n================ GenEval PPD experiment ================\n'
printf 'Arm:          %s\n' "${PPD_ARM}"
printf 'Config:       %s\n' "${CONFIG}"
printf 'W&B entity:   %s (baseline m0aunm6h lives in vlaa-med)\n' "${WANDB_ENTITY}"
printf 'Reminder:     control must log ppd/control_zero == 0 every step\n'
printf '=========================================================\n\n'
exec ./scripts/train_nft_geneval_stock_ppd.sh
