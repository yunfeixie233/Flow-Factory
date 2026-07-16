#!/usr/bin/env bash

set -Eeuo pipefail

# Launch the third, row-preserving no-rewrite ablation through the common
# GenEval training launcher.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG="examples/nft/lora/sd3_5/geneval_no_rewrite.yaml"
exec "${SCRIPT_DIR}/train_nft_geneval_baseline.sh"
