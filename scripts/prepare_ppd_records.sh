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

# Stage privileged-prompt distillation records onto the local-SSD runtime.
# Run AFTER scripts/prepare_flowfactory_runtime.sh; training must read records
# from RUNTIME_ROOT, never from Sensei FS.
#
#   1. Stock-GenEval PPD records are derived from the knowledge-intrinsic
#      rewrite pairs by exact prompt-text join (see the Python builder).
#   2. Pick-a-Pic PPD records are the balanced_v0 pairs records, copied
#      verbatim beside the already-staged rewrite dataset.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_CHECKOUT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/lib/load_env.sh"
flowfactory_load_env "${FLOWFACTORY_ENV_FILE:-${SOURCE_CHECKOUT}/.env}"
flowfactory_require_env RUNTIME_ROOT REPO_ROOT CONDA_ENV \
  PICKAPIC_REWRITE_DATASET_SOURCE PICKAPIC_REWRITE_DATASET_NAME

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

[[ -f "${RUNTIME_ROOT}/.ready" ]] || \
  die "runtime is not staged; run scripts/prepare_flowfactory_runtime.sh first"
[[ -x "${CONDA_ENV}/bin/python" ]] || die "Python not found in ${CONDA_ENV}"

GENEVAL_PAIRS_SOURCE="${GENEVAL_PAIRS_SOURCE:-$(dirname "${PICKAPIC_REWRITE_DATASET_SOURCE}")/geneval_knowledge_intrinsic_v0_pairs}"
[[ -f "${GENEVAL_PAIRS_SOURCE}/records.jsonl" ]] || \
  die "knowledge-intrinsic pairs records not found: ${GENEVAL_PAIRS_SOURCE}/records.jsonl"
[[ -f "${REPO_ROOT}/dataset/geneval/train.jsonl" ]] || \
  die "staged stock GenEval dataset not found under ${REPO_ROOT}"

"${CONDA_ENV}/bin/python" \
  "${REPO_ROOT}/scripts/build_geneval_stock_ppd_records.py" \
  --stock-train "${REPO_ROOT}/dataset/geneval/train.jsonl" \
  --rewrite-records "${GENEVAL_PAIRS_SOURCE}/records.jsonl" \
  --output-dir "${RUNTIME_ROOT}/data/geneval_stock_ppd_pairs"

pickapic_destination="${RUNTIME_ROOT}/data/${PICKAPIC_REWRITE_DATASET_NAME}"
[[ -d "${pickapic_destination}" ]] || \
  die "staged Pick-a-Pic rewrite dataset not found: ${pickapic_destination}"
[[ -f "${PICKAPIC_REWRITE_DATASET_SOURCE}/records.jsonl" ]] || \
  die "Pick-a-Pic pairs records not found: ${PICKAPIC_REWRITE_DATASET_SOURCE}/records.jsonl"

# The balanced_v0 pairs predate the PPD record schema: normalize to
# schema_version 1 and recompute `changed` from the prompts, failing on any
# disagreement with the source flag instead of staging inconsistent rows.
"${CONDA_ENV}/bin/python" - \
  "${PICKAPIC_REWRITE_DATASET_SOURCE}/records.jsonl" \
  "${pickapic_destination}/records.jsonl" <<'PY'
import json
import os
import sys

source, destination = sys.argv[1:]
rows = changed_rows = 0
temporary = destination + ".tmp"
with open(source, encoding="utf-8") as src, open(temporary, "w", encoding="utf-8") as dst:
    for line_number, line in enumerate(src, start=1):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        original = record["original_prompt"]
        conditioning = record["conditioning_prompt"]
        changed = conditioning != original
        if bool(record.get("changed")) != changed:
            raise SystemExit(
                f"{source}:{line_number}: source changed flag disagrees with the prompts"
            )
        rows += 1
        changed_rows += changed
        dst.write(
            json.dumps(
                {
                    "schema_version": 1,
                    "dataset_index": record["dataset_index"],
                    "original_prompt": original,
                    "conditioning_prompt": conditioning,
                    "changed": changed,
                    "source": record.get("arm", "pickscore_balanced_v0_pairs"),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
os.replace(temporary, destination)
print(
    f"rows={rows} changed={changed_rows} ({100.0 * changed_rows / max(rows, 1):.2f}%)"
)
PY
printf 'ready: Pick-a-Pic PPD records at %s\n' "${pickapic_destination}/records.jsonl"
printf 'PPD records staged under %s/data\n' "${RUNTIME_ROOT}"
