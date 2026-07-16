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

"""Build privileged-prompt records for the stock GenEval training dataset.

The knowledge-intrinsic rewrite cache is keyed by the 50,000-row expanded
GenEval dataset, whose unique original prompts are exactly the stock GenEval
training prompts. This builder joins the stock rows to those rewrites by exact
prompt text (first occurrence wins for duplicated originals) and writes a
row-keyed ``records.jsonl`` for ``ppd.records_path``, plus a ``manifest.json``
binding the output to its inputs by SHA-256. Stock rows without a changed
rewrite fall back to the identity record with ``changed=false`` so coverage
stays total and identity rows are masked inactive by ``ppd.mask_identity``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

RECORD_SCHEMA_VERSION = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_stock_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"{path}:{line_number}: missing prompt field")
            prompts.append(prompt)
    if not prompts:
        raise ValueError(f"no rows in stock dataset {path}")
    return prompts


def load_rewrite_map(path: Path) -> tuple[dict[str, tuple[str, bool]], int]:
    mapping: dict[str, tuple[str, bool]] = {}
    conflicts = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("schema_version") != RECORD_SCHEMA_VERSION:
                raise ValueError(
                    f"{path}:{line_number}: unsupported schema_version "
                    f"{record.get('schema_version')!r}"
                )
            original = record["original_prompt"]
            conditioning = record["conditioning_prompt"]
            changed = bool(record["changed"])
            if changed == (conditioning == original):
                raise ValueError(f"{path}:{line_number}: changed flag contradicts the prompts")
            existing = mapping.get(original)
            if existing is None:
                mapping[original] = (conditioning, changed)
            elif existing[0] != conditioning:
                conflicts += 1
    if not mapping:
        raise ValueError(f"no records in rewrite source {path}")
    return mapping, conflicts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stock-train",
        type=Path,
        required=True,
        help="Stock GenEval train.jsonl (rows keyed by position).",
    )
    parser.add_argument(
        "--rewrite-records",
        type=Path,
        required=True,
        help="Knowledge-intrinsic pairs records.jsonl providing the rewrites.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for records.jsonl and manifest.json.",
    )
    args = parser.parse_args()

    prompts = load_stock_prompts(args.stock_train)
    rewrite_map, conflicts = load_rewrite_map(args.rewrite_records)

    matched = 0
    changed = 0
    records: list[dict[str, object]] = []
    for index, prompt in enumerate(prompts):
        entry = rewrite_map.get(prompt)
        if entry is not None:
            matched += 1
            conditioning, is_changed = entry
        else:
            conditioning, is_changed = prompt, False
        if is_changed:
            changed += 1
        records.append(
            {
                "schema_version": RECORD_SCHEMA_VERSION,
                "dataset_index": index,
                "original_prompt": prompt,
                "conditioning_prompt": conditioning,
                "changed": is_changed,
                "source": "geneval_knowledge_intrinsic_v0_pairs",
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=args.output_dir,
        prefix="records.jsonl.tmp.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    records_path = args.output_dir / "records.jsonl"
    os.replace(temporary, records_path)

    manifest = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "variant": "geneval_stock_ppd_v0",
        "rows": len(records),
        "matched_rows": matched,
        "changed_rows": changed,
        "rewrite_conflicts_first_wins": conflicts,
        "source_stock_train_sha256": sha256(args.stock_train),
        "source_rewrite_records_sha256": sha256(args.rewrite_records),
        "records_sha256": sha256(records_path),
    }
    manifest_path = args.output_dir / "manifest.json"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=args.output_dir,
        prefix="manifest.json.tmp.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, manifest_path)

    print(
        f"rows={len(records)} matched={matched} "
        f"({100.0 * matched / len(records):.2f}%) changed={changed} "
        f"({100.0 * changed / len(records):.2f}%) conflicts={conflicts}"
    )
    print(f"records: {records_path}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
