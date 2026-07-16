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

"""Build a Flow-Factory JSONL dataset from the frozen Pick-a-Pic rewrites."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nonempty_lines(path: Path) -> Iterator[str]:
    """Yield newline-stripped, non-empty text rows."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = line.rstrip("\r\n")
            if value:
                yield value


def validate_source(source_dir: Path, baseline_test: Path) -> Dict[str, Any]:
    """Validate the frozen rewrite artifact and return its manifest."""
    manifest_path = source_dir / "manifest.json"
    records_path = source_dir / "records.jsonl"
    rewritten_path = source_dir / "train.txt"
    original_path = source_dir / "original_train.txt"
    required = [manifest_path, records_path, rewritten_path, original_path, baseline_test]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required rewrite inputs are missing: {missing}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("variant") != "balanced_v0":
        raise ValueError(f"expected balanced_v0 manifest, got {manifest.get('variant')!r}")
    expected_hashes = {
        rewritten_path: manifest["outputs"]["train_sha256"],
        original_path: manifest["outputs"]["original_train_sha256"],
        records_path: manifest["outputs"]["records_sha256"],
    }
    for path, expected in expected_hashes.items():
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")
    if manifest["sources"]["source_train_sha256"] != sha256(original_path):
        raise ValueError("original_train.txt does not match the source Pick-a-Pic split")
    return manifest


def build_dataset(source_dir: Path, baseline_test: Path, output_dir: Path) -> None:
    """Build the row-aligned conditioning/reward dataset atomically."""
    manifest = validate_source(source_dir, baseline_test)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        rewritten_rows = nonempty_lines(source_dir / "train.txt")
        original_rows = nonempty_lines(source_dir / "original_train.txt")
        records_path = source_dir / "records.jsonl"
        output_train = temporary / "train.jsonl"
        row_count = 0
        changed_count = 0
        skip_count = 0
        fallback_count = 0

        with records_path.open(encoding="utf-8") as records, output_train.open(
            "w", encoding="utf-8"
        ) as output:
            for row_count, (record_line, conditioning_prompt, original_prompt) in enumerate(
                zip(records, rewritten_rows, original_rows, strict=True), start=1
            ):
                record = json.loads(record_line)
                expected_index = row_count - 1
                if record["dataset_index"] != expected_index:
                    raise ValueError(
                        f"records.jsonl index mismatch at row {row_count}: "
                        f"expected {expected_index}, got {record['dataset_index']}"
                    )
                if record["conditioning_prompt"] != conditioning_prompt:
                    raise ValueError(f"conditioning prompt mismatch at row {row_count}")
                if record["original_prompt"] != original_prompt:
                    raise ValueError(f"reward prompt mismatch at row {row_count}")

                changed_count += int(record["changed"])
                skip_count += int(record["intentional_skip"])
                fallback_count += int(record["failed_fallback"])
                output_record = {
                    "prompt": conditioning_prompt,
                    "reward_prompt": original_prompt,
                    "rewrite_arm": record["arm"],
                    "rewrite_changed": record["changed"],
                    "rewrite_intentional_skip": record["intentional_skip"],
                    "rewrite_failed_fallback": record["failed_fallback"],
                    "rewrite_validation_reason": record["validation_reason"],
                    "rewrite_dataset_index": expected_index,
                }
                output.write(json.dumps(output_record, ensure_ascii=False) + "\n")

        expected_rows = manifest["rows"]
        counts = {
            "rows": row_count,
            "changed_rows": changed_count,
            "intentional_skips": skip_count,
            "failed_fallbacks": fallback_count,
        }
        expected_counts = {
            "rows": expected_rows,
            "changed_rows": manifest["changed_rows"],
            "intentional_skips": manifest["intentional_skips"],
            "failed_fallbacks": manifest["failed_fallbacks"],
        }
        if counts != expected_counts:
            raise ValueError(f"rewrite count mismatch: expected {expected_counts}, got {counts}")

        shutil.copy2(baseline_test, temporary / "test.txt")
        output_manifest = {
            "schema_version": 1,
            "variant": manifest["variant"],
            "arm": manifest["arm"],
            **counts,
            "conditioning_prompt_field": "prompt",
            "reward_prompt_field": "reward_prompt",
            "reward_prompt_contract": manifest["reward_prompt_contract"],
            "source_manifest_sha256": sha256(source_dir / "manifest.json"),
            "source_records_sha256": sha256(source_dir / "records.jsonl"),
            "source_conditioning_sha256": sha256(source_dir / "train.txt"),
            "source_reward_prompt_sha256": sha256(source_dir / "original_train.txt"),
            "baseline_test_sha256": sha256(baseline_test),
            "train_jsonl_sha256": sha256(output_train),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(output_manifest, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--baseline-test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run the dataset builder."""
    args = parse_args()
    build_dataset(
        args.source_dir.resolve(), args.baseline_test.resolve(), args.output_dir.resolve()
    )


if __name__ == "__main__":
    main()
