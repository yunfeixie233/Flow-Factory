#!/usr/bin/env python3
"""Build a row-preserving no-rewrite ablation from a Flow-Factory dataset.

The source dataset is the rewritten intrinsic-knowledge treatment. Its
``train.jsonl`` already has the Arrow-safe GenEval metadata needed by
Flow-Factory and retains the unmodified text in ``original_prompt``. This
builder changes only the model-input ``prompt`` field back to that original
text. It never filters, samples, sorts, or deduplicates rows. The evaluation
split is copied byte-for-byte so treatment and control use identical examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterator


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank rows are not allowed")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, record


def atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def build_train(source: Path, destination: Path) -> dict[str, int]:
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    rows = 0
    rows_changed_from_treatment = 0
    unique_prompts: set[str] = set()

    try:
        with temporary.open("w", encoding="utf-8") as output:
            for line_number, record in iter_jsonl(source):
                treatment_prompt = record.get("prompt")
                original_prompt = record.get("original_prompt")
                if not isinstance(treatment_prompt, str) or not treatment_prompt.strip():
                    raise ValueError(f"{source}:{line_number}: missing non-empty prompt")
                if not isinstance(original_prompt, str) or not original_prompt.strip():
                    raise ValueError(
                        f"{source}:{line_number}: missing non-empty original_prompt"
                    )

                if treatment_prompt != original_prompt:
                    rows_changed_from_treatment += 1
                record["prompt"] = original_prompt
                unique_prompts.add(original_prompt)
                output.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                rows += 1
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "train_rows": rows,
        "train_unique_prompts": len(unique_prompts),
        "train_duplicate_rows_retained": rows - len(unique_prompts),
        "rows_changed_from_treatment": rows_changed_from_treatment,
    }


def validate_train(source: Path, output: Path) -> None:
    source_rows = iter_jsonl(source)
    output_rows = iter_jsonl(output)
    compared = 0

    while True:
        try:
            source_item = next(source_rows)
        except StopIteration:
            source_item = None
        try:
            output_item = next(output_rows)
        except StopIteration:
            output_item = None

        if source_item is None or output_item is None:
            if source_item != output_item:
                raise ValueError("source and output row counts differ")
            break

        source_line, source_record = source_item
        output_line, output_record = output_item
        if source_line != output_line:
            raise AssertionError("row order changed unexpectedly")

        expected_prompt = source_record["original_prompt"]
        if output_record.get("prompt") != expected_prompt:
            raise ValueError(f"output row {output_line} does not use original_prompt")

        source_without_prompt = dict(source_record)
        output_without_prompt = dict(output_record)
        source_without_prompt.pop("prompt", None)
        output_without_prompt.pop("prompt", None)
        if output_without_prompt != source_without_prompt:
            raise ValueError(f"metadata changed at output row {output_line}")
        compared += 1

    if compared == 0:
        raise ValueError("training dataset is empty")


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Rewritten Flow-Factory dataset containing train.jsonl and test.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination for the row-preserving no-rewrite dataset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if source_dir == output_dir:
        raise ValueError("source and output directories must be different")

    source_train = source_dir / "train.jsonl"
    source_test = source_dir / "test.jsonl"
    for required in (source_train, source_test):
        if not required.is_file():
            raise FileNotFoundError(required)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_train = output_dir / "train.jsonl"
    output_test = output_dir / "test.jsonl"

    counts = build_train(source_train, output_train)
    atomic_copy(source_test, output_test)
    validate_train(source_train, output_train)

    test_rows = sum(1 for _ in iter_jsonl(output_test))
    if sha256(source_test) != sha256(output_test):
        raise ValueError("evaluation split is not byte-identical to the treatment")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "variant": "no_rewrite",
        "deduplicated": False,
        "source_dataset": str(source_dir),
        "source_train_sha256": sha256(source_train),
        "source_test_sha256": sha256(source_test),
        "output_train_sha256": sha256(output_train),
        "output_test_sha256": sha256(output_test),
        "test_rows": test_rows,
        **counts,
    }
    write_manifest(output_dir / "manifest.json", manifest)

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
