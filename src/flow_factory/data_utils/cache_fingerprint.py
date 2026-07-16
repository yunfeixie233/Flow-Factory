"""Lightweight helpers for dataset-cache invalidation."""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache


@lru_cache(maxsize=128)
def _hash_file(path: str, size: int, mtime_ns: int, ctime_ns: int) -> str:
    """Hash *path*; stat fields are cache-key inputs for in-process reuse."""
    del size, mtime_ns, ctime_ns
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_source_fingerprint(dataset_dir: str, split: str) -> str:
    """Return a stable identity for the JSONL/TXT file backing a split."""
    root = os.path.abspath(os.path.expanduser(dataset_dir))
    candidates = (
        os.path.join(root, f"{split}.jsonl"),
        os.path.join(root, f"{split}.txt"),
    )
    source = next((path for path in candidates if os.path.isfile(path)), None)
    if source is None:
        raise FileNotFoundError(
            f"could not fingerprint dataset split {split!r}; expected one of: "
            + ", ".join(candidates)
        )
    stat = os.stat(source)
    digest = _hash_file(source, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    return f"{source}|{stat.st_size}|sha256:{digest}"
