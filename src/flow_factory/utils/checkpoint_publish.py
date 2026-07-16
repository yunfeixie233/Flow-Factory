"""Completion markers, retention, and direct checkpoint publication."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import uuid
from typing import Dict, List, Optional, Tuple


CHECKPOINT_DIR_PATTERN = re.compile(r"^checkpoint-(\d+)$")
CHECKPOINT_COMPLETE_MARKER = "_COMPLETE"


def write_checkpoint_complete_marker(
    checkpoint_dir: str,
    metadata: Optional[Dict] = None,
) -> str:
    """Atomically mark a local checkpoint as complete."""
    checkpoint_dir = os.path.abspath(os.path.expanduser(checkpoint_dir))
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(checkpoint_dir)

    marker = os.path.join(checkpoint_dir, CHECKPOINT_COMPLETE_MARKER)
    temporary = f"{marker}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    payload = dict(metadata or {})
    payload.setdefault("checkpoint", os.path.basename(checkpoint_dir))
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, marker)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return marker


def publish_checkpoint_to_s3(
    source_checkpoint: str,
    destination_root: str,
    *,
    upload_tool: str = "s5cmd",
    concurrency: int = 32,
) -> str:
    """Upload a completed local checkpoint directly to S3.

    Data files are uploaded first and ``_COMPLETE`` is uploaded last. Consumers
    must ignore an S3 checkpoint prefix until that marker exists.
    """
    source_checkpoint = os.path.abspath(os.path.expanduser(source_checkpoint))
    checkpoint_name = os.path.basename(source_checkpoint)
    if not CHECKPOINT_DIR_PATTERN.fullmatch(checkpoint_name):
        raise ValueError(
            f"checkpoint directory must be named checkpoint-N, got {checkpoint_name!r}"
        )
    if not os.path.isdir(source_checkpoint):
        raise FileNotFoundError(source_checkpoint)

    marker = os.path.join(source_checkpoint, CHECKPOINT_COMPLETE_MARKER)
    if not os.path.isfile(marker):
        raise FileNotFoundError(
            f"refusing to upload incomplete checkpoint without {CHECKPOINT_COMPLETE_MARKER}: "
            f"{source_checkpoint}"
        )
    if not destination_root.startswith("s3://"):
        raise ValueError(f"S3 destination must start with s3://, got {destination_root!r}")
    if concurrency <= 0:
        raise ValueError(f"upload concurrency must be positive, got {concurrency}")

    destination = f"{destination_root.rstrip('/')}/{checkpoint_name}"
    if upload_tool != "s5cmd":
        raise ValueError(
            f"unsupported checkpoint upload tool {upload_tool!r}; expected 's5cmd'"
        )
    if shutil.which("s5cmd") is None:
        raise FileNotFoundError("s5cmd is required for checkpoint publication")

    # S3 checkpoints are immutable. --no-clobber makes retries resume a partial
    # marker-less upload without changing objects under an already-published
    # prefix. The marker remains the final publication boundary.
    subprocess.run(
        [
            "s5cmd",
            "cp",
            "--no-clobber",
            "--concurrency",
            str(concurrency),
            "--exclude",
            CHECKPOINT_COMPLETE_MARKER,
            source_checkpoint + os.sep,
            destination + "/",
        ],
        check=True,
    )
    subprocess.run(
        [
            "s5cmd",
            "cp",
            "--no-clobber",
            marker,
            f"{destination}/{CHECKPOINT_COMPLETE_MARKER}",
        ],
        check=True,
    )
    return destination


def prune_checkpoint_directories(checkpoint_root: str, keep: int) -> List[str]:
    """Delete older completed ``checkpoint-N`` directories."""
    if keep < 0:
        raise ValueError(f"checkpoint retention must be non-negative, got {keep}")
    if keep == 0:
        return []

    checkpoint_root = os.path.abspath(os.path.expanduser(checkpoint_root))
    if not os.path.isdir(checkpoint_root):
        return []

    candidates: List[Tuple[int, str]] = []
    for name in os.listdir(checkpoint_root):
        match = CHECKPOINT_DIR_PATTERN.fullmatch(name)
        path = os.path.join(checkpoint_root, name)
        if match is None or not os.path.isdir(path):
            continue
        if not os.path.isfile(os.path.join(path, CHECKPOINT_COMPLETE_MARKER)):
            continue
        candidates.append((int(match.group(1)), path))

    candidates.sort(key=lambda item: (item[0], item[1]))
    removed: List[str] = []
    for _, path in candidates[:-keep]:
        shutil.rmtree(path)
        removed.append(path)
    return removed
