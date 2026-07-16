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

"""Deterministic semantic guards for critique replacement captions."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple

_WORD_RE = re.compile(r"[a-z]+")
_COUNT_MARKERS = {
    "a",
    "an",
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "single",
    "exactly",
}
_COLORS = {"red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"}
_RELATIONS = {"left", "right", "above", "below", "over", "under", "front", "behind"}
_FORBIDDEN_NEW_WORDS = {"directly", "no", "not", "without"}
_GENEVAL_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "computer mouse",
    "tv remote",
    "computer keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


def _tokens(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def _token_matches_class(token: str, class_token: str) -> bool:
    return token == class_token or token.startswith(class_token)


def _class_positions(tokens: List[str], classname: str) -> List[int]:
    parts = classname.split()
    return [
        start
        for start in range(len(tokens) - len(parts) + 1)
        if all(
            _token_matches_class(tokens[start + offset], part) for offset, part in enumerate(parts)
        )
    ]


def _signature(tokens: List[str], vocabulary: Set[str]) -> List[str]:
    return [token for token in tokens if token in vocabulary]


def validate_geneval_rewrite(
    original: str, rewrite: str, metadata: Dict[str, Any]
) -> Tuple[bool, str]:
    """Reject rewrites that alter immutable GenEval semantics.

    Args:
        original: Original GenEval target caption.
        rewrite: Candidate replacement caption.
        metadata: GenEval metadata containing required objects and colors.

    Returns:
        ``(is_valid, reason)``.
    """

    original = str(original or "").strip()
    rewrite = str(rewrite or "").strip()
    if not rewrite:
        return False, "empty_rewrite"
    if rewrite != rewrite.lower() and original == original.lower():
        return False, "capitalization_changed"

    original_tokens = _tokens(original)
    rewrite_tokens = _tokens(rewrite)
    if _signature(rewrite_tokens, _COUNT_MARKERS) != _signature(original_tokens, _COUNT_MARKERS):
        return False, "count_or_article_changed"
    if _signature(rewrite_tokens, _COLORS) != _signature(original_tokens, _COLORS):
        return False, "color_changed"
    if _signature(rewrite_tokens, _RELATIONS) != _signature(original_tokens, _RELATIONS):
        return False, "relation_changed"
    for word in _FORBIDDEN_NEW_WORDS:
        if word in rewrite_tokens and word not in original_tokens:
            return False, f"forbidden_word_added:{word}"

    target_classes = [str(req["class"]).lower() for req in (metadata or {}).get("include", [])]
    original_order = []
    rewrite_order = []
    for classname in target_classes:
        original_positions = _class_positions(original_tokens, classname)
        rewrite_positions = _class_positions(rewrite_tokens, classname)
        if not original_positions or not rewrite_positions:
            return False, f"target_object_missing_or_renamed:{classname}"
        original_order.append((original_positions[0], classname))
        rewrite_order.append((rewrite_positions[0], classname))
    if [name for _, name in sorted(rewrite_order)] != [name for _, name in sorted(original_order)]:
        return False, "target_object_order_changed"

    required_colors = {
        str(req.get("color", "")).lower() for req in (metadata or {}).get("include", [])
    }
    target_class_tokens = [set(name.split()) for name in target_classes]
    for candidate in _GENEVAL_CLASSES:
        candidate_tokens = set(candidate.split())
        if any(
            candidate_tokens <= target or target <= candidate_tokens
            for target in target_class_tokens
        ):
            continue
        if candidate in required_colors:
            continue
        if _class_positions(rewrite_tokens, candidate) and not _class_positions(
            original_tokens, candidate
        ):
            return False, f"extra_object_added:{candidate}"

    return True, "ok"
