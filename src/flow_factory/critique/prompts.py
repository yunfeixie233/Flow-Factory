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

"""Provider-independent prompt recipes for T2I critique.

The system prompts and user-message builders are exact ports of the validated
AdvantageFlow critic prompts. Their wording was tuned over several ablation
rounds (anti-embellishment, exact-object-noun preservation, no spatial flips,
no negations) and is load-bearing for reproduction — do not edit a shipped
recipe in place; add a new recipe name (in code or via a ``prompts_yaml``
overlay file) for deliberate experiments.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional, Tuple

import yaml

# Minimal on-target rewrite for compositional (GenEval-style) rewards: copy the
# target prompt, make only the failed requirements explicit, add nothing else.
GENEVAL_REWRITE_SYSTEM = (
    "You are an expert prompt engineer for a text-to-image model. You are given a target prompt, the image the "
    "model produced for it, and a checklist that marks each of the target's requirements (each object, its exact "
    "count, each color, and each spatial relation) as MET or NOT MET on that image. Produce a caption by COPYING "
    "the target prompt and changing as little as possible: make ONLY the NOT-MET requirements explicit and "
    "unambiguous, and keep every already-MET part word-for-word as the target states it. Hard rules: describe the "
    "target, never the image's mistakes; never lower a count or drop an object; you may restate the failed "
    "requirement more firmly on-target (its exact color, count, or position). When the failed requirement is a "
    "SPATIAL relation, keep the SAME objects in the SAME order and the SAME direction — make it more explicit "
    "(e.g. 'directly below', 'to the left of') but NEVER invert it (do NOT turn 'A below B' into 'B above A'). Keep "
    "the EXACT object nouns and color words from the target — never swap an object name for a synonym (never "
    "'cat'->'feline', 'car'->'vehicle'). Use no negations (never 'no', 'not', 'without'). Add NOTHING ELSE — no extra "
    "objects or scenery (no skies, clouds, grass, tables, walls), no mood or lighting, and no decorative adjectives "
    "(no 'majestic', 'beautiful', 'pristine') — because any off-target detail makes the model break the "
    "requirements it was already getting right. Write counts as words, place each object explicitly, use plain "
    "fluent language (not a bullet list or edit commands), and keep the caption close to the target's length. Reply "
    "with only the caption."
)

# ``geneval_rewrite`` + the anti-hallucination guard — the winning arm
# ("C_antihalluc") of the GenEval critique-prompt ablation: acc_delta +0.091
# vs +0.068 unguarded (n=384, Gemini 2.5 Flash, reasoning=low), test-split
# confirmed.
GENEVAL_REWRITE_ANTIHAL_SYSTEM = GENEVAL_REWRITE_SYSTEM[: -len("Reply with only the caption.")] + (
    "Describe ONLY the objects named in the target prompt; never add, name, or describe an object you see in the "
    "image that the target did not ask for, and never describe the image's mistakes -- always state the target as "
    "it should be. Reply with only the caption."
)

# ``geneval_rewrite`` variant that additionally bans the cosmetic edits
# (a/an -> 'one', inserting 'directly'/'exactly') that empirically tied or
# reduced reward.
GENEVAL_REWRITE_NOCOSMETIC_SYSTEM = (
    "You are an expert prompt engineer for a text-to-image model. You are given a target prompt, the image the "
    "model produced for it, and a checklist marking the target requirements as MET or NOT MET. Return a minimally "
    "revised target caption that makes only a genuinely failed requirement clearer. Preserve every already-MET "
    "phrase word-for-word and preserve the target's articles, counts, object nouns, colors, modifiers, object "
    "order, and relation direction. Prior rewrites mostly made cosmetic edits such as replacing 'a' or 'an' with "
    "'one', or inserting 'directly' or 'exactly'; these edits overwhelmingly tied or reduced reward and must be "
    "avoided. Never replace 'a' or 'an' with 'one' (or otherwise rewrite an article), and never add 'directly' or "
    "'exactly' unless that exact word is already present in the target. Do not paraphrase, polish, normalize "
    "capitalization, or change wording merely for emphasis. Describe the target rather than the image's mistakes; "
    "never lower a count, drop an object, invert a relation, use a synonym, add an object or scenery, or introduce "
    "negation. If no non-cosmetic semantic correction to a NOT-MET requirement is possible while obeying these "
    "rules, copy the target exactly. Use plain fluent language and reply with only the caption."
)

# Aesthetic enrichment for preference rewards (HPS/PickScore-style): keep the
# subject, add vivid photographic detail.
DETAIL_REWRITE_SYSTEM = (
    "You are an expert prompt engineer for a text-to-image model. You are given a short target prompt and the "
    "image the model produced for it. Produce ONE improved caption that describes the SAME scene and subject as "
    "the target, but spelled out in vivid, concrete visual detail to make a more beautiful, higher-quality image. "
    "Keep every object and subject from the target (never drop or change what the scene is about); ADD specific, "
    "photographic detail: lighting (e.g. soft morning light, golden hour, studio lighting), composition and "
    "framing, materials and textures, color palette, atmosphere/mood, and tasteful quality cues (sharp focus, "
    "highly detailed, professional photography). Do not invent a different subject or contradict the target. Use "
    "fluent natural language (not a bullet list or edit commands), no negations. Reply with only the caption."
)


def _primary_score(axis_scores: Dict[str, float], preferred: Tuple[str, ...]) -> Optional[float]:
    """Pick the reward score a builder should mention, preferring known keys."""
    for key in preferred:
        if key in axis_scores:
            return float(axis_scores[key])
    if len(axis_scores) == 1:
        return float(next(iter(axis_scores.values())))
    return None


def build_geneval_user(
    prompt: str,
    axis_scores: Dict[str, float],
    clause_report: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a minimal-change compositional replacement request.

    When a per-clause detector report is available, present it as a plain
    MET / NOT-MET checklist; otherwise fall back to the aggregate score.

    Args:
        prompt: Original target prompt.
        axis_scores: Named round-1 reward scores.
        clause_report: Optional per-requirement detector report.

    Returns:
        Provider-neutral user-message text.
    """
    if clause_report and clause_report.get("items"):
        lines = []
        for desc, sc in clause_report["items"]:
            met = isinstance(sc, (int, float)) and sc >= 0.999
            lines.append(f"  - {desc}: {'MET' if met else 'NOT MET — fix this'}")
        reason = str(clause_report.get("reason") or "").strip()
        reason_line = f"\nWhat the detector saw: {reason}" if reason else ""
        return (
            f"Target prompt: {prompt}\n"
            f"Requirement checklist for the current image:\n"
            f"{chr(10).join(lines)}{reason_line}\n"
            f"Copy the target prompt and make the smallest change that states the NOT-MET requirement(s) above "
            f"explicitly; keep the MET parts word-for-word and add nothing else (no extra objects, scenery, mood, "
            f"or adjectives). Reply with only the caption."
        )
    score = _primary_score(axis_scores, ("geneval", "accuracy", "avg"))
    s = (
        f" (the detector says its compositional score is {score:.2f} out of 1.0, so it is missing or mis-rendering "
        f"at least one required element)"
        if isinstance(score, float)
        else ""
    )
    return (
        f"Target prompt: {prompt}\n"
        f"The attached image is the model's attempt{s}. Copy the target prompt and change as little as possible to "
        f"make every required object, exact count, color, and spatial relation explicit and unambiguous — describing "
        f"the target, not the current image, never reducing a count or dropping an object, and adding no extra "
        f"objects, scenery, mood, or decorative adjectives. Reply with only the caption."
    )


def build_detail_user(
    prompt: str,
    axis_scores: Dict[str, float],
    clause_report: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a same-subject aesthetic/detail replacement request.

    Args:
        prompt: Original target prompt.
        axis_scores: Named round-1 reward scores.
        clause_report: Unused optional detector report.

    Returns:
        Provider-neutral user-message text.
    """
    del clause_report
    score = _primary_score(axis_scores, ("hpsv2", "pick_score", "pickscore", "avg"))
    s = f" (its current human-preference score is {score:.3f})" if isinstance(score, float) else ""
    return (
        f"Target prompt: {prompt}\n"
        f"The attached image is the model's current attempt{s}. Rewrite the target into a more vivid, detailed, "
        f"aesthetically appealing caption of the SAME scene and subject to earn a higher-quality image. "
        f"Reply with only the caption."
    )


PromptBuilder = Callable[[str, Dict[str, float], Optional[Dict[str, Any]]], str]

_USER_BUILDERS: Dict[str, PromptBuilder] = {
    "geneval_rewrite": build_geneval_user,
    "detail_rewrite": build_detail_user,
}

_PROMPTS: Dict[str, Tuple[str, str]] = {
    # recipe name -> (system prompt, user_builder key)
    "geneval_rewrite": (GENEVAL_REWRITE_SYSTEM, "geneval_rewrite"),
    "geneval_rewrite_antihal": (GENEVAL_REWRITE_ANTIHAL_SYSTEM, "geneval_rewrite"),
    "geneval_rewrite_nocosmetic": (GENEVAL_REWRITE_NOCOSMETIC_SYSTEM, "geneval_rewrite"),
    "detail_rewrite": (DETAIL_REWRITE_SYSTEM, "detail_rewrite"),
}

_YAML_CACHE: Dict[str, Any] = {"path": None, "mtime": None, "recipes": {}}


def _yaml_recipes(path: Optional[str]) -> Dict[str, Dict[str, str]]:
    """Load recipes from a prompts YAML overlay, cached on file mtime.

    The file schema mirrors the AdvantageFlow ``config/prompts.yaml``::

        recipes:
          my_recipe:
            user_builder: geneval_rewrite   # or detail_rewrite
            system: <full system prompt>

    The file is re-read whenever its mtime changes, so edits apply to a live
    run's next critique batch without a restart. A recipe named after a
    built-in mode overrides it.
    """
    if not path:
        return {}
    mtime = os.path.getmtime(path)
    if _YAML_CACHE["path"] != path or _YAML_CACHE["mtime"] != mtime:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        recipes = data.get("recipes") or {}
        for name, recipe in recipes.items():
            if not isinstance(recipe, dict) or not recipe.get("system"):
                raise ValueError(
                    f"prompts_yaml recipe {name!r} must be a dict with a 'system' string"
                )
            builder = recipe.get("user_builder", "geneval_rewrite")
            if builder not in _USER_BUILDERS:
                raise ValueError(
                    f"prompts_yaml recipe {name!r} names unknown user_builder {builder!r} "
                    f"(available: {tuple(_USER_BUILDERS)})"
                )
        _YAML_CACHE.update(path=path, mtime=mtime, recipes=recipes)
    return _YAML_CACHE["recipes"]


def get_critique_prompt(
    mode: str,
    system_override: Optional[str] = None,
    prompts_yaml: Optional[str] = None,
) -> Tuple[str, PromptBuilder]:
    """Resolve ``(system_prompt, user_builder)`` for a critique recipe.

    Args:
        mode: Recipe name — a built-in mode or a ``prompts_yaml`` recipe name.
        system_override: Optional complete system-prompt replacement.
        prompts_yaml: Optional overlay YAML path; its recipes are resolved
            first and hot-reload on file modification.

    Returns:
        Tuple of the system prompt and the user-message builder.
    """
    recipes = _yaml_recipes(prompts_yaml)
    if mode in recipes:
        recipe = recipes[mode]
        system = recipe["system"]
        builder = _USER_BUILDERS[recipe.get("user_builder", "geneval_rewrite")]
    elif mode in _PROMPTS:
        system, builder_key = _PROMPTS[mode]
        builder = _USER_BUILDERS[builder_key]
    else:
        available = sorted(set(_PROMPTS) | set(recipes))
        raise ValueError(f"Unknown critique mode {mode!r}; available: {available}")
    return system_override or system, builder
