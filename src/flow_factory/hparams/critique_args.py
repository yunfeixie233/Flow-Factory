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

"""Configuration for the optional T2I critique/refinement component."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

from .abc import ArgABC


def _standardize_clip_range(value) -> Tuple[float, float]:
    if not isinstance(value, (tuple, list)):
        bound = abs(float(value))
        return (-bound, bound)
    lower, upper = float(value[0]), float(value[1])
    if lower >= upper:
        raise ValueError("critique.advantage_clip_range lower bound must be less than upper bound")
    return lower, upper


@dataclass
class CritiqueArguments(ArgABC):
    """Arguments shared by critique-capable T2I trainers.

    The component is disabled by default.  Enabling it does not select a
    training loss by itself: trainers opt in explicitly.  DiffusionNFT is the
    first consumer.
    """

    enabled: bool = field(
        default=False,
        metadata={"help": "Enable paired T2I critique/refinement rollouts."},
    )
    backend: str = field(
        default="openai-compatible",
        metadata={
            "help": (
                "Critique backend name, or a custom 'module.path:ClassName'. "
                "Custom classes receive this CritiqueArguments instance."
            )
        },
    )
    model: Optional[str] = field(
        default=None,
        metadata={"help": "Model name sent to the critique backend."},
    )
    base_url: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "OpenAI-compatible API base URL or full /chat/completions URL. "
                "Required by the built-in backend."
            )
        },
    )
    api_key_env: str = field(
        default="GEMINI_API_KEY",
        metadata={"help": "Environment variable containing the critique API key."},
    )
    mode: str = field(
        default="geneval_rewrite",
        metadata={
            "help": (
                "Critique prompt recipe. Built-ins: geneval_rewrite, geneval_rewrite_antihal, "
                "geneval_rewrite_nocosmetic, detail_rewrite. A prompts_yaml overlay may add recipes."
            )
        },
    )
    prompts_yaml: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional prompt-recipe overlay YAML (schema: recipes.<name>.{system, user_builder}). "
                "Hot-reloaded on file modification; a recipe named after a built-in mode overrides it."
            )
        },
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "Optional complete system-prompt override."},
    )
    reward_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Reward used for round-2 improvement. May be omitted only when "
                "the training configuration has exactly one reward."
            )
        },
    )
    validator: Literal["none", "geneval"] = field(
        default="geneval",
        metadata={"help": "Deterministic semantic guard for replacement captions."},
    )
    advantage_mode: Literal["signed", "nonnegative"] = field(
        default="nonnegative",
        metadata={
            "help": (
                "Whether negative round-2 improvement advantages are retained. 'nonnegative' "
                "(default) clips the critique advantage to [0, upper]; 'signed' keeps repulsive "
                "negative rows and should be paired with a small critique_loss_weight."
            )
        },
    )
    advantage_clip_range: Tuple[float, float] = field(
        default=(-1.0, 1.0),
        metadata={"help": "Clip range for the round-2 critique advantage."},
    )
    max_tokens: int = field(default=128, metadata={"help": "Maximum critique output tokens."})
    temperature: float = field(default=0.0, metadata={"help": "Critique decoding temperature."})
    timeout: float = field(default=90.0, metadata={"help": "Per-request HTTP timeout in seconds."})
    max_retries: int = field(default=2, metadata={"help": "Retries after a failed API request."})
    retry_backoff: float = field(default=1.0, metadata={"help": "Base exponential retry delay."})
    num_workers: int = field(default=8, metadata={"help": "Concurrent critique API rows per rank."})
    image_format: Literal["png", "jpeg", "webp"] = field(
        default="jpeg",
        metadata={"help": "Image format embedded in multimodal API requests."},
    )
    image_quality: int = field(default=90, metadata={"help": "JPEG/WebP quality in [1, 100]."})
    reasoning_effort: Optional[str] = field(
        default="none",
        metadata={"help": "Optional OpenAI-compatible reasoning_effort request field."},
    )
    prompt_encoding_device: Literal["accelerator", "cpu"] = field(
        default="accelerator",
        metadata={"help": "Temporary device used to encode rewritten prompts."},
    )

    def __post_init__(self) -> None:
        self.advantage_clip_range = _standardize_clip_range(self.advantage_clip_range)
        if self.advantage_mode == "nonnegative" and self.advantage_clip_range[0] < 0:
            self.advantage_clip_range = (0.0, self.advantage_clip_range[1])
        if self.max_tokens <= 0:
            raise ValueError("critique.max_tokens must be positive")
        if self.timeout <= 0:
            raise ValueError("critique.timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("critique.max_retries must be nonnegative")
        if self.retry_backoff < 0:
            raise ValueError("critique.retry_backoff must be nonnegative")
        if self.num_workers <= 0:
            raise ValueError("critique.num_workers must be positive")
        if not 1 <= self.image_quality <= 100:
            raise ValueError("critique.image_quality must be in [1, 100]")
        if self.enabled and self.prompts_yaml and not os.path.isfile(self.prompts_yaml):
            raise ValueError(f"critique.prompts_yaml file does not exist: {self.prompts_yaml!r}")
        if self.enabled and self.backend == "openai-compatible":
            if not self.base_url:
                raise ValueError("critique.base_url is required for the openai-compatible backend")
            if not self.model:
                raise ValueError("critique.model is required for the openai-compatible backend")
