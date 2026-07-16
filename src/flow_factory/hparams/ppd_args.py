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

"""Configuration for privileged-prompt distillation (PPD).

PPD attaches a records-based privileged conditioning prompt to every training
row and adds a small same-state CFG-distillation auxiliary loss on top of the
native objective.  Rollout generation and all reward scoring stay on the
original prompt; the privileged prompt is visible to the loss only.  The
matched control arm executes identical plumbing with ``rho=0``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from .abc import ArgABC


@dataclass
class PPDArguments(ArgABC):
    """Arguments for the privileged-prompt distillation auxiliary loss.

    The component is disabled by default.  Enabling it does not select a
    training loss by itself: trainers opt in explicitly.  DiffusionNFT is the
    first consumer.
    """

    enabled: bool = field(
        default=False,
        metadata={"help": "Enable records-based privileged-prompt distillation."},
    )
    records_path: str = field(
        default="",
        metadata={
            "help": (
                "JSONL file with one record per line: {schema_version, dataset_index, "
                "original_prompt, conditioning_prompt, changed}. Rows are joined to "
                "training samples by exact original-prompt text."
            )
        },
    )
    rho: float = field(
        default=0.0,
        metadata={
            "help": (
                "Auxiliary loss coefficient. The matched control arm pins rho=0 and "
                "must report ppd/control_zero == 0 at every step."
            )
        },
    )
    kappa: float = field(
        default=1.0,
        metadata={
            "help": (
                "CFG interpolation strength in [0, 1] for the teacher target "
                "v_old(x_t,c) + kappa * (v_old(x_t,c') - v_old(x_t,c))."
            )
        },
    )
    timestep_weighted: bool = field(
        default=True,
        metadata={"help": "Weight per-row distillation MSE by sigma_t^2."},
    )
    mask_identity: bool = field(
        default=True,
        metadata={
            "help": "Zero the auxiliary loss on rows whose privileged prompt equals the original."
        },
    )
    require_records_coverage: bool = field(
        default=True,
        metadata={
            "help": (
                "Raise when a training prompt has no record. When false, uncovered rows "
                "fall back to the original prompt and are masked inactive."
            )
        },
    )
    prompt_encoding_device: Literal["accelerator", "cpu"] = field(
        default="accelerator",
        metadata={"help": "Temporary device used to encode privileged prompts."},
    )

    def __post_init__(self) -> None:
        self.rho = float(self.rho)
        self.kappa = float(self.kappa)
        if not math.isfinite(self.rho) or self.rho < 0.0:
            raise ValueError(f"ppd.rho must be finite and nonnegative, got {self.rho}")
        if not 0.0 <= self.kappa <= 1.0:
            raise ValueError(f"ppd.kappa must lie in [0, 1], got {self.kappa}")
        if self.enabled and not str(self.records_path):
            raise ValueError("ppd.records_path is required when PPD is enabled")
