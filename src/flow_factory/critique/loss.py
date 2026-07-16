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

"""Small testable loss primitives for critique-guided T2I training."""

from __future__ import annotations

from typing import Tuple

import torch


def critique_direction_loss(
    student_velocity: torch.Tensor,
    rewrite_velocity: torch.Tensor,
    advantage: torch.Tensor,
    sigma: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the auxiliary paired-rewrite direction loss.

    Args:
        student_velocity: Current-policy velocity under the original condition.
        rewrite_velocity: Stop-gradient online target under the rewritten condition.
        advantage: Per-row critique improvement advantage.
        sigma: Per-row normalized flow-matching noise level.

    Returns:
        Per-row weighted loss and unweighted per-row direction MSE.
    """

    if student_velocity.shape != rewrite_velocity.shape:
        raise ValueError(
            "student_velocity and rewrite_velocity must have identical shapes; "
            f"got {student_velocity.shape} and {rewrite_velocity.shape}"
        )
    if student_velocity.ndim < 2:
        raise ValueError("velocity tensors must contain batch and feature dimensions")
    batch_size = student_velocity.shape[0]
    if advantage.ndim != 1 or advantage.shape[0] != batch_size:
        raise ValueError(f"advantage must have shape [{batch_size}], got {advantage.shape}")
    if sigma.ndim != 1 or sigma.shape[0] != batch_size:
        raise ValueError(f"sigma must have shape [{batch_size}], got {sigma.shape}")

    reduce_dims = tuple(range(1, student_velocity.ndim))
    direction_mse = ((student_velocity - rewrite_velocity.detach()) ** 2).mean(dim=reduce_dims)
    sigma_sq = sigma.to(device=student_velocity.device, dtype=student_velocity.dtype).square()
    rows = (
        advantage.to(device=student_velocity.device, dtype=student_velocity.dtype)
        * sigma_sq
        * direction_mse
    )
    return rows, direction_mse

