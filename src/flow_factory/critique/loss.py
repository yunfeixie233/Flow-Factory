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

from typing import Optional, Sequence, Tuple

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


def gradient_norm_ratio(
    numerator_loss: torch.Tensor,
    denominator_loss: torch.Tensor,
    parameters: Sequence[torch.Tensor],
    probe_scale: float = 16384.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Measure ``||d numerator/d params|| / ||d denominator/d params||``.

    Both losses are scaled by ``probe_scale`` before differentiation so half
    precision gradients do not underflow; the ratio is scale-invariant. The
    computation uses ``torch.autograd.grad`` with ``retain_graph=True`` and
    does not touch ``param.grad``, so the caller's subsequent backward pass
    over the combined loss is unaffected.
    """

    parameters = [p for p in parameters if p.requires_grad]
    if not parameters:
        raise ValueError("gradient_norm_ratio requires at least one trainable parameter")

    def _norm(loss: torch.Tensor) -> torch.Tensor:
        grads = torch.autograd.grad(
            loss * probe_scale,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        total = None
        for grad in grads:
            if grad is None:
                continue
            value = grad.float().square().sum()
            total = value if total is None else total + value
        if total is None:
            return torch.zeros((), device=loss.device)
        return total.sqrt()

    numerator = _norm(numerator_loss)
    denominator = _norm(denominator_loss)
    return (numerator / (denominator + eps)).detach()


def ppd_same_state_distillation_loss(
    student_velocity: torch.Tensor,
    teacher_base_velocity: torch.Tensor,
    teacher_rewrite_velocity: torch.Tensor,
    kappa: float,
    active: torch.Tensor,
    sigma: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-row privileged-prompt CFG-distillation loss at one shared state.

    The detached teacher target is the bounded CFG interpolation

        v_tgt = sg[v_old(x_t, c) + kappa * (v_old(x_t, c') - v_old(x_t, c))],

    where both teacher velocities were predicted by the lagged sampling policy
    at exactly the state the student is trained on.  There is no reward gate
    and no signed coefficient: rows are weighted only by the ``active`` mask
    and, when ``sigma`` is supplied, by ``sigma^2`` (matching the implicit
    ``t^2`` factor of the clean-prediction parameterization).

    Args:
        student_velocity: Current-policy velocity under the original condition.
        teacher_base_velocity: Lagged-policy velocity under the original condition.
        teacher_rewrite_velocity: Lagged-policy velocity under the privileged condition.
        kappa: CFG interpolation strength in [0, 1].
        active: Per-row nonnegative activity mask (0 disables a row).
        sigma: Optional per-row normalized flow-matching noise level.

    Returns:
        Per-row weighted loss and unweighted per-row distillation MSE.
    """

    if student_velocity.shape != teacher_base_velocity.shape or (
        student_velocity.shape != teacher_rewrite_velocity.shape
    ):
        raise ValueError(
            "student, teacher base, and teacher rewrite velocities must have identical shapes; "
            f"got {student_velocity.shape}, {teacher_base_velocity.shape}, "
            f"and {teacher_rewrite_velocity.shape}"
        )
    if student_velocity.ndim < 2:
        raise ValueError("velocity tensors must contain batch and feature dimensions")
    if not 0.0 <= float(kappa) <= 1.0:
        raise ValueError(f"kappa must lie in [0, 1], got {kappa}")
    batch_size = student_velocity.shape[0]
    if active.ndim != 1 or active.shape[0] != batch_size:
        raise ValueError(f"active must have shape [{batch_size}], got {active.shape}")
    if torch.any(active < 0):
        raise ValueError("active mask must be nonnegative")
    if sigma is not None and (sigma.ndim != 1 or sigma.shape[0] != batch_size):
        raise ValueError(f"sigma must have shape [{batch_size}], got {sigma.shape}")

    base = teacher_base_velocity.detach()
    rewrite = teacher_rewrite_velocity.detach()
    target = (base + float(kappa) * (rewrite - base)).detach()

    reduce_dims = tuple(range(1, student_velocity.ndim))
    mse = ((student_velocity - target) ** 2).mean(dim=reduce_dims)
    rows = active.to(device=mse.device, dtype=mse.dtype) * mse
    if sigma is not None:
        rows = rows * sigma.to(device=mse.device, dtype=mse.dtype).square()
    return rows, mse
