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

"""Shared paired-rollout orchestration for critique-capable T2I trainers."""

from __future__ import annotations

import json
from collections import OrderedDict
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..data_utils.dataset import METADATA_COLUMN
from ..hparams import CritiqueArguments
from ..rewards import REWARD_METADATA_KEY
from ..samples import BaseSample, I2ISample, T2ISample
from ..utils.base import filter_kwargs
from ..utils.image import standardize_image_batch
from ..utils.logger_utils import setup_logger
from .abc import BaseCritiqueBackend, CritiqueRequest, CritiqueResult
from .registry import load_critique_backend
from .validators import validate_geneval_rewrite

if TYPE_CHECKING:
    from ..trainers.abc import BaseTrainer


logger = setup_logger(__name__)


def group_normalized_improvement(
    round1_rewards: np.ndarray,
    round2_rewards: np.ndarray,
    group_indices: np.ndarray,
    global_std: float,
    clip_range: Tuple[float, float],
    mode: str,
) -> np.ndarray:
    """Compute ``(r2 - mean_group(r1)) / std_global(r1)``.

    This is pure NumPy so the statistical contract is independently testable;
    distributed collection/local slicing remains the processor's responsibility.

    Args:
        round1_rewards: Round-1 rewards in collected sample order.
        round2_rewards: Round-2 rewards in collected sample order.
        group_indices: Dense prompt-group index for each collected row.
        global_std: Global round-1 reward standard deviation.
        clip_range: Inclusive lower/upper advantage bounds.
        mode: ``signed`` or ``nonnegative``.

    Returns:
        Clipped critique advantages in collected sample order.
    """

    round1_rewards = np.asarray(round1_rewards, dtype=np.float64).reshape(-1)
    round2_rewards = np.asarray(round2_rewards, dtype=np.float64).reshape(-1)
    group_indices = np.asarray(group_indices, dtype=np.int64).reshape(-1)
    if not (len(round1_rewards) == len(round2_rewards) == len(group_indices)):
        raise ValueError("round1, round2, and group_indices must have the same length")
    if not np.isfinite(round1_rewards).all() or not np.isfinite(round2_rewards).all():
        raise ValueError("Critique rewards must be finite for every refined sample")
    if global_std <= 0 or not np.isfinite(global_std):
        raise ValueError(f"global_std must be finite and positive, got {global_std}")

    group_count = int(group_indices.max()) + 1 if len(group_indices) else 0
    sums = np.bincount(group_indices, weights=round1_rewards, minlength=group_count)
    counts = np.bincount(group_indices, minlength=group_count)
    if np.any(counts == 0):
        raise ValueError("group_indices must be contiguous from zero")
    group_means = sums / counts
    advantage = (round2_rewards - group_means[group_indices]) / global_std
    lower, upper = clip_range
    if mode == "nonnegative":
        lower = max(0.0, lower)
    elif mode != "signed":
        raise ValueError(f"Unknown critique advantage mode {mode!r}")
    return np.clip(advantage, lower, upper).astype(np.float32)


class CritiqueProcessor:
    """Backend-neutral paired T2I critique service.

    The processor owns critique requests, semantic validation, prompt
    re-encoding, same-seed round-2 rollout, original-prompt reward evaluation,
    and critique-advantage attachment.  It does not own an optimizer loss;
    trainers decide whether and how to consume the attached pair.
    """

    _CONDITIONING_EXCLUDE = {
        "t",
        "t_next",
        "latents",
        "next_latents",
        "timesteps",
        "all_latents",
        "log_probs",
        "image",
        "video",
        "audio",
        "advantage",
        "rewards",
        "critique",
    }

    def __init__(
        self,
        config: CritiqueArguments,
        backend: Optional[BaseCritiqueBackend] = None,
    ) -> None:
        self.config = config
        self.backend = backend or load_critique_backend(config)

    @staticmethod
    def _metadata(sample: BaseSample) -> Dict[str, Any]:
        metadata = sample.extra_kwargs.get(METADATA_COLUMN, {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                return {}
        return metadata if isinstance(metadata, dict) else {}

    def _reward_name(self, rewards: Dict[str, torch.Tensor]) -> str:
        if self.config.reward_name:
            if self.config.reward_name not in rewards:
                raise KeyError(
                    f"critique.reward_name={self.config.reward_name!r} is not among "
                    f"the training rewards {sorted(rewards)}"
                )
            return self.config.reward_name
        if len(rewards) != 1:
            raise ValueError(
                "critique.reward_name is required when more than one training reward is configured; "
                f"available rewards: {sorted(rewards)}"
            )
        return next(iter(rewards))

    @staticmethod
    def _has_condition(value: Any) -> bool:
        """Return whether an optional conditioning field carries content."""
        if value is None:
            return False
        if isinstance(value, (list, tuple, dict, set)):
            return bool(value)
        return True

    @classmethod
    def _is_t2i_sample(cls, sample: BaseSample) -> bool:
        """Recognize typed T2I outputs and unconditioned dual-mode I2I outputs.

        Flux2 adapters expose one ``I2ISample`` subclass for both T2I and I2I.
        Such a sample represents T2I only when no input-image conditioning (or
        its encoded latent) is present. True edit/I2I rows remain unsupported.
        """
        if sample.video is not None or sample.audio is not None:
            return False
        if isinstance(sample, T2ISample):
            return True
        if not isinstance(sample, I2ISample):
            return False
        return not any(
            cls._has_condition(getattr(sample, field, None))
            for field in ("condition_images", "condition_videos", "image_latents")
        )

    @staticmethod
    def _reward_clause_report(
        sample: BaseSample,
        reward_name: str,
        dataset_metadata: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Resolve the selected reward's row-level scorecard for the critic."""
        by_reward = sample.extra_kwargs.get(REWARD_METADATA_KEY, {})
        reward_metadata = by_reward.get(reward_name, {}) if isinstance(by_reward, dict) else {}
        if isinstance(reward_metadata, dict):
            materialized = reward_metadata.get("clause_report")
            if isinstance(materialized, dict):
                return materialized
            breakdown = reward_metadata.get("breakdown")
            if isinstance(breakdown, (list, tuple)):
                return {
                    "items": list(breakdown),
                    "reason": str(reward_metadata.get("reason") or ""),
                }

        # Preserve support for custom datasets that already provide a report.
        fallback = dataset_metadata.get("clause_report")
        return fallback if isinstance(fallback, dict) else None

    def _requests(
        self,
        samples: Sequence[BaseSample],
        rewards: Dict[str, torch.Tensor],
        reward_name: str,
    ) -> List[CritiqueRequest]:
        requests: List[CritiqueRequest] = []
        for index, sample in enumerate(samples):
            metadata = self._metadata(sample)
            axis_scores = {
                name: float(torch.as_tensor(values[index]).item())
                for name, values in rewards.items()
                if torch.isfinite(torch.as_tensor(values[index])).item()
            }
            clause_report = self._reward_clause_report(sample, reward_name, metadata)
            image = standardize_image_batch(sample.image, output_type="pil")[0]
            requests.append(
                CritiqueRequest(
                    image=image,
                    prompt=str(sample.prompt or ""),
                    axis_scores=axis_scores,
                    metadata=metadata,
                    clause_report=clause_report,
                )
            )
        return requests

    def submit(
        self,
        samples: Sequence[BaseSample],
        rewards: Dict[str, torch.Tensor],
        reward_name: Optional[str] = None,
    ) -> List[Future[CritiqueResult]]:
        """Launch all critique rows so API latency overlaps paired rendering.

        Args:
            samples: Local round-1 T2I samples.
            rewards: Named local round-1 reward tensors.

        Returns:
            Row-aligned backend futures.
        """

        selected_reward = reward_name or self._reward_name(rewards)
        return self.backend.submit(self._requests(samples, rewards, selected_reward))

    def _validate(
        self,
        sample: BaseSample,
        result: CritiqueResult,
    ) -> Tuple[str, bool, str]:
        original = str(sample.prompt or "").strip()
        rewrite = str(result.rewrite or "").strip()
        if result.error:
            return original, False, f"backend_error:{result.error}"
        if not rewrite:
            return original, False, "empty_rewrite"
        if rewrite == original:
            return original, False, "unchanged_rewrite"
        if self.config.validator == "geneval":
            valid, reason = validate_geneval_rewrite(original, rewrite, self._metadata(sample))
            return (rewrite if valid else original), valid, reason
        return rewrite, True, "ok"

    @staticmethod
    def _group_indices(samples: Sequence[BaseSample]) -> List[List[int]]:
        groups: "OrderedDict[Any, List[int]]" = OrderedDict()
        for index, sample in enumerate(samples):
            batch_id = sample.extra_kwargs.get("critique_batch_id")
            if batch_id is None:
                raise RuntimeError(
                    "Critique requires samples generated through a paired-rollout hook; "
                    "missing sample.extra_kwargs['critique_batch_id']"
                )
            groups.setdefault(batch_id, []).append(index)
        return list(groups.values())

    @classmethod
    def _conditioning(cls, trainer: "BaseTrainer", sample: BaseSample) -> Dict[str, Any]:
        sample_dict = sample.to_dict()
        accepted = filter_kwargs(trainer.adapter.forward, **sample_dict)
        return {
            key: value
            for key, value in accepted.items()
            if key not in cls._CONDITIONING_EXCLUDE and value is not None
        }

    @staticmethod
    def _copy_reward_context(original: BaseSample, refined: BaseSample) -> None:
        """Make round 2 score as the original prompt/group, not the rewrite."""

        refined.prompt = original.prompt
        refined.prompt_ids = original.prompt_ids
        refined.negative_prompt = original.negative_prompt
        refined.negative_prompt_ids = original.negative_prompt_ids
        refined.source = original.source
        refined.source_id = original.source_id
        refined.applicable_rewards = set()
        for key, value in original.extra_kwargs.items():
            if key not in {"rewards", "advantage", "critique", REWARD_METADATA_KEY}:
                refined.extra_kwargs[key] = value
        refined.reset_unique_id()

    @staticmethod
    def _selected_round2_reward(
        trainer: "BaseTrainer",
        reward_name: str,
        samples: List[BaseSample],
    ) -> torch.Tensor:
        reward_processor = trainer.reward_processor
        if reward_name in reward_processor._pointwise_models:
            result = reward_processor._compute_pointwise_rewards(
                samples,
                epoch=trainer.epoch,
                models={reward_name: reward_processor._pointwise_models[reward_name]},
            )
        elif reward_name in reward_processor._groupwise_models:
            result = reward_processor._compute_groupwise_rewards(
                samples,
                epoch=trainer.epoch,
                models={reward_name: reward_processor._groupwise_models[reward_name]},
            )
        else:
            raise KeyError(f"No loaded training reward model named {reward_name!r}")
        trainer.accelerator.wait_for_everyone()
        return result[reward_name].detach().cpu()

    def _encode_rewrites(
        self,
        trainer: "BaseTrainer",
        rewrites: List[str],
        originals: Sequence[BaseSample],
    ) -> Dict[str, Any]:
        negative_prompts = [sample.negative_prompt for sample in originals]
        negative_prompt = (
            [str(value or "") for value in negative_prompts]
            if any(value is not None for value in negative_prompts)
            else None
        )
        device = (
            trainer.accelerator.device
            if self.config.prompt_encoding_device == "accelerator"
            else torch.device("cpu")
        )
        encode_kwargs = {
            **trainer.training_args,
            "prompt": rewrites,
            "negative_prompt": negative_prompt,
            "device": device,
        }
        encoded = trainer.adapter.encode_prompt(
            **filter_kwargs(trainer.adapter.encode_prompt, **encode_kwargs)
        )
        if not isinstance(encoded, dict) or not encoded:
            raise RuntimeError(
                f"Adapter {type(trainer.adapter).__name__} did not return prompt conditioning for critique"
            )
        return encoded

    def _compute_advantage(
        self,
        trainer: "BaseTrainer",
        samples: List[BaseSample],
        round1_reward: torch.Tensor,
        round2_reward: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        collected, group_indices, _ = trainer.advantage_processor.collect_group_rewards(
            samples,
            {"round1": round1_reward, "round2": round2_reward},
        )
        _, global_std = trainer.advantage_processor._global_mean_std(collected["round1"])
        advantage = group_normalized_improvement(
            round1_rewards=collected["round1"],
            round2_rewards=collected["round2"],
            group_indices=group_indices,
            global_std=global_std,
            clip_range=self.config.advantage_clip_range,
            mode=self.config.advantage_mode,
        )
        local = trainer.advantage_processor._to_local(advantage)
        return local * valid_mask.to(device=local.device, dtype=local.dtype)

    def refine(
        self,
        trainer: "BaseTrainer",
        samples: List[BaseSample],
        rewards: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Run paired round 2 and attach the training pair to round-1 samples.

        Args:
            trainer: Trainer providing adapter, reward, advantage, and distributed services.
            samples: Local round-1 T2I samples.
            rewards: Named local round-1 reward tensors.

        Returns:
            Globally reduced critique metrics.
        """

        if not samples:
            return {}
        if not all(self._is_t2i_sample(sample) for sample in samples):
            raise TypeError(
                "The critique component currently supports text-to-image samples only; "
                f"got {[type(sample).__name__ for sample in samples[:3]]}"
            )
        if any(sample.image is None for sample in samples):
            raise ValueError("Critique requires a decoded image on every sample")

        reward_name = self._reward_name(rewards)
        round1_reward = torch.as_tensor(rewards[reward_name], dtype=torch.float32).detach().cpu()
        if not torch.isfinite(round1_reward).all():
            raise ValueError(
                f"critique.reward_name={reward_name!r} must apply to every refined sample and return finite values"
            )

        futures = self.submit(samples, rewards, reward_name=reward_name)
        groups = self._group_indices(samples)
        round2_samples: List[BaseSample] = []
        conditioning: List[Optional[Dict[str, Any]]] = [None] * len(samples)
        clean_latents: List[Optional[torch.Tensor]] = [None] * len(samples)
        rewrites: List[str] = [""] * len(samples)
        reasons: List[str] = [""] * len(samples)
        valid = torch.zeros(len(samples), dtype=torch.bool)

        manage_text_encoders = trainer.config.data_args.enable_preprocess
        if manage_text_encoders:
            encode_device = (
                trainer.accelerator.device
                if self.config.prompt_encoding_device == "accelerator"
                else torch.device("cpu")
            )
            trainer.adapter.on_load_text_encoders(encode_device)
        try:
            trainer.adapter.rollout()
            for group in groups:
                originals = [samples[index] for index in group]
                resolved: List[str] = []
                for index in group:
                    rewrite, is_valid, reason = self._validate(
                        samples[index], futures[index].result()
                    )
                    rewrites[index] = rewrite
                    reasons[index] = reason
                    valid[index] = is_valid
                    resolved.append(rewrite)

                encoded = self._encode_rewrites(trainer, resolved, originals)
                seed = originals[0].extra_kwargs.get("critique_seed")
                if seed is None or any(
                    sample.extra_kwargs.get("critique_seed") != seed for sample in originals
                ):
                    raise RuntimeError(
                        "Each critique rollout batch must carry one consistent critique_seed"
                    )
                generator = torch.Generator().manual_seed(int(seed))
                negative_prompts = [sample.negative_prompt for sample in originals]
                round2_batch = {
                    "prompt": resolved,
                    "negative_prompt": (
                        [str(value or "") for value in negative_prompts]
                        if any(value is not None for value in negative_prompts)
                        else None
                    ),
                    **encoded,
                }
                with torch.no_grad(), trainer.autocast():
                    refined_group = trainer.sample_batch(
                        round2_batch,
                        reward_buffer=None,
                        compute_log_prob=False,
                        trajectory_indices=[-1],
                        generator=generator,
                    )
                if len(refined_group) != len(group):
                    raise RuntimeError(
                        f"Round-2 inference returned {len(refined_group)} samples for a group of {len(group)}"
                    )
                for index, original, refined in zip(group, originals, refined_group):
                    conditioning[index] = self._conditioning(trainer, refined)
                    if refined.all_latents is None:
                        raise RuntimeError("Round-2 inference did not record the final latent")
                    clean_latents[index] = refined.all_latents[-1]
                    self._copy_reward_context(original, refined)
                round2_samples.extend(refined_group)
        finally:
            if manage_text_encoders:
                trainer.adapter.off_load_text_encoders()

        round2_reward = self._selected_round2_reward(trainer, reward_name, round2_samples)
        advantage = self._compute_advantage(trainer, samples, round1_reward, round2_reward, valid)

        for index, sample in enumerate(samples):
            target_device = sample.all_latents.device
            if clean_latents[index] is None or conditioning[index] is None:
                raise RuntimeError(f"Critique pair at local sample index {index} is incomplete")
            sample.extra_kwargs["critique"] = {
                "rewrite": rewrites[index],
                "valid": valid[index],
                "reason": reasons[index],
                "round2_reward": round2_reward[index].detach().to(target_device),
                "advantage": advantage[index].detach().to(target_device),
                "clean_latents": clean_latents[index].detach().to(target_device),
                "conditioning": conditioning[index],
            }

        local_stats = torch.tensor(
            [
                float(len(samples)),
                float(valid.sum().item()),
                float((round2_reward - round1_reward).sum().item()),
                float(advantage.sum().item()),
            ],
            device=trainer.accelerator.device,
        )
        global_stats = trainer.accelerator.reduce(local_stats, reduction="sum")
        count = max(global_stats[0].item(), 1.0)
        return {
            "critique/valid_rate": global_stats[1].item() / count,
            "critique/reward_improvement_mean": global_stats[2].item() / count,
            "critique/advantage_mean": global_stats[3].item() / count,
        }

    def close(self, wait: bool = True) -> None:
        """Release critique backend resources.

        Args:
            wait: Wait for in-flight critique rows when true.
        """

        self.backend.close(wait=wait)
