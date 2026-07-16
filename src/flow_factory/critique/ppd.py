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

"""Records-based privileged-prompt attachment for distillation-capable trainers.

The processor joins each training sample to a precomputed privileged
(conditioning) prompt by exact original-prompt text, encodes the privileged
prompts with the adapter's text encoders, and attaches per-sample conditioning
plus an activity mask under ``sample.extra_kwargs['ppd']``.  It performs no
API calls, no second rendering, and no reward evaluation: rollouts and rewards
remain conditioned on the original prompt everywhere.  Trainers decide whether
and how to consume the attached conditioning in their loss.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Tuple

import torch

from ..hparams import PPDArguments
from ..utils.base import filter_kwargs
from ..utils.logger_utils import setup_logger

if TYPE_CHECKING:
    from ..samples import BaseSample
    from ..trainers.abc import BaseTrainer


logger = setup_logger(__name__)

PPD_RECORD_SCHEMA_VERSION = 1


def load_ppd_records(records_path: str) -> Dict[str, Tuple[str, bool]]:
    """Load and validate a privileged-prompt records JSONL file.

    Returns:
        Mapping from original prompt text to ``(conditioning_prompt, changed)``.
        The first record wins when several records share one original prompt;
        later records with a different conditioning prompt are counted and
        reported, matching the deterministic first-occurrence join used to
        build row-keyed record files.
    """

    if not os.path.isfile(records_path):
        raise FileNotFoundError(f"PPD records file does not exist: {records_path!r}")

    mapping: Dict[str, Tuple[str, bool]] = {}
    conflicting = 0
    rows = 0
    with open(records_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{records_path}:{line_number}: invalid JSON record: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"{records_path}:{line_number}: record must be an object")
            version = record.get("schema_version")
            if version != PPD_RECORD_SCHEMA_VERSION:
                raise ValueError(
                    f"{records_path}:{line_number}: unsupported schema_version {version!r}"
                )
            original = record.get("original_prompt")
            conditioning = record.get("conditioning_prompt")
            changed = record.get("changed")
            if not isinstance(original, str) or not original:
                raise ValueError(f"{records_path}:{line_number}: missing original_prompt")
            if not isinstance(conditioning, str) or not conditioning:
                raise ValueError(f"{records_path}:{line_number}: missing conditioning_prompt")
            if not isinstance(changed, bool):
                raise ValueError(f"{records_path}:{line_number}: missing boolean changed flag")
            if changed == (conditioning == original):
                raise ValueError(
                    f"{records_path}:{line_number}: changed flag contradicts the prompts"
                )
            rows += 1
            existing = mapping.get(original)
            if existing is None:
                mapping[original] = (conditioning, changed)
            elif existing[0] != conditioning:
                conflicting += 1

    if not mapping:
        raise ValueError(f"PPD records file contains no records: {records_path!r}")
    if conflicting:
        logger.warning(
            "[ppd/data] %d of %d records share an original prompt with a different "
            "conditioning prompt; the first occurrence is used for every duplicate.",
            conflicting,
            rows,
        )
    return mapping


class PPDProcessor:
    """Privileged-prompt conditioning attachment service.

    The processor owns records loading/validation, privileged-prompt text
    encoding, and per-sample conditioning attachment.  It does not own an
    optimizer loss; trainers decide whether and how to consume the attached
    conditioning.  The matched ``rho=0`` control arm runs this exact plumbing.
    """

    def __init__(self, config: PPDArguments) -> None:
        self.config = config
        self._records = load_ppd_records(config.records_path)
        changed = sum(1 for _, is_changed in self._records.values() if is_changed)
        logger.info(
            "[ppd/data] records=%s unique_prompts=%d changed=%d (%.2f%%)",
            config.records_path,
            len(self._records),
            changed,
            100.0 * changed / max(len(self._records), 1),
        )

    def _resolve(self, prompt: str) -> Tuple[str, bool, bool]:
        """Return ``(conditioning_prompt, active, covered)`` for one prompt."""
        record = self._records.get(prompt)
        if record is None:
            if self.config.require_records_coverage:
                raise KeyError(
                    "PPD records do not cover the training prompt "
                    f"{prompt!r}; regenerate the records file or set "
                    "ppd.require_records_coverage=false to mask uncovered rows"
                )
            return prompt, False, False
        conditioning, changed = record
        active = changed if self.config.mask_identity else True
        return conditioning, active, True

    def _encode_privileged(
        self,
        trainer: "BaseTrainer",
        prompts: List[str],
        originals: Sequence["BaseSample"],
    ) -> Dict[str, torch.Tensor]:
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
            "prompt": prompts,
            "negative_prompt": negative_prompt,
            "device": device,
        }
        encoded = trainer.adapter.encode_prompt(
            **filter_kwargs(trainer.adapter.encode_prompt, **encode_kwargs)
        )
        if not isinstance(encoded, dict) or not encoded:
            raise RuntimeError(
                f"Adapter {type(trainer.adapter).__name__} did not return prompt "
                "conditioning for privileged-prompt distillation"
            )
        tensor_conditioning = {
            key: value for key, value in encoded.items() if isinstance(value, torch.Tensor)
        }
        if not tensor_conditioning:
            raise RuntimeError(
                "Privileged-prompt encoding returned no tensor conditioning fields; "
                f"got keys {sorted(encoded)}"
            )
        return tensor_conditioning

    def attach(
        self,
        trainer: "BaseTrainer",
        samples: List["BaseSample"],
    ) -> Dict[str, float]:
        """Attach privileged conditioning and activity masks to local samples.

        Args:
            trainer: Trainer providing adapter, accelerator, and config services.
            samples: Local round-1 samples in collection order.

        Returns:
            Globally reduced data-level PPD metrics.
        """

        privileged: List[str] = []
        active_rows: List[float] = []
        covered_rows: List[float] = []
        for sample in samples:
            conditioning_prompt, active, covered = self._resolve(str(sample.prompt or ""))
            privileged.append(conditioning_prompt)
            active_rows.append(float(active))
            covered_rows.append(float(covered))

        # An empty local sample list still participates in the metric reduce
        # below: every rank must join the collective or the others deadlock.
        if samples:
            manage_text_encoders = trainer.config.data_args.enable_preprocess
            if manage_text_encoders:
                encode_device = (
                    trainer.accelerator.device
                    if self.config.prompt_encoding_device == "accelerator"
                    else torch.device("cpu")
                )
                trainer.adapter.on_load_text_encoders(encode_device)
            try:
                with torch.no_grad():
                    conditioning = self._encode_privileged(trainer, privileged, samples)
            finally:
                if manage_text_encoders:
                    trainer.adapter.off_load_text_encoders()

        for index, sample in enumerate(samples):
            target_device = sample.all_latents.device if sample.all_latents is not None else None
            row_conditioning: Dict[str, Any] = {}
            for key, value in conditioning.items():
                row_value = value[index].detach()
                if target_device is not None:
                    row_value = row_value.to(target_device)
                row_conditioning[key] = row_value
            sample.extra_kwargs["ppd"] = {
                "active": torch.tensor(active_rows[index], dtype=torch.float32),
                "conditioning": row_conditioning,
            }

        local_stats = torch.tensor(
            [float(len(samples)), float(sum(active_rows)), float(sum(covered_rows))],
            device=trainer.accelerator.device,
        )
        global_stats = trainer.accelerator.reduce(local_stats, reduction="sum")
        count = max(global_stats[0].item(), 1.0)
        return {
            "ppd/data_active_rate": global_stats[1].item() / count,
            "ppd/data_coverage_rate": global_stats[2].item() / count,
        }
