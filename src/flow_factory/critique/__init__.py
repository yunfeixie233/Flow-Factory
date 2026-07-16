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

"""Standalone T2I critique/refinement component."""

from .abc import BaseCritiqueBackend, CritiqueRequest, CritiqueResult
from .loss import critique_direction_loss, ppd_same_state_distillation_loss
from .ppd import PPDProcessor, load_ppd_records
from .processor import CritiqueProcessor
from .registry import load_critique_backend
from .validators import validate_geneval_rewrite

__all__ = [
    "BaseCritiqueBackend",
    "CritiqueRequest",
    "CritiqueResult",
    "CritiqueProcessor",
    "PPDProcessor",
    "critique_direction_loss",
    "load_critique_backend",
    "load_ppd_records",
    "ppd_same_state_distillation_loss",
    "validate_geneval_rewrite",
]
