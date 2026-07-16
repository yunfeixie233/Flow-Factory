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

"""Backend-neutral critique request and result contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from PIL import Image


@dataclass(frozen=True)
class CritiqueRequest:
    """One image/prompt row sent to a critique backend."""

    image: Image.Image
    prompt: str
    axis_scores: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    clause_report: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class CritiqueResult:
    """Row-aligned backend response.

    Backends report transport/provider failures in ``error`` instead of
    throwing away other usable rows from the same rollout batch.
    """

    rewrite: str = ""
    error: Optional[str] = None


class BaseCritiqueBackend(ABC):
    """Asynchronous row-level interface implemented by all critic providers."""

    @abstractmethod
    def submit(self, requests: List[CritiqueRequest]) -> List[Future[CritiqueResult]]:
        """Submit rows without waiting.

        Args:
            requests: Image/prompt rows to critique.

        Returns:
            Futures in the same order as ``requests``.
        """

    def critique(self, requests: List[CritiqueRequest]) -> List[CritiqueResult]:
        """Resolve a batch synchronously.

        Args:
            requests: Image/prompt rows to critique.

        Returns:
            Row-aligned critique results.
        """

        return [future.result() for future in self.submit(requests)]

    def close(self, wait: bool = True) -> None:
        """Release provider resources.

        Args:
            wait: Wait for in-flight rows when true; otherwise request non-blocking shutdown.
        """
