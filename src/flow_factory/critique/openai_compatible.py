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

"""OpenAI-compatible multimodal critique backend."""

from __future__ import annotations

import base64
import io
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List

import requests
from PIL import Image

from ..hparams import CritiqueArguments
from .abc import BaseCritiqueBackend, CritiqueRequest, CritiqueResult
from .prompts import get_critique_prompt


class OpenAICompatibleCritiqueBackend(BaseCritiqueBackend):
    """Concurrent row-aligned client for OpenAI-compatible chat endpoints."""

    def __init__(self, config: CritiqueArguments) -> None:
        self.config = config
        self._resolve_prompt()
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise ValueError(
                f"Critique API key environment variable {config.api_key_env!r} is not set"
            )
        self.api_key = api_key
        base_url = str(config.base_url).rstrip("/")
        self.url = (
            base_url if base_url.endswith("/chat/completions") else base_url + "/chat/completions"
        )
        self._executor = ThreadPoolExecutor(
            max_workers=config.num_workers, thread_name_prefix="flow-factory-critique"
        )
        # Share the underlying urllib3 connection pool across rows and epochs.
        # The session is immutable after construction, so concurrent requests
        # only share its thread-safe transport pool.
        self._session = requests.Session()

    def _resolve_prompt(self) -> None:
        """Resolve the active prompt recipe.

        Called at construction and again per :meth:`submit` so a
        ``prompts_yaml`` overlay edit is picked up by a live run's next
        critique batch (the resolver caches on file mtime).
        """
        self.system_prompt, self.user_builder = get_critique_prompt(
            self.config.mode,
            self.config.system_prompt,
            prompts_yaml=self.config.prompts_yaml,
        )

    def _image_data_uri(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        image = image.convert("RGB")
        if self.config.image_format == "png":
            image.save(buffer, format="PNG")
            mime = "image/png"
        elif self.config.image_format == "jpeg":
            image.save(
                buffer,
                format="JPEG",
                quality=self.config.image_quality,
                subsampling=0,
            )
            mime = "image/jpeg"
        else:
            image.save(
                buffer,
                format="WEBP",
                quality=self.config.image_quality,
                method=0,
            )
            mime = "image/webp"
        payload = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:{mime};base64,{payload}"

    @staticmethod
    def _extract_text(payload: Dict[str, Any]) -> str:
        content = payload["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "\n".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                for part in content
            )
        text = str(content or "").strip().strip('"').strip().strip("*").strip()
        lowered = text.lower()
        if "prompt:" in lowered:
            text = text[lowered.rfind("prompt:") + len("prompt:") :].strip()
        lines = [
            line.strip().strip('"').strip("*").strip() for line in text.splitlines() if line.strip()
        ]
        return max(lines, key=len) if lines else ""

    def _critique_one(self, request: CritiqueRequest) -> CritiqueResult:
        user_text = self.user_builder(request.prompt, request.axis_scores, request.clause_report)
        body: Dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": self._image_data_uri(request.image)},
                        },
                    ],
                },
            ],
        }
        if self.config.reasoning_effort:
            body["reasoning_effort"] = self.config.reasoning_effort
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self._session.post(
                    self.url,
                    headers=headers,
                    json=body,
                    timeout=self.config.timeout,
                )
                response.raise_for_status()
                payload = response.json()
                finish_reason = str(payload["choices"][0].get("finish_reason") or "").lower()
                if finish_reason in {"length", "max_tokens"}:
                    return CritiqueResult(error="length_truncated")
                rewrite = self._extract_text(payload)
                return CritiqueResult(rewrite=rewrite, error=None if rewrite else "empty")
            except Exception as exc:  # provider errors are isolated per row
                last_error = exc
                if attempt < self.config.max_retries and self.config.retry_backoff:
                    time.sleep(self.config.retry_backoff * (2**attempt))
        assert last_error is not None
        return CritiqueResult(error=f"{type(last_error).__name__}: {last_error}")

    def submit(self, requests: List[CritiqueRequest]) -> List[Future[CritiqueResult]]:
        """Submit critique rows to the persistent worker pool.

        Args:
            requests: Image/prompt rows to critique.

        Returns:
            Row-aligned provider futures.
        """
        if self.config.prompts_yaml:
            self._resolve_prompt()
        return [self._executor.submit(self._critique_one, request) for request in requests]

    def close(self, wait: bool = True) -> None:
        """Close the worker and HTTP pools.

        Args:
            wait: Wait for submitted requests when true.
        """

        self._executor.shutdown(wait=wait, cancel_futures=not wait)
        self._session.close()
