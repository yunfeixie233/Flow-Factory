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

"""Critique backend registry and custom-class loader."""

from __future__ import annotations

import importlib

from ..hparams import CritiqueArguments
from .abc import BaseCritiqueBackend
from .openai_compatible import OpenAICompatibleCritiqueBackend

_BACKENDS = {
    "openai-compatible": OpenAICompatibleCritiqueBackend,
}


def load_critique_backend(config: CritiqueArguments) -> BaseCritiqueBackend:
    """Instantiate a registered backend or ``module.path:ClassName``.

    Args:
        config: Shared critique configuration.

    Returns:
        Initialized critique backend.
    """

    backend_cls = _BACKENDS.get(config.backend)
    if backend_cls is None:
        if ":" not in config.backend:
            raise ValueError(
                f"Unknown critique backend {config.backend!r}; available: {sorted(_BACKENDS)}, "
                "or use 'module.path:ClassName'"
            )
        module_name, class_name = config.backend.rsplit(":", 1)
        module = importlib.import_module(module_name)
        backend_cls = getattr(module, class_name)
    backend = backend_cls(config)
    if not isinstance(backend, BaseCritiqueBackend):
        raise TypeError(
            f"Critique backend {config.backend!r} must inherit BaseCritiqueBackend, got {type(backend)}"
        )
    return backend
