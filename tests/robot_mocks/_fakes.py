# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for constructing fake vendor modules."""

from __future__ import annotations

import importlib.machinery
import types
from typing import Any


def module(name: str, **members: Any) -> types.ModuleType:
    """Create a module with the supplied members.

    It carries a ``__spec__`` because libraries check for a package with
    ``importlib.util.find_spec``, which raises on a module that has none.
    """
    fake = types.ModuleType(name)
    fake.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    for key, value in members.items():
        setattr(fake, key, value)
    return fake


def package(name: str, *parts: str) -> list[types.ModuleType]:
    """Return every parent of a dotted module name, outermost first."""
    made = []
    pieces = name.split(".")
    for index in range(1, len(pieces)):
        made.append(module(".".join(pieces[:index])))
    return made


class Recorder:
    """Record arbitrary SDK method calls and return configured responses."""

    def __init__(self, name: str = "sdk", **answers: Any) -> None:
        self._name = name
        self._answers = answers
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, attribute: str) -> Any:
        if attribute.startswith("_"):
            raise AttributeError(attribute)
        if attribute in self._answers:
            return self._answers[attribute]

        def call(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((attribute, args, kwargs))
            return self._answers.get(f"{attribute}()", None)

        return call

    def called(self, name: str) -> bool:
        """Return whether a method with ``name`` was called."""
        return any(call == name for call, _, _ in self.calls)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Recorder {self._name} calls={[c for c, _, _ in self.calls]}>"
