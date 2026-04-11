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

from dataclasses import fields, is_dataclass
from typing import Any

from torch.utils import _pytree

_REGISTERED_PYTREE_DATACLASSES: set[type[Any]] = set()


def register_pytree_dataclass_type(cls: type[Any]) -> None:
    """Register a dataclass type so torch pytree can recurse into it."""
    if cls in _REGISTERED_PYTREE_DATACLASSES:
        return
    if not is_dataclass(cls):
        return

    field_names = tuple(field.name for field in fields(cls))

    def _flatten(instance: Any):
        return [getattr(instance, name) for name in field_names], field_names

    def _unflatten(values: list[Any], context: tuple[str, ...]):
        kwargs = dict(zip(context, values))
        return cls(**kwargs)

    try:
        _pytree.register_pytree_node(cls, _flatten, _unflatten)
    except ValueError:
        # Already registered in this process.
        pass
    _REGISTERED_PYTREE_DATACLASSES.add(cls)


def register_pytree_dataclasses(obj: Any) -> None:
    """Recursively register dataclass instances contained in an object tree."""
    if is_dataclass(obj) and not isinstance(obj, type):
        register_pytree_dataclass_type(type(obj))
        for field in fields(obj):
            register_pytree_dataclasses(getattr(obj, field.name))
        return
    if isinstance(obj, dict):
        for value in obj.values():
            register_pytree_dataclasses(value)
        return
    if isinstance(obj, (list, tuple)):
        for value in obj:
            register_pytree_dataclasses(value)
