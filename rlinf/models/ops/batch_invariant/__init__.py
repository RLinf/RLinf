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


from collections.abc import Callable
from typing import Any

from torch import Tensor

PatchEmbedding = Callable[[Tensor, Tensor, Tensor | None], Tensor]
Operator = Callable[..., Any]
OperatorLoader = Callable[[], Operator]
_operators: dict[tuple[str, str], OperatorLoader] = {}


def register_batch_invariant_op(name: str, backend: str, load: OperatorLoader) -> None:
    """Register a lazy loader for one operator implementation."""
    _operators[name, backend] = load


def get_batch_invariant_op(name: str, backend: str) -> Operator:
    """Load an implementation when constructing a model."""
    try:
        load = _operators[name, backend]
    except KeyError:
        raise ValueError(
            f"No batch-invariant {name!r} implementation for backend {backend!r}."
        ) from None
    return load()


def _load_triton_linear() -> Operator:
    from .linear import TritonLinear

    return TritonLinear


def _load_triton_patch_embedding() -> PatchEmbedding:
    from .patch_embedding import triton_patch_embedding

    return triton_patch_embedding


register_batch_invariant_op("linear", "triton", _load_triton_linear)
register_batch_invariant_op("patch_embedding", "triton", _load_triton_patch_embedding)
