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

"""``torch.compile`` of large CUDA batches, shared by the SuperMimic envs,
training scans and kinematics.

Batches on the CPU or under 256 rows run eagerly. A piece that fails to
compile warns once and stays eager for the rest of the process; the result is
the same, only slower.
"""

from __future__ import annotations

from typing import Callable

import torch

from rlinf.utils.logging import get_logger

MIN_COMPILED_ROWS = 256


class CompileCache:
    """Compiled pieces and fallback state of one owner.

    Args:
        enabled: Whether large CUDA batches are compiled at all.
        fullgraph: Passed to ``torch.compile`` for every piece of this owner.
    """

    def __init__(self, enabled: bool = True, fullgraph: bool = False):
        self.enabled = bool(enabled)
        self.fullgraph = bool(fullgraph)
        self.compiled: dict[str, Callable] = {}
        self.failed: set[str] = set()


def _leading_batch(args: tuple) -> torch.Tensor | None:
    for value in args:
        if torch.is_tensor(value) and value.dim() >= 1 and value.shape[0] > 0:
            return value
    return None


def run_compiled(name: str, fn: Callable, *args, cache: CompileCache):
    """Return ``fn(*args)``, compiled when the batch is large and on CUDA.

    The batch is the first non-empty tensor argument. The first compiled call
    under ``name`` compiles ``fn`` and later calls reuse that compile, so
    every ``fn`` passed under one name must compute the same function.
    """
    probe = _leading_batch(args)
    eager = (
        not cache.enabled
        or name in cache.failed
        or probe is None
        or not probe.is_cuda
        or int(probe.shape[0]) < MIN_COMPILED_ROWS
    )
    if eager:
        return fn(*args)
    compiled = cache.compiled.get(name)
    if compiled is None:
        get_logger().info(
            f"Compiling SuperMimic {name} for large CUDA batches. The first "
            "such batch waits; later batches reuse it."
        )
        compiled = torch.compile(fn, dynamic=True, fullgraph=cache.fullgraph)
        cache.compiled[name] = compiled
    try:
        return compiled(*args)
    except Exception as exc:
        module = type(exc).__module__
        if module.startswith("torch._dynamo") or "Compiler" in type(exc).__name__:
            cache.failed.add(name)
            cache.compiled.pop(name, None)
            get_logger().warning(
                f"SuperMimic {name} stayed eager ({type(exc).__name__})."
            )
            return fn(*args)
        raise
