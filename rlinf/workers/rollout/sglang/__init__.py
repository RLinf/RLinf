# Copyright 2025 The RLinf Authors.
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

"""SGLang rollout workers.

``Engine`` and ``io_struct`` need a supported sglang install and load on first
access, so the HTTP-only ``SGLangEmbodiedWorker`` imports without sglang.
"""

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from packaging.version import parse

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rlinf.hybrid_engines.sglang.common import io_struct
    from rlinf.hybrid_engines.sglang.common.sgl_engine import Engine

__all__ = ["Engine", "io_struct"]


def get_version(pkg):
    try:
        return parse(version(pkg))
    except PackageNotFoundError:
        return None


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    package_version = get_version("sglang")
    if package_version is None:
        raise ValueError("sglang is not installed.")
    if not parse("0.4.4") <= package_version <= parse("0.5.19"):
        raise ValueError(
            f"sglang version {package_version} is not supported "
            "(supported range: >=0.4.4, <=0.5.19)."
        )
    from rlinf.hybrid_engines.sglang.common import io_struct
    from rlinf.hybrid_engines.sglang.common.sgl_engine import Engine

    globals().update(Engine=Engine, io_struct=io_struct)
    return globals()[name]
