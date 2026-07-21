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

from importlib.metadata import PackageNotFoundError, version

from packaging.version import parse


def get_version(pkg):
    try:
        return parse(version(pkg))
    except PackageNotFoundError:
        return None


package_name = "sglang"
package_version = get_version(package_name)

sglang_version = None

if package_version is None:
    raise ValueError(
        "sglang is not installed. Install sglang_main (e.g. "
        "`pip install -e <sglang>/python`)."
    )
else:
    sglang_version = package_version
    import types as _types

    io_struct = _types.ModuleType("io_struct")
    Engine = None  # noqa: F841
    try:
        from rlinf.hybrid_engines.sglang.common import io_struct  # noqa: F811
        from rlinf.hybrid_engines.sglang.common.sgl_engine import (
            Engine,
        )
    except Exception as _e:  # pragma: no cover - depends on sglang version
        import logging as _logging

        _logging.getLogger(__name__).warning(
            "SRT hybrid engine unavailable on sglang %s (%s). The legacy "
            "SRT/LLM rollout path is disabled; the Cosmos3 evaluate path "
            "(sglang.multimodal_gen) is unaffected.",
            package_version,
            _e,
        )

__all__ = ["Engine", "io_struct"]
