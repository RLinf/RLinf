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

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

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
    raise ValueError("sglang is not installed.")
elif package_version >= parse("0.4.4") and package_version <= parse("0.5.19"):
    sglang_version = package_version
else:
    raise ValueError(
        f"sglang version {package_version} is not supported "
        "(supported range: >=0.4.4, <=0.5.19)."
    )

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rlinf.hybrid_engines.sglang.common import io_struct
    from rlinf.hybrid_engines.sglang.common.sgl_engine import Engine
    from rlinf.workers.rollout.sglang.launcher import (
        get_sglang_api_url,
        launch_sglang_api,
        launch_sglang_router_and_server,
    )
    from rlinf.workers.rollout.sglang.router_worker import SGLangRouterWorker
    from rlinf.workers.rollout.sglang.server_worker import SGLangServerWorker

# The patched Engine and io_struct are imported lazily: callers that only
# need the standalone server/router helpers below must not pay for loading
# the RLinf sglang patch (and its heavyweight import chain).
_LAZY_ATTRS = {
    "Engine": ("rlinf.hybrid_engines.sglang.common.sgl_engine", "Engine"),
    "io_struct": ("rlinf.hybrid_engines.sglang.common.io_struct", None),
    "SGLangRouterWorker": (
        "rlinf.workers.rollout.sglang.router_worker",
        "SGLangRouterWorker",
    ),
    "SGLangServerWorker": (
        "rlinf.workers.rollout.sglang.server_worker",
        "SGLangServerWorker",
    ),
    "get_sglang_api_url": (
        "rlinf.workers.rollout.sglang.launcher",
        "get_sglang_api_url",
    ),
    "launch_sglang_api": (
        "rlinf.workers.rollout.sglang.launcher",
        "launch_sglang_api",
    ),
    "launch_sglang_router_and_server": (
        "rlinf.workers.rollout.sglang.launcher",
        "launch_sglang_router_and_server",
    ),
}

__all__ = [
    "Engine",
    "SGLangRouterWorker",
    "SGLangServerWorker",
    "get_sglang_api_url",
    "io_struct",
    "launch_sglang_api",
    "launch_sglang_router_and_server",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        module_name, attr = _LAZY_ATTRS[name]
        module = import_module(module_name)
        value = module if attr is None else getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
