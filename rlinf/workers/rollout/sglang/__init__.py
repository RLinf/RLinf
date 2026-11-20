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
elif parse("0.4.4") <= package_version <= parse("0.5.19"):
    sglang_version = package_version
else:
    from rlinf.hybrid_engines.sglang.multimodal.token_space import is_token_space_build

    if not is_token_space_build():
        raise ValueError(
            f"sglang version {package_version} is not supported "
            "(supported range: >=0.4.4, <=0.5.19 or the pinned token-space build)."
        )
    sglang_version = package_version


def require_supported_sglang(*, token_space: bool = False) -> None:
    """Restrict the pinned development build to explicit token-space rollout."""
    if not parse("0.4.4") <= sglang_version <= parse("0.5.19") and not token_space:
        raise ValueError(
            "The pinned experimental SGLang build requires "
            "rollout.sglang.enable_shared_multi_modal_processor=true. "
            "Use a supported release for other rollout, reward and embodied servers."
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


def multimodal_server_args(cfg, server_fields) -> dict:
    """Resolve shared image preprocessing options supported by the installed server."""
    import os

    from rlinf.config import shared_multi_modal_processor_enabled

    arguments = {}
    shared_processor = shared_multi_modal_processor_enabled(cfg)
    if shared_processor and str(sglang_version) == "0.5.12.post1":
        if os.environ.get("SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE") != (
            "rlinf.hybrid_engines.sglang.multimodal"
        ):
            raise ValueError(
                "Release image capture requires the RLinf external Qwen processor"
            )
    elif shared_processor:
        from rlinf.hybrid_engines.sglang.multimodal.token_space import (
            require_token_space_support,
        )

        if os.environ.get("SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE"):
            raise ValueError(
                "Shared multimodal preprocessing requires removing SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE"
            )
        require_token_space_support()
        arguments.update(
            enable_token_space_processor=True,
            disable_fast_image_processor=True,
            image_processor_backend="pil",
        )

    for name in (
        "enable_token_space_preprocess_cache",
        "token_space_preprocess_cache_dir",
        "token_space_preprocess_cache_trust_url",
        "token_space_preprocess_cache_check_rate",
    ):
        if name not in cfg:
            continue
        value = cfg[name]
        if name in server_fields:
            arguments[name] = value
        elif value:
            raise ValueError(
                f"Installed SGLang does not support {name}={value!r}; "
                "use a compatible build or disable this option."
            )
    return arguments


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        module_name, attr = _LAZY_ATTRS[name]
        module = import_module(module_name)
        value = module if attr is None else getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
