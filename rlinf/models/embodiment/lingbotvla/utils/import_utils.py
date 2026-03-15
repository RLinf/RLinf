# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


"""Import utils"""

import importlib.metadata
import importlib.util
from functools import lru_cache
from typing import TYPE_CHECKING

from packaging import version

if TYPE_CHECKING:
    from packaging.version import Version


def _is_package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _get_package_version(name: str) -> "Version":
    try:
        return version.parse(importlib.metadata.version(name))
    except Exception:
        return version.parse("0.0.0")


_PACKAGE_FLAGS: dict[str, bool] = {
    "flash_attn": _is_package_available("flash_attn"),
    "liger_kernel": _is_package_available("liger_kernel"),
    "torch_npu": _is_package_available("torch_npu"),
    "vescale": _is_package_available("vescale"),
    "seed_kernels": _is_package_available("seed_kernels"),
    "bytecheckpoint": _is_package_available("bytecheckpoint"),
    "diffusers": _is_package_available("diffusers"),
    "av": _is_package_available("av"),
    "librosa": _is_package_available("librosa"),
    "soundfile": _is_package_available("soundfile"),
    "triton": _is_package_available("triton"),
}


def is_flash_attn_2_available() -> bool:
    return _PACKAGE_FLAGS["flash_attn"]


def is_liger_kernel_available() -> bool:
    return _PACKAGE_FLAGS["liger_kernel"]


def is_torch_npu_available() -> bool:
    return _PACKAGE_FLAGS["torch_npu"]


def is_vescale_available() -> bool:
    return _PACKAGE_FLAGS["vescale"]


def is_seed_kernels_available() -> bool:
    return _PACKAGE_FLAGS["seed_kernels"]


def is_bytecheckpoint_available() -> bool:
    return _PACKAGE_FLAGS["bytecheckpoint"]


def is_diffusers_available() -> bool:
    return _PACKAGE_FLAGS["diffusers"]


def is_fused_moe_available() -> bool:
    import torch

    return torch.cuda.is_available() and _PACKAGE_FLAGS["triton"]


def is_video_audio_available() -> bool:
    return (
        _PACKAGE_FLAGS["av"]
        and _PACKAGE_FLAGS["librosa"]
        and _PACKAGE_FLAGS["soundfile"]
    )


@lru_cache
def is_torch_version_greater_than(value: str) -> bool:
    return _get_package_version("torch") >= version.parse(value)


@lru_cache
def is_transformers_version_greater_or_equal_to(value: str) -> bool:
    return _get_package_version("transformers") > version.parse(value)
