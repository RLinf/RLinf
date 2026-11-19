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

"""Sphinx autodoc mock configuration for building RLinf docs without runtime deps."""

from __future__ import annotations

import asyncio
import importlib.abc
import importlib.metadata
import pathlib
import sys
from typing import Any

from sphinx.ext.autodoc.mock import _MockModule, _MockObject

# Third-party runtime dependencies of RLinf mocked during autodoc builds.
# Keep Python stdlib modules and Sphinx's own packages (e.g. packaging, yaml,
# requests, urllib3, typing_extensions, starlette, uvicorn) out of this list.
AUTODOC_MOCK_IMPORTS = [
    # Core RLinf dependencies
    "accelerate",
    "cloudpickle",
    "datasets",
    "debugpy",
    "einops",
    "filelock",
    "gsutil",
    "huggingface_hub",
    "hydra",
    "icmplib",
    "liger_kernel",
    "lz4",
    "ninja",
    "numpy",
    "nvitop",
    "omegaconf",
    "psutil",
    "pybind11",
    "pytest",
    "ray",
    "ruff",
    "scipy",
    "sortedcontainers",
    "swanlab",
    "tensorboard",
    "torch",
    "torchdata",
    "tqdm",
    "wandb",
    "zstandard",
    # Agentic stack & LLM rollout / training engines
    "agentlightning",
    "aiohttp",
    "fastapi",
    "flash_attn",
    "latex2sympy2",
    "math_verify",
    "mcp",
    "megatron",
    "nvidia",
    "nvidia_resiliency_ext",
    "peft",
    "pydantic",
    "pylatexenc",
    "regex",
    "rlinf_latex2sympy2",
    "sglang",
    "sglang_router",
    "sympy",
    "transformer_engine",
    "uvloop",
    "vllm",
    "word2number",
    "zmq",
    # Embodied stack, models, simulators & robotics SDKs
    "ABot",
    "PIL",
    "av",
    "calvin_agent",
    "calvin_env",
    "cosmos_framework",
    "cv2",
    "d4rl",
    "decord",
    "dexbotic",
    "diffsynth",
    "diffusers",
    "embodichain",
    "fastwam",
    "flax",
    "franka_sim",
    "franky",
    "genesis",
    "gr00t",
    "groot",
    "gym",
    "gymnasium",
    "habitat",
    "habitat_baselines",
    "habitat_sim",
    "imageio",
    "isaaclab",
    "isaaclab_tasks",
    "jax",
    "lerobot",
    "libero",
    "liberoplus",
    "liberopro",
    "lingbotvla",
    "mani_skill",
    "metasim",
    "metaworld",
    "mink",
    "mplib",
    "mujoco",
    "mujoco_warp",
    "networkx",
    "omni",
    "omnigibson",
    "onnxruntime",
    "openpi",
    "openpi_client",
    "opensora",
    "orbax",
    "pandas",
    "phyai",
    "pinocchio",
    "polaris",
    "prismatic",
    "pyarrow",
    "quaternion",
    "rlinf_geniesim",
    "robocasa",
    "robodojo_runtime",
    "robolab",
    "robosuite",
    "robotwin",
    "safetensors",
    "sapien",
    "starVLA",
    "tensorflow",
    "tianshou",
    "timm",
    "torch_npu",
    "torch_supa",
    "torchcodec",
    "torchvision",
    "transformers",
    "transforms3d",
    "tree",
    "tyro",
    "wan_video_action",
    "warp",
]

# Valid versions for packages whose metadata or __version__ is checked at import time.
_MOCK_PACKAGE_VERSIONS = {
    "ray": "2.47.0",
    "sglang": "0.5.12",
    "torch": "2.6.0",
    "vllm": "0.8.5",
}

_SHIMS_INSTALLED = False


class _MockDistribution(importlib.metadata.Distribution):
    def __init__(self, name: str, version: str) -> None:
        self._name = name
        self._version = version

    def read_text(self, filename: str) -> str | None:
        if filename == "METADATA":
            return (
                f"Metadata-Version: 2.1\nName: {self._name}\nVersion: {self._version}\n"
            )
        return None

    def locate_file(self, path: str | pathlib.Path) -> pathlib.Path:
        return pathlib.Path(path)


class _MockDistributionFinder(importlib.abc.MetaPathFinder):
    """Fallback metadata finder for mocked runtime packages."""

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> None:
        return None

    def find_distributions(
        self,
        context: importlib.metadata.DistributionFinder.Context = (
            importlib.metadata.DistributionFinder.Context()
        ),
    ) -> list[_MockDistribution]:
        name = context.name
        if name is None:
            return [_MockDistribution(k, v) for k, v in _MOCK_PACKAGE_VERSIONS.items()]
        if name in _MOCK_PACKAGE_VERSIONS:
            return [_MockDistribution(name, _MOCK_PACKAGE_VERSIONS[name])]
        return []


def install_autodoc_mock_shims() -> None:
    """Extend Sphinx's autodoc mocks for import-time version checks and PEP 604 unions."""
    global _SHIMS_INSTALLED
    if _SHIMS_INSTALLED:
        return
    _SHIMS_INSTALLED = True

    sys.meta_path.append(_MockDistributionFinder())

    orig_mod_getattr = _MockModule.__getattr__

    def _mock_module_getattr(self: _MockModule, name: str) -> Any:
        if name == "__version__":
            return _MOCK_PACKAGE_VERSIONS.get(self.__name__, "0.0.0")
        if self.__name__ == "uvloop" and name == "EventLoopPolicy":
            return asyncio.DefaultEventLoopPolicy
        obj = orig_mod_getattr(self, name)
        # Preserve identity across repeated accesses (e.g. torch.uint8 dict keys).
        setattr(self, name, obj)
        return obj

    _MockModule.__getattr__ = _mock_module_getattr  # type: ignore[method-assign]

    orig_obj_getattr = _MockObject.__getattr__

    def _mock_object_getattr(self: _MockObject, key: str) -> Any:
        # inspect.signature() inspects __signature__ and loops over __wrapped__.
        if key in ("__signature__", "__wrapped__"):
            raise AttributeError(key)
        return orig_obj_getattr(self, key)

    def _mock_or(self: _MockObject, other: Any) -> Any:
        other_t = other.__class__ if isinstance(other, _MockObject) else other
        try:
            return self.__class__ | other_t
        except TypeError:
            return self

    def _mock_ror(self: _MockObject, other: Any) -> Any:
        other_t = other.__class__ if isinstance(other, _MockObject) else other
        try:
            return other_t | self.__class__
        except TypeError:
            return self

    _MockObject.__getattr__ = _mock_object_getattr  # type: ignore[method-assign]
    _MockObject.__or__ = _mock_or  # type: ignore[attr-defined]
    _MockObject.__ror__ = _mock_ror  # type: ignore[attr-defined]
    _MockObject.__lt__ = lambda self, other: False  # type: ignore[attr-defined]
    _MockObject.__le__ = lambda self, other: False  # type: ignore[attr-defined]
    _MockObject.__gt__ = lambda self, other: False  # type: ignore[attr-defined]
    _MockObject.__ge__ = lambda self, other: False  # type: ignore[attr-defined]
    _MockObject.__fspath__ = lambda self: self.__display_name__  # type: ignore[attr-defined]
