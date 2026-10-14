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

"""Compatibility shim for robosuite's import-time EGL device check.

``NvidiaGPUManager.setup_worker_process_env`` points ``MUJOCO_EGL_DEVICE_ID`` at
the EGL index of the GPU a worker owns. robosuite 1.4.1 rejects exactly that
value while importing ``robosuite.utils.binding_utils``::

    CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if CUDA_VISIBLE_DEVICES != "":
        MUJOCO_EGL_DEVICE_ID = os.environ.get("MUJOCO_EGL_DEVICE_ID", None)
        if MUJOCO_EGL_DEVICE_ID is not None:
            assert MUJOCO_EGL_DEVICE_ID.isdigit() and (
                MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES
            ), "MUJOCO_EGL_DEVICE_ID needs to be set to one of the device id ..."

The check assumes the two namespaces agree, so a correct EGL index fails it
whenever they do not. It is also the only import-time use of the variable:
robosuite reads it again in ``EGLGLContext.__init__``, long after the import, so
hiding it for the duration of that one module leaves device selection intact.

The shim is installed from :mod:`rlinf.envs`, which every process that loads a
simulator imports first -- including the simulator subprocesses LIBERO spawns,
which start a fresh interpreter and inherit only the environment variables. This
module deliberately imports nothing beyond the standard library so that those
subprocesses stay cheap to start.
"""

from __future__ import annotations

import importlib.machinery
import os
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Optional, Sequence

_ROBOSUITE_BINDINGS_MODULE = "robosuite.utils.binding_utils"

_MUJOCO_EGL_DEVICE_ID_ENV = "MUJOCO_EGL_DEVICE_ID"


class _EGLDeviceHidingLoader(Loader):
    """Runs a module with ``MUJOCO_EGL_DEVICE_ID`` temporarily unset."""

    def __init__(self, loader: Loader) -> None:
        self._loader = loader

    def create_module(self, spec: ModuleSpec) -> Optional[ModuleType]:
        """Delegate module creation to the wrapped loader."""
        return self._loader.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        """Execute the module without the EGL device visible to it."""
        device_id = os.environ.pop(_MUJOCO_EGL_DEVICE_ID_ENV, None)
        try:
            self._loader.exec_module(module)
        finally:
            if device_id is not None:
                os.environ[_MUJOCO_EGL_DEVICE_ID_ENV] = device_id

    def __getattr__(self, name: str):
        """Expose the wrapped loader's remaining API (``get_source``, ...)."""
        return getattr(self._loader, name)


class _RobosuiteBindingsFinder(MetaPathFinder):
    """Claims robosuite's MuJoCo bindings so they load under the shim."""

    def find_spec(
        self,
        fullname: str,
        path: Optional[Sequence[str]] = None,
        target: Optional[ModuleType] = None,
    ) -> Optional[ModuleSpec]:
        """Return a spec that hides the EGL device, for that one module only."""
        if fullname != _ROBOSUITE_BINDINGS_MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _EGLDeviceHidingLoader(spec.loader)
        # The check runs once per process, so stop inspecting further imports.
        if self in sys.meta_path:
            sys.meta_path.remove(self)
        return spec


def install_robosuite_egl_device_shim() -> None:
    """Let robosuite import while ``MUJOCO_EGL_DEVICE_ID`` holds an EGL index.

    Idempotent, and a no-op for every process that never imports robosuite.
    """
    if any(isinstance(finder, _RobosuiteBindingsFinder) for finder in sys.meta_path):
        return
    sys.meta_path.insert(0, _RobosuiteBindingsFinder())
