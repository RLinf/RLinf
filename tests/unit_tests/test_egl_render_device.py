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

from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
import textwrap

import pytest

import rlinf.scheduler.hardware.accelerators.nvidia_gpu as nvidia_gpu
import rlinf.utils.robosuite_compat as robosuite_compat
from rlinf.scheduler.hardware.accelerators.nvidia_gpu import (
    EGL_DEVICE_ID_ENV_VARS,
    NvidiaGPUManager,
)

MUJOCO_EGL_DEVICE_ID, EGL_DEVICE_ID = EGL_DEVICE_ID_ENV_VARS

_MANAGED_ENV_VARS = ("CUDA_VISIBLE_DEVICES", "MUJOCO_GL", *EGL_DEVICE_ID_ENV_VARS)

# The node from the bug report: nine EGL devices, of which four are GPUs, and an
# EGL enumeration order that does not follow the CUDA one.
_EGL_INDEX_BY_CUDA_ORDINAL = {0: 2, 1: 3, 2: 0, 3: 1}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _MANAGED_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    nvidia_gpu._egl_index_by_cuda_device.cache_clear()
    yield
    nvidia_gpu._egl_index_by_cuda_device.cache_clear()


@pytest.fixture(autouse=True)
def _restore_meta_path():
    saved = list(sys.meta_path)
    yield
    sys.meta_path[:] = saved


@pytest.fixture
def egl_devices(monkeypatch):
    """Enumerate the bug report's EGL topology instead of the real driver."""
    monkeypatch.setattr(
        nvidia_gpu,
        "_query_egl_index_by_cuda_ordinal",
        lambda: dict(_EGL_INDEX_BY_CUDA_ORDINAL),
    )


@pytest.fixture
def no_egl(monkeypatch):
    """A node whose driver cannot be asked about EGL devices."""

    def unavailable():
        raise OSError("libEGL.so.1: cannot open shared object file")

    monkeypatch.setattr(nvidia_gpu, "_query_egl_index_by_cuda_ordinal", unavailable)


def _become_a_rendering_worker(monkeypatch, cuda_device_id="3", egl_device_id="1"):
    """Reproduce a worker whose EGL index and CUDA device id disagree."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cuda_device_id)
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv(MUJOCO_EGL_DEVICE_ID, egl_device_id)


def _accelerator_env_var(monkeypatch, visible_accelerators: list[str]) -> dict:
    monkeypatch.setattr(nvidia_gpu, "_torch_needs_avoid_record_streams", lambda: False)
    return NvidiaGPUManager.get_accelerator_env_var(visible_accelerators)


# ---------------------------------------------------------------------------
# CUDA -> EGL mapping
# ---------------------------------------------------------------------------


def test_every_cuda_device_maps_to_its_own_egl_index(egl_devices):
    resolved = {
        cuda_id: NvidiaGPUManager.get_egl_device_id(cuda_id) for cuda_id in range(4)
    }

    assert resolved == _EGL_INDEX_BY_CUDA_ORDINAL


def test_the_mapping_accepts_the_string_ids_placement_speaks_in(egl_devices):
    assert NvidiaGPUManager.get_egl_device_id("2") == 0


def test_ordinals_are_translated_through_cuda_visible_devices(monkeypatch, egl_devices):
    # EGL_CUDA_DEVICE_NV reports a device's position in CUDA_VISIBLE_DEVICES, so
    # in a process that only sees some GPUs those ordinals are not device ids.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")

    assert NvidiaGPUManager.get_egl_device_id(4) == 2
    assert NvidiaGPUManager.get_egl_device_id(7) == 1
    # Devices this process cannot see have no readable mapping.
    assert NvidiaGPUManager.get_egl_device_id(0) is None


def test_an_unmapped_device_has_no_egl_index(egl_devices):
    assert NvidiaGPUManager.get_egl_device_id(8) is None


def test_a_uuid_device_has_no_egl_index(egl_devices):
    assert NvidiaGPUManager.get_egl_device_id("GPU-05d35c06-da01") is None


def test_the_mapping_is_empty_when_the_driver_cannot_be_queried(no_egl):
    assert NvidiaGPUManager.get_egl_device_id(0) is None


def test_the_driver_is_queried_once(monkeypatch):
    calls = []

    def counting_query():
        calls.append(None)
        return dict(_EGL_INDEX_BY_CUDA_ORDINAL)

    monkeypatch.setattr(nvidia_gpu, "_query_egl_index_by_cuda_ordinal", counting_query)

    NvidiaGPUManager.get_egl_device_id(0)
    NvidiaGPUManager.get_egl_device_id(1)

    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Worker env vars
# ---------------------------------------------------------------------------


def test_the_worker_gets_the_egl_index_of_its_own_gpu(monkeypatch, egl_devices):
    monkeypatch.setenv("MUJOCO_GL", "egl")

    env_vars = _accelerator_env_var(monkeypatch, ["3"])

    assert env_vars["CUDA_VISIBLE_DEVICES"] == "3"
    # A CUDA device id is not an EGL index, so it must not be passed through.
    assert env_vars[MUJOCO_EGL_DEVICE_ID] == "1"
    assert env_vars[EGL_DEVICE_ID] == "1"


def test_a_multi_gpu_worker_renders_on_its_first_gpu(monkeypatch, egl_devices):
    monkeypatch.setenv("MUJOCO_GL", "egl")

    env_vars = _accelerator_env_var(monkeypatch, ["2", "3"])

    assert env_vars[MUJOCO_EGL_DEVICE_ID] == "0"


@pytest.mark.parametrize("backend", ["osmesa", "glx"])
def test_cpu_rendering_sets_no_egl_device(monkeypatch, egl_devices, backend):
    monkeypatch.setenv("MUJOCO_GL", backend)

    env_vars = _accelerator_env_var(monkeypatch, ["3"])

    assert not [name for name in EGL_DEVICE_ID_ENV_VARS if name in env_vars]


@pytest.mark.parametrize("backend", ["", "glfw", "OSMesa"])
def test_backends_robosuite_rewrites_to_egl_get_a_device(
    monkeypatch, egl_devices, backend
):
    # robosuite 1.4.1 forces GPU rendering to EGL for every value that is not
    # literally "osmesa" or "glx", down to the casing, so anything else has to
    # be treated as EGL here too.
    monkeypatch.setenv("MUJOCO_GL", backend)

    env_vars = _accelerator_env_var(monkeypatch, ["3"])

    assert env_vars[MUJOCO_EGL_DEVICE_ID] == "1"


def test_a_worker_without_gpus_gets_no_egl_device(monkeypatch, egl_devices):
    monkeypatch.setenv("MUJOCO_GL", "egl")

    env_vars = _accelerator_env_var(monkeypatch, [])

    assert not [name for name in EGL_DEVICE_ID_ENV_VARS if name in env_vars]


def test_an_unmappable_device_falls_back_to_the_cuda_id(monkeypatch, no_egl):
    # The pre-existing behaviour: right whenever the two namespaces happen to
    # agree, which is the common single-node case.
    monkeypatch.setenv("MUJOCO_GL", "egl")

    env_vars = _accelerator_env_var(monkeypatch, ["3"])

    assert env_vars[MUJOCO_EGL_DEVICE_ID] == "3"
    assert env_vars[EGL_DEVICE_ID] == "3"


# ---------------------------------------------------------------------------
# robosuite import shim
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_robosuite(tmp_path, monkeypatch):
    """A stand-in for robosuite that records the environment it was imported with."""
    bindings = tmp_path / "robosuite" / "utils"
    bindings.mkdir(parents=True)
    (tmp_path / "robosuite" / "__init__.py").write_text("")
    (bindings / "__init__.py").write_text("")
    (bindings / "binding_utils.py").write_text(
        textwrap.dedent(
            """
            import os

            SAW_DEVICE_ID = os.environ.get("MUJOCO_EGL_DEVICE_ID", None)
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    yield tmp_path
    for name in [n for n in sys.modules if n.split(".")[0] == "robosuite"]:
        del sys.modules[name]


def test_the_device_is_hidden_from_the_robosuite_import_check(
    monkeypatch, fake_robosuite
):
    # robosuite 1.4.1 asserts MUJOCO_EGL_DEVICE_ID occurs in
    # CUDA_VISIBLE_DEVICES, which a correct EGL index generally does not.
    _become_a_rendering_worker(monkeypatch)
    robosuite_compat.install_robosuite_egl_device_shim()

    bindings = importlib.import_module("robosuite.utils.binding_utils")

    assert bindings.SAW_DEVICE_ID is None
    assert os.environ[MUJOCO_EGL_DEVICE_ID] == "1"


def test_other_modules_import_with_the_device_visible(monkeypatch, fake_robosuite):
    _become_a_rendering_worker(monkeypatch)
    robosuite_compat.install_robosuite_egl_device_shim()

    assert importlib.import_module("robosuite") is not None
    assert os.environ[MUJOCO_EGL_DEVICE_ID] == "1"


def test_a_failed_robosuite_import_still_restores_the_device(
    monkeypatch, fake_robosuite
):
    _become_a_rendering_worker(monkeypatch)
    (fake_robosuite / "robosuite" / "utils" / "binding_utils.py").write_text(
        "raise RuntimeError('import failed')"
    )
    robosuite_compat.install_robosuite_egl_device_shim()

    with pytest.raises(RuntimeError, match="import failed"):
        importlib.import_module("robosuite.utils.binding_utils")

    assert os.environ[MUJOCO_EGL_DEVICE_ID] == "1"


def test_installing_the_shim_repeatedly_leaves_one_finder():
    # Importing rlinf.envs has already installed it once.
    robosuite_compat.install_robosuite_egl_device_shim()
    robosuite_compat.install_robosuite_egl_device_shim()

    finders = [
        finder
        for finder in sys.meta_path
        if isinstance(finder, robosuite_compat._RobosuiteBindingsFinder)
    ]
    assert len(finders) == 1


def test_importing_the_env_package_installs_the_shim():
    # Simulator subprocesses re-import rlinf.envs in a fresh interpreter and rely
    # on it to install the shim before any simulator is imported.
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import sys

                import rlinf.envs  # noqa: F401
                from rlinf.utils.robosuite_compat import _RobosuiteBindingsFinder

                assert any(
                    isinstance(f, _RobosuiteBindingsFinder) for f in sys.meta_path
                )
                assert "rlinf.scheduler" not in sys.modules, (
                    "the env package must not drag the scheduler into every "
                    "simulator subprocess"
                )
                """
            ),
        ],
        capture_output=True,
        text=True,
    )

    assert child.returncode == 0, child.stderr


@pytest.mark.skipif(
    importlib.util.find_spec("robosuite") is None, reason="robosuite is not installed"
)
def test_real_robosuite_imports_with_a_mismatched_egl_index():
    # The EGL index and the CUDA ordinal disagree here, which is what robosuite
    # 1.4.1 refuses to import with.
    env = os.environ.copy()
    env.update(
        {"CUDA_VISIBLE_DEVICES": "3", "MUJOCO_GL": "osmesa", MUJOCO_EGL_DEVICE_ID: "1"}
    )
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import os

                import rlinf.envs  # installs the shim
                import robosuite.utils.binding_utils  # noqa: F401

                assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "1"
                """
            ),
        ],
        capture_output=True,
        text=True,
        env=env,
    )

    assert child.returncode == 0, child.stderr
