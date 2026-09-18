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

"""Tests for accelerator manager contracts."""

import sys
from types import SimpleNamespace

import pytest

from rlinf.scheduler.hardware.accelerators import (
    AcceleratorManager,
    AcceleratorType,
    AcceleratorUtil,
    BirenSUPAManager,
)
from rlinf.scheduler.hardware.accelerators.biren_supa import _ensure_torch_supa


def test_biren_supa_manager_is_registered():
    assert (
        AcceleratorManager.manager_register[AcceleratorType.BIREN_GPU]
        is BirenSUPAManager
    )
    assert AcceleratorType.BIREN_GPU in AcceleratorUtil.CCL_SUPPORT_LIST


def test_biren_supa_manager_contract(monkeypatch):
    supa = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        get_device_name=lambda index: "BRxxx",
    )
    torch = SimpleNamespace(supa=supa)
    monkeypatch.setitem(sys.modules, "torch_supa", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch", torch)

    assert _ensure_torch_supa()
    assert BirenSUPAManager.get_num_devices() == 2
    assert BirenSUPAManager.get_accelerator_model() == "BRxxx"
    assert BirenSUPAManager.get_torch_platform() is supa
    assert BirenSUPAManager.get_device_type() == "supa"
    assert BirenSUPAManager.get_ccl_backend() == "bccl"
    assert BirenSUPAManager.get_ccl_socket_ifname_env_var() == "BCCL_SOCKET_IFNAME"


def test_biren_supa_visibility(monkeypatch):
    monkeypatch.setenv("SUPA_VISIBLE_DEVICES", "0, 3")
    assert BirenSUPAManager.get_visible_devices() == [0, 3]
    assert BirenSUPAManager.get_accelerator_env_var(["1", "2"]) == {
        "SUPA_VISIBLE_DEVICES": "1,2",
        "RAY_EXPERIMENTAL_NOSET_SUPA_VISIBLE_DEVICES": "1",
    }


def test_biren_supa_visibility_rejects_non_integer(monkeypatch):
    monkeypatch.setenv("SUPA_VISIBLE_DEVICES", "0,invalid")
    with pytest.raises(ValueError, match="integers separated by commas"):
        BirenSUPAManager.get_visible_devices()
