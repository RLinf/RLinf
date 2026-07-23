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

"""Tests for IntelGPUManager torch.xpu compatibility fallback."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

if "torch" not in sys.modules:
    sys.modules["torch"] = ModuleType("torch")

from rlinf.scheduler.hardware.accelerators.intel_gpu import IntelGPUManager


def test_get_torch_platform_adds_ipc_collect_when_missing(monkeypatch):
    torch_module = ModuleType("torch")
    xpu_platform = SimpleNamespace()
    torch_module.xpu = xpu_platform
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    result = IntelGPUManager.get_torch_platform()

    assert result is xpu_platform
    assert hasattr(xpu_platform, "ipc_collect")
    assert callable(xpu_platform.ipc_collect)
    assert xpu_platform.ipc_collect() is None


def test_get_torch_platform_keeps_existing_ipc_collect(monkeypatch):
    torch_module = ModuleType("torch")
    sentinel = object()

    def _existing_ipc_collect():
        return sentinel

    xpu_platform = SimpleNamespace(ipc_collect=_existing_ipc_collect)
    torch_module.xpu = xpu_platform
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    result = IntelGPUManager.get_torch_platform()

    assert result is xpu_platform
    assert xpu_platform.ipc_collect is _existing_ipc_collect
    assert xpu_platform.ipc_collect() is sentinel
