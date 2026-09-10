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

import torch

from rlinf.scheduler import Worker
from rlinf.utils.data_iter_utils import _get_runtime_device


class _FakeAcceleratorPlatform:
    def current_device(self):
        return 3


class _FailingAcceleratorPlatform:
    def current_device(self):
        raise AssertionError("accelerator runtime is unavailable")


def test_get_runtime_device_prefers_worker_accelerator_for_cpu_batch(monkeypatch):
    monkeypatch.setattr(
        Worker, "torch_platform", _FakeAcceleratorPlatform(), raising=False
    )
    monkeypatch.setattr(Worker, "torch_device_type", "cuda", raising=False)

    device = _get_runtime_device({"input_ids": torch.ones(2, dtype=torch.long)})

    assert device == torch.device("cuda:3")


def test_get_runtime_device_falls_back_to_batch_device(monkeypatch):
    monkeypatch.setattr(
        Worker, "torch_platform", _FailingAcceleratorPlatform(), raising=False
    )
    monkeypatch.setattr(Worker, "torch_device_type", "cuda", raising=False)

    device = _get_runtime_device({"input_ids": torch.ones(2, dtype=torch.long)})

    assert device == torch.device("cpu")
