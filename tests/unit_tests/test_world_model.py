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

"""World-model operator compatibility and optional accelerator support."""

import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from rlinf.envs.sim.world_model.ascend_patch import install_ascend_patch
from rlinf.envs.sim.world_model.ascend_patch import wan_video_dit as wan_ops
from rlinf.utils.patcher import Patcher


@pytest.fixture
def wan_module(monkeypatch):
    """Supply the optional diffsynth dependency at its module boundary."""
    package = ModuleType("diffsynth")
    models = ModuleType("diffsynth.models")
    dit = ModuleType("diffsynth.models.wan_video_dit")
    package.models = models
    models.wan_video_dit = dit
    for name in ("flash_attention", "rope_apply", "RMSNorm"):
        setattr(dit, name, type(name, (), {}))
    for module in (package, models, dit):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        "rlinf.utils.logging.get_logger", lambda: logging.getLogger(__name__)
    )
    Patcher.clear()
    yield dit
    Patcher.clear()


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_wan_patch_leaves_other_devices_unchanged(wan_module, device_type):
    originals = vars(wan_module).copy()
    install_ascend_patch(torch.device(device_type))
    assert vars(wan_module) == originals


def test_wan_patch_missing_dependency(wan_module, monkeypatch, caplog):
    originals = vars(wan_module).copy()
    monkeypatch.setattr(wan_ops, "MINDIESD_ENABLE", False)
    install_ascend_patch(SimpleNamespace(type="npu"))
    assert vars(wan_module) == originals
    assert "keeping the original diffsynth operators" in caplog.text


def test_wan_patch_missing_operator_is_atomic(wan_module, monkeypatch, caplog):
    del wan_module.rope_apply
    originals = vars(wan_module).copy()
    monkeypatch.setattr(wan_ops, "MINDIESD_ENABLE", True)
    install_ascend_patch(SimpleNamespace(type="npu"))
    assert vars(wan_module) == originals
    assert "rope_apply" in caplog.text


def test_wan_patch_can_be_installed_twice(wan_module, monkeypatch):
    monkeypatch.setattr(wan_ops, "MINDIESD_ENABLE", True)
    for _ in range(2):
        install_ascend_patch(SimpleNamespace(type="npu"))
        for name in ("flash_attention", "rope_apply", "RMSNorm"):
            assert getattr(wan_module, name) is getattr(wan_ops, name)


@pytest.mark.parametrize("compatibility_mode", [False, True])
@pytest.mark.parametrize("mindiesd_available", [False, True])
def test_wan_cpu_attention_matches_sdpa(
    monkeypatch, compatibility_mode, mindiesd_available
):
    monkeypatch.setattr(wan_ops, "MINDIESD_ENABLE", mindiesd_available)
    for name in (
        "FLASH_ATTN_3_AVAILABLE",
        "FLASH_ATTN_2_AVAILABLE",
        "SAGE_ATTN_AVAILABLE",
    ):
        monkeypatch.setattr(wan_ops, name, False)
    q, k, v = [torch.randn(2, seq, 24) for seq in (7, 5, 5)]
    expected = F.scaled_dot_product_attention(
        *[rearrange(t, "b s (h d) -> b h s d", h=3) for t in (q, k, v)]
    )
    expected = rearrange(expected, "b h s d -> b s (h d)")
    actual = wan_ops.flash_attention(q, k, v, 3, compatibility_mode)
    torch.testing.assert_close(actual, expected)


def _rope_reference(x, freqs, heads):
    pairs = x.double().reshape(x.shape[0], x.shape[1], heads, -1, 2)
    return (
        torch.view_as_real(torch.view_as_complex(pairs) * freqs).flatten(2).to(x.dtype)
    )


@pytest.mark.parametrize("mindiesd_available", [False, True])
def test_wan_cpu_rope_and_rmsnorm(monkeypatch, mindiesd_available):
    monkeypatch.setattr(wan_ops, "MINDIESD_ENABLE", mindiesd_available)
    x = torch.randn(2, 7, 24, dtype=torch.bfloat16)
    angles = torch.randn(7, 1, 4, dtype=torch.float64)
    freqs = torch.polar(torch.ones_like(angles), angles)
    torch.testing.assert_close(
        wan_ops.rope_apply(x, freqs, 3), _rope_reference(x, freqs, 3)
    )
    norm = wan_ops.RMSNorm(24, eps=1e-6).to(dtype=x.dtype)
    norm.weight.data.uniform_(0.5, 1.5)
    expected = x.float() * torch.rsqrt(
        x.float().square().mean(-1, keepdim=True) + norm.eps
    )
    torch.testing.assert_close(norm(x), expected.to(x.dtype) * norm.weight)
    assert list(norm.state_dict()) == ["weight"]
