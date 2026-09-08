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

"""The matmul form of Qwen-VL's patch embedding computes the conv's output."""

import torch
import torch.nn as nn

from rlinf.models.embodiment.reward.qwen3vl_rocm_patch import patch_vision_patch_embed

IN_CHANNELS = 3
TEMPORAL_PATCH = 2
PATCH = 16
EMBED_DIM = 32


class Qwen3VLVisionPatchEmbed(nn.Module):
    """Same shape contract as the transformers module the patch targets."""

    def __init__(self):
        super().__init__()
        kernel_size = [TEMPORAL_PATCH, PATCH, PATCH]
        self.proj = nn.Conv3d(
            IN_CHANNELS, EMBED_DIM, kernel_size=kernel_size, stride=kernel_size
        )

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(
            -1, IN_CHANNELS, TEMPORAL_PATCH, PATCH, PATCH
        )
        return self.proj(hidden_states).view(-1, EMBED_DIM)


class _Tower(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = Qwen3VLVisionPatchEmbed()
        self.other = nn.Linear(EMBED_DIM, EMBED_DIM)


def test_patched_forward_matches_the_convolution():
    torch.manual_seed(0)
    model = _Tower().eval()
    patches = torch.randn(7, IN_CHANNELS * TEMPORAL_PATCH * PATCH * PATCH)

    with torch.no_grad():
        expected = model.patch_embed(patches)
        assert patch_vision_patch_embed(model, force=True) == 1
        actual = model.patch_embed(patches)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected)


def test_patching_leaves_the_parameters_alone():
    model = _Tower()
    before = {name: value.clone() for name, value in model.state_dict().items()}

    patch_vision_patch_embed(model, force=True)

    after = model.state_dict()
    assert sorted(after) == sorted(before)
    for name, value in before.items():
        torch.testing.assert_close(after[name], value)


def test_off_rocm_the_model_is_untouched(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    model = _Tower()

    assert patch_vision_patch_embed(model) == 0
