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

"""Regression tests for OpenPI observation preprocessing augmentations.

Covers issue #1694: with ``train=True`` and ``rng=None`` (what every
training caller passes), the augmentations must draw from the global torch
RNG instead of folding to constants (fixed top-left crop, zero rotation,
skipped color jitter).
"""

from __future__ import annotations

import pytest
import torch
import torchvision.transforms.functional as TF

from rlinf.models.embodiment.openpi.modules.model import (
    IMAGE_RESOLUTION,
    Observation,
    preprocess_observation,
)

pytest.importorskip("torchvision")

B, C = 2, 3
H, W = IMAGE_RESOLUTION


@pytest.fixture()
def observation() -> Observation:
    torch.manual_seed(0)
    v = torch.linspace(-1, 1, H).view(H, 1, 1).expand(H, W, C).clone()
    x = torch.linspace(-1, 1, W).view(1, W, 1).expand(H, W, C)
    img = 0.6 * v + 0.4 * x  # unique 2D gradient pattern, in [-1, 1]
    images = {
        "base_0_rgb": img.unsqueeze(0).expand(B, H, W, C).clone(),
        "left_wrist_0_rgb": img.unsqueeze(0).expand(B, H, W, C).clone(),
        "right_wrist_0_rgb": img.unsqueeze(0).expand(B, H, W, C).clone(),
    }
    masks = {key: torch.ones(B, dtype=torch.bool) for key in images}
    state = torch.arange(B * 8, dtype=torch.float32).view(B, 8) / 16.0
    return Observation(images=images, image_masks=masks, state=state)


def _manual_fixed_top_left_crop(image: torch.Tensor) -> torch.Tensor:
    """Reference for the degenerate augmentation: crop(0, 0, 95%), rotate(0), no jitter."""
    im = image[0] / 2.0 + 0.5  # (H, W, C) in [0, 1]
    im = im.permute(2, 0, 1)  # (C, H, W)
    crop_h, crop_w = int(H * 0.95), int(W * 0.95)
    im = TF.crop(im, 0, 0, crop_h, crop_w)
    im = TF.resize(im, [H, W], antialias=True)
    im = TF.rotate(im, 0.0, fill=0)
    return im.permute(1, 2, 0) * 2.0 - 1.0


def test_train_rng_none_draws_from_global_rng(observation: Observation):
    """Two train calls without rng must differ: augmentation is random."""
    out_a = preprocess_observation(observation, train=True)
    out_b = preprocess_observation(observation, train=True)
    assert not torch.equal(out_a.images["base_0_rgb"], out_b.images["base_0_rgb"])


def test_train_rng_none_is_not_fixed_top_left_crop(observation: Observation):
    """Guard against regressing to the constant-folded crop(0, 0, 95%)."""
    torch.manual_seed(0)
    out = preprocess_observation(observation, train=True)
    reference = _manual_fixed_top_left_crop(observation.images["base_0_rgb"])
    assert not torch.allclose(out.images["base_0_rgb"][0], reference, atol=1e-6)


def test_train_rng_none_applies_jitter_to_wrist(observation: Observation):
    """Wrist cameras skip crop/rotate but must still get color jitter.

    Compared with a loose tolerance: the [0, 1] round-trip alone only
    perturbs values at float32 rounding level (~1e-7), while jitter moves
    pixels by O(0.1).
    """
    torch.manual_seed(0)
    out = preprocess_observation(observation, train=True)
    assert not torch.allclose(
        out.images["left_wrist_0_rgb"][0],
        observation.images["left_wrist_0_rgb"][0],
        atol=0.01,
    )


def test_explicit_generator_reproducibility(observation: Observation):
    """Passing an explicit generator must stay deterministic per seed."""
    gen_a1, gen_a2 = (
        torch.Generator().manual_seed(7),
        torch.Generator().manual_seed(7),
    )
    gen_b = torch.Generator().manual_seed(8)
    out_a1 = preprocess_observation(observation, train=True, rng=gen_a1)
    out_a2 = preprocess_observation(observation, train=True, rng=gen_a2)
    out_b = preprocess_observation(observation, train=True, rng=gen_b)
    assert torch.equal(out_a1.images["base_0_rgb"], out_a2.images["base_0_rgb"])
    assert not torch.equal(out_a1.images["base_0_rgb"], out_b.images["base_0_rgb"])


def test_eval_path_unchanged(observation: Observation):
    """train=False applies no augmentation and must stay byte-identical."""
    out = preprocess_observation(observation, train=False)
    for key, image in observation.images.items():
        assert torch.equal(out.images[key], image)
