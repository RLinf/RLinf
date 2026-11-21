# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Small, independent actor/Q Rectify encoders for opt-in Tianji Stage2."""
import torch
from torch import nn

from rlinf.utils.xense_tactile import TACTILE_IMAGE_SHAPE, TACTILE_KEYS, TACTILE_ROLES

ROLES = len(TACTILE_ROLES)
PIXELS = ROLES * 3 * 64 * 64
PACKED_DIM = PIXELS + 2 * ROLES


def pack_tactile(obs: dict, *, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    """Preserve four-role order; uint8 replay pixels become normalized floats."""
    missing = [key for key in TACTILE_KEYS if key not in obs]
    if missing:
        raise ValueError(f"tactile-enabled model needs new tactile replay fields: {missing}")
    image, valid, age = (obs[k] for k in TACTILE_KEYS)
    if image.ndim != 5 or tuple(image.shape[1:]) != TACTILE_IMAGE_SHAPE or image.dtype != torch.uint8:
        raise ValueError(f"tactile_images must be uint8 [B,{ROLES},3,64,64] ({', '.join(TACTILE_ROLES)})")
    if valid.shape != image.shape[:2] or age.shape != image.shape[:2] or valid.dtype != torch.bool:
        raise ValueError(f"tactile_valid/age must be [B,{ROLES}], validity bool")
    if not bool(torch.isfinite(age).all()) or bool(((age < 0) | (age > 100)).any()):
        raise ValueError("tactile_age_ms must be finite and within [0,100]")
    pixels = image.to(device=device, dtype=dtype).flatten(1) / 127.5 - 1.0
    return torch.cat((pixels, valid.to(device=device, dtype=dtype),
                      age.to(device=device, dtype=dtype) / 100.0), dim=-1)


class TactileEncoder(nn.Module):
    """Shared CNN across the tactile roles; separate module per actor/Q/V/prior."""

    def __init__(self, output_dim: int = 64) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
        )
        self.projection = nn.Sequential(nn.Linear(ROLES * (128 + 2), output_dim),
                                        nn.LayerNorm(output_dim), nn.Tanh())

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        if packed.ndim != 2 or packed.shape[-1] != PACKED_DIM:
            raise ValueError("invalid packed tactile feature shape")
        batch = packed.shape[0]
        valid = packed[:, PIXELS:PIXELS + ROLES].reshape(batch, ROLES, 1)
        age = packed[:, -ROLES:].reshape(batch, ROLES, 1) * valid
        images = packed[:, :PIXELS].reshape(batch * ROLES, 3, 64, 64)
        images = images * valid.reshape(batch * ROLES, 1, 1, 1)
        features = self.cnn(images).reshape(batch, ROLES, 128) * valid
        features = torch.cat((features, valid, age), dim=-1).flatten(1)
        return self.projection(features) * valid.any(dim=1).to(features.dtype)
