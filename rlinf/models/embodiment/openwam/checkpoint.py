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

"""Temporary checkpoint configuration overrides for OpenWAM loading."""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def _checkpoint_with_encoder_override(
    model_path: str, encoder_model_path: str | None
) -> Iterator[str]:
    """Yield a checkpoint directory with deploy-time encoder fixes applied.

    OpenWAM checkpoints can retain training-host paths and legacy encoder names
    (``vjepa2_1``, ``flux_vae``, ``wan_vae``). A temporary staging directory
    rewrites only those config fields without editing the checkpoint or source
    checkout. All other files are symlinked, so this adds no model-storage cost.
    """
    checkpoint_dir = Path(model_path).expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"OpenWAM checkpoint directory not found: {checkpoint_dir}"
        )

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(checkpoint_dir / "config.yaml")
    encoder_cfg = OmegaConf.select(cfg, "model.video_backbone.encoder", default=None)
    encoder_name = (
        str(getattr(encoder_cfg, "name", "")) if encoder_cfg is not None else ""
    )
    name_aliases = {
        "vjepa2_1": "vjepa21",
        "flux_vae": "flux2_vae",
        "wan_vae": "wan22_vae",
    }
    canonical_name = name_aliases.get(encoder_name, encoder_name)
    needs_staging = encoder_model_path is not None or canonical_name != encoder_name
    if not needs_staging:
        yield model_path
        return
    if encoder_cfg is None:
        raise ValueError(
            "OpenWAM encoder override/alias normalization requires "
            "model.video_backbone.encoder in the checkpoint config"
        )
    if encoder_model_path is not None:
        encoder_dir = Path(encoder_model_path).expanduser().resolve()
        if not encoder_dir.is_dir():
            raise FileNotFoundError(
                f"OpenWAM encoder_model_path is not a directory: {encoder_dir}"
            )
        encoder_cfg.model_path = str(encoder_dir)
    if canonical_name != encoder_name:
        encoder_cfg.name = canonical_name

    with tempfile.TemporaryDirectory(prefix="rlinf-openwam-") as tmp:
        staged = Path(tmp) / checkpoint_dir.name
        staged.mkdir()
        for entry in checkpoint_dir.iterdir():
            if entry.name != "config.yaml":
                os.symlink(
                    entry, staged / entry.name, target_is_directory=entry.is_dir()
                )
        OmegaConf.save(cfg, staged / "config.yaml")
        yield str(staged)
