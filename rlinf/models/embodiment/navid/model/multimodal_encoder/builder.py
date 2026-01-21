#
# Copyright 2023 Haotian Liu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from .eva_vit import EVAVisionTowerLavis


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(
        vision_tower_cfg,
        "mm_vision_tower",
        getattr(vision_tower_cfg, "vision_tower", None),
    )
    image_processor = getattr(
        vision_tower_cfg,
        "image_processor",
        getattr(
            vision_tower_cfg,
            "image_processor",
            "./model_zoo/OpenAI/clip-vit-large-patch14",
        ),
    )

    # Allow overriding vision tower path via env var (useful for local deployments)
    env_override = os.environ.get("NAVID_VISION_TOWER", None)
    if env_override:
        vision_tower = env_override

    # Allow overriding image processor path via env var
    imgproc_override = os.environ.get("NAVID_IMAGE_PROCESSOR", None)
    if imgproc_override:
        image_processor = imgproc_override

    def _normalize_rel(p: str) -> str:
        # Turn "./model_zoo/eva_vit_g.pth" into "model_zoo/eva_vit_g.pth" for joining.
        return p[2:] if p.startswith("./") else p

    def _resolve_path(p: str, *, model_path: str | None) -> str:
        """Resolve a possibly-relative path by searching common candidate roots."""
        if not p:
            return p
        if os.path.exists(p) or os.path.isabs(p):
            return p
        rel = _normalize_rel(p)
        candidates: list[str] = []
        if model_path:
            mp = os.path.abspath(model_path)
            candidates.extend(
                [
                    os.path.join(mp, rel),
                    os.path.join(os.path.dirname(mp), rel),
                    os.path.join(os.path.dirname(os.path.dirname(mp)), rel),
                    os.path.join(
                        os.path.dirname(os.path.dirname(os.path.dirname(mp))), rel
                    ),
                ]
            )
        # relative to this repo's navid package (…/embodiment/navid)
        navid_pkg_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        candidates.append(os.path.join(navid_pkg_dir, rel))
        # relative to CWD
        candidates.append(os.path.abspath(rel))
        # common RLinf layout root
        candidates.append(os.path.join("/data/RLinf/VLN-CE/models", rel))
        for cand in candidates:
            if os.path.exists(cand):
                return cand
        return p

    # Try to resolve relative paths for both vision tower and image processor.
    is_absolute_path_exists = os.path.exists(vision_tower)
    if not is_absolute_path_exists and vision_tower and not os.path.isabs(vision_tower):
        model_path = getattr(vision_tower_cfg, "model_path", None) or getattr(
            vision_tower_cfg, "_model_path_for_vision_tower", None
        )
        vision_tower = _resolve_path(vision_tower, model_path=model_path)
        is_absolute_path_exists = os.path.exists(vision_tower)

    # Resolve image_processor path similarly (it can be a local folder id)
    model_path = getattr(vision_tower_cfg, "model_path", None) or getattr(
        vision_tower_cfg, "_model_path_for_vision_tower", None
    )
    image_processor = _resolve_path(image_processor, model_path=model_path)

    if not is_absolute_path_exists:
        raise ValueError(
            f"Not find vision tower: {vision_tower}. Please ensure the vision tower checkpoint exists."
        )

    if "lavis" in vision_tower.lower() or "eva" in vision_tower.lower():
        return EVAVisionTowerLavis(
            vision_tower, image_processor, args=vision_tower_cfg, **kwargs
        )
    else:
        raise ValueError(f"Unknown vision tower: {vision_tower}")
