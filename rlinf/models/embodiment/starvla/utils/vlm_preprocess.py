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

"""VLM input preprocessing utilities for starVLA rollouts and training."""

from __future__ import annotations

from typing import Any, Optional

import torch
from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils.trainer_tools import (
    resize_images as resize_images,
)

from .profile import resolve_vlm_interface


def get_train_image_size(
    starvla_model: Any,
) -> Optional[tuple[int, int] | int]:
    """Read training image size from checkpoint config if provided.

    Checks ``datasets.vla_data.image_size`` first, then falls back to
    ``datasets.vla_data.default_image_resolution`` (format ``[C, H, W]``)
    which many starVLA training configs use instead.
    """
    cfg = getattr(starvla_model, "config", None)
    if cfg is None:
        return None
    vla_data = getattr(getattr(cfg, "datasets", None), "vla_data", None)
    if vla_data is None:
        return None

    size = getattr(vla_data, "image_size", None)
    if size is not None:
        return size

    default_res = getattr(vla_data, "default_image_resolution", None)
    if default_res is not None:
        try:
            res = list(default_res)
        except (TypeError, ValueError):
            return None
        if len(res) == 3:
            return (int(res[1]), int(res[2]))
        if len(res) == 2:
            return (int(res[0]), int(res[1]))

    return None


_TEMPLATE_CACHE: dict[tuple[str, int], str] = {}


def _prompt_of(starvla_model: Any, instruction: str) -> str:
    vla_data = getattr(
        getattr(getattr(starvla_model, "config", None), "datasets", None),
        "vla_data",
        None,
    )
    cot = getattr(vla_data, "CoT_prompt", None) if vla_data is not None else None
    return cot.replace("{instruction}", instruction) if cot else instruction


def _templated_text(processor: Any, instruction: str, num_images: int) -> str:
    key = (instruction, num_images)
    text = _TEMPLATE_CACHE.get(key)
    if text is None:
        conversation = [
            {
                "role": "user",
                "content": [{"type": "image"}] * num_images
                + [{"type": "text", "text": instruction}],
            }
        ]
        text = processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        if len(_TEMPLATE_CACHE) >= 64:
            _TEMPLATE_CACHE.clear()
        _TEMPLATE_CACHE[key] = text
    return text


def build_device_vlm_inputs(
    starvla_model: Any,
    examples: list[dict[str, Any]],
    iface: Any,
    instructions: Optional[list[str]] = None,
) -> dict[str, torch.Tensor]:
    """Qwen-family inputs from device-tensor views, template cached per task.

    ``instructions`` overrides each example's ``lang`` when an action head
    decorates the prompt (for example OFT's action-token suffix).
    """
    import torchvision.transforms.functional as TVF

    target = get_train_image_size(starvla_model)
    if isinstance(target, int):
        target = (target, target)

    texts, images = [], []
    for i, example in enumerate(examples):
        views = example["image"]
        raw = instructions[i] if instructions is not None else example["lang"]
        prompt = _prompt_of(starvla_model, raw)
        texts.append(_templated_text(iface.processor, prompt, len(views)))
        for img in views:
            chw = img.permute(2, 0, 1) if img.shape[-1] in (1, 3) else img
            if target is not None:
                # Legacy path resizes PIL images (bicubic); match it on device.
                chw = TVF.resize(
                    chw,
                    [target[1], target[0]],
                    interpolation=TVF.InterpolationMode.BICUBIC,
                    antialias=True,
                )
            images.append(chw)

    device = images[0].device if images else None
    if device is not None and device.type == "npu":
        # CANN rejects the >8-dim copies inside the fast image processor, so
        # only the resize runs on the NPU; the patch packing stays on host.
        images = [img.cpu() for img in images]
    inputs = iface.processor(
        text=texts, images=images, padding=True, return_tensors="pt"
    )
    return {
        k: v.to(device) if device is not None and torch.is_tensor(v) else v
        for k, v in inputs.items()
    }


def build_base_vlm_inputs(
    starvla_model: Any,
    *,
    examples: list[dict[str, Any]],
    vlm_type: Optional[str] = None,
    vlm_interface: Any = None,
) -> dict[str, torch.Tensor]:
    """Build backbone-only VLM inputs from rollout examples."""
    iface = vlm_interface or resolve_vlm_interface(starvla_model)
    if (
        examples
        and torch.is_tensor(examples[0]["image"][0])
        and hasattr(iface, "processor")
        and vlm_type not in ("florence",)
    ):
        return build_device_vlm_inputs(starvla_model, examples, iface)

    batch_images = [to_pil_preserve(example["image"]) for example in examples]
    instructions = [example["lang"] for example in examples]

    train_obs_image_size = get_train_image_size(starvla_model)
    if train_obs_image_size:
        batch_images = resize_images(batch_images, target_size=train_obs_image_size)

    if vlm_type == "florence":
        single_image_batch = []
        for idx, views in enumerate(batch_images):
            if not isinstance(views, (list, tuple)) or len(views) == 0:
                raise ValueError(
                    f"Florence backbone expects non-empty image list per sample, got sample index {idx}."
                )
            single_image_batch.append([views[0]])
        batch_images = single_image_batch

    build_inputs = getattr(iface, "build_qwenvl_inputs", None)
    if not callable(build_inputs):
        raise RuntimeError("VLM interface does not provide 'build_qwenvl_inputs(...)'.")
    return dict(
        build_inputs(
            images=batch_images,
            instructions=instructions,
        )
    )
