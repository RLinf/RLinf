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

"""Data conversion and rollout-cache utilities for the starVLA RLinf wrapper."""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import torch
from PIL import Image

from .profile import resolve_vlm_interface


def tensor_to_numpy_compatible(tensor: torch.Tensor) -> np.ndarray:
    """Detach tensor and convert to NumPy (with bf16-safe cast)."""
    t = tensor.detach().cpu()
    if t.dtype == torch.bfloat16:
        t = t.to(dtype=torch.float32)
    return t.numpy()


def build_examples_from_env_obs(
    env_obs: dict[str, Any],
    state_adapter_name: str,
    prepare_state_tensor: Callable[..., Optional[torch.Tensor]],
    include_state: bool = True,
) -> list[dict[str, Any]]:
    """Convert env observations into starVLA 'examples' format."""
    main_images = env_obs["main_images"]
    extra_view_images = env_obs.get("extra_view_images")
    wrist_images = env_obs.get("wrist_images")
    task_desc = env_obs.get("task_descriptions")
    states = env_obs.get("states")

    def to_numpy(x: Any) -> np.ndarray:
        return tensor_to_numpy_compatible(x) if torch.is_tensor(x) else np.asarray(x)

    def to_pil(img_arr: np.ndarray) -> Image.Image:
        arr = img_arr
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    main_images_np = to_numpy(main_images)
    extra_view_images_np = (
        to_numpy(extra_view_images) if extra_view_images is not None else None
    )
    wrist_images_np = to_numpy(wrist_images) if wrist_images is not None else None

    examples: list[dict[str, Any]] = []
    for i in range(main_images_np.shape[0]):
        views: list[Image.Image] = [to_pil(main_images_np[i])]
        for stack in (extra_view_images_np, wrist_images_np):
            if stack is None:
                continue
            arr = stack[i]
            if arr.ndim == 4:
                views.extend(to_pil(arr[v]) for v in range(arr.shape[0]))
            elif arr.ndim == 3:
                views.append(to_pil(arr))

        sample: dict[str, Any] = {
            "image": views,
            "lang": "" if task_desc is None else str(task_desc[i]),
        }

        if include_state and states is not None:
            state_i = prepare_state_tensor(
                states[i],
                state_adapter_name=state_adapter_name,
                context="build_examples_from_env_obs",
            )
            if state_i is not None:
                sample["state"] = tensor_to_numpy_compatible(state_i)[0]
        examples.append(sample)
    return examples


def get_scalar(value: Any, default: Any, cast):
    """Extract scalar from python/tensor value with fallback."""
    if value is None:
        return default
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        return cast(value.reshape(-1)[0].item())
    return cast(value)


def collect_tensor_inputs(
    data: dict[str, Any],
    skip_keys: set[str],
    ignored_keys: set[str],
) -> dict[str, torch.Tensor]:
    """Collect model-input tensors while skipping rollout-only keys."""
    skip_all = set(skip_keys) | ignored_keys
    return {
        key: value
        for key, value in data.items()
        if key not in skip_all and isinstance(value, torch.Tensor)
    }


def forward_input_check(data: dict[str, torch.Tensor]) -> None:
    """Validate replay batch has mandatory action/prompt tensors."""
    if "action" not in data:
        raise KeyError(
            "Missing 'action' in training batch. Rollout must store forward_inputs['action']."
        )
    if "input_ids" not in data or "attention_mask" not in data:
        raise KeyError(
            "Missing prompt inputs ('input_ids'/'attention_mask') in training batch. "
            "Rollout must cache VLM prompt tensors in forward_inputs."
        )


_PACKABLE_MODEL_INPUT_KEYS: tuple[tuple[str, str], ...] = (
    ("pixel_values", "pixel_values_lens"),
    ("image_grid_thw", "image_grid_thw_lens"),
    ("pixel_values_videos", "pixel_values_videos_lens"),
    ("video_grid_thw", "video_grid_thw_lens"),
    ("second_per_grid_ts", "second_per_grid_ts_lens"),
)


def restore_pixel_values_for_forward(
    model_inputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Unpack packed multi-image/video tensors before default_forward."""
    restored = dict(model_inputs)
    for key, lens_key in _PACKABLE_MODEL_INPUT_KEYS:
        lens = restored.pop(lens_key, None)
        tensor = restored.get(key)
        if not isinstance(tensor, torch.Tensor):
            continue
        if not isinstance(lens, torch.Tensor):
            continue
        if tensor.ndim < 2:
            continue

        lens_flat = lens.reshape(-1).to(dtype=torch.long)
        if lens_flat.numel() != tensor.shape[0]:
            continue
        if torch.any(lens_flat <= 0):
            raise RuntimeError(f"Invalid '{lens_key}': non-positive token count.")

        max_len = int(lens_flat.max().item())
        if tensor.shape[1] < max_len:
            raise RuntimeError(
                f"Invalid packed '{key}': second dim smaller than '{lens_key}'."
            )

        if torch.all(lens_flat == lens_flat[0]):
            per_sample = int(lens_flat[0].item())
            if tensor.shape[1] > per_sample:
                tensor = tensor[:, :per_sample]
            restored[key] = tensor.reshape(
                tensor.shape[0] * per_sample,
                *tensor.shape[2:],
            )
            continue

        pieces: list[torch.Tensor] = []
        for i, n in enumerate(lens_flat.tolist()):
            pieces.append(tensor[i, : int(n)])
        restored[key] = torch.cat(pieces, dim=0)
    return restored


def collect_default_forward_model_inputs(
    data: dict[str, Any],
    *,
    skip_keys: set[str],
    ignored_keys: set[str],
) -> dict[str, torch.Tensor]:
    """Collect and restore default_forward model inputs from replay batch."""
    model_inputs = collect_tensor_inputs(
        data,
        skip_keys=skip_keys,
        ignored_keys=ignored_keys,
    )
    return restore_pixel_values_for_forward(model_inputs)


def fetch_action_for_logprob_for_default_forward(
    policy,
    *,
    data: dict[str, Any],
    reference: torch.Tensor,
) -> torch.Tensor:
    """Resolve action tensor for default_forward.

    Preferred path: use rollout-cached ``action_for_logprob`` directly so training
    avoids extra env<->model remapping. No fallback is provided if the key is missing
    """
    action_for_logprob = data.get("action_for_logprob")
    if isinstance(action_for_logprob, torch.Tensor):
        if action_for_logprob.ndim == 2:
            bsz = action_for_logprob.shape[0]
            action_for_logprob = action_for_logprob.view(
                bsz, policy.num_action_chunks, policy.action_dim
            )
        elif action_for_logprob.ndim != 3:
            raise ValueError(
                "Expected 'action_for_logprob' [B, T*D] or [B,T,D], "
                f"got {action_for_logprob.shape}"
            )
        return action_for_logprob.to(device=reference.device, dtype=reference.dtype)
    else:
        raise KeyError(
            "Missing 'action_for_logprob' in training batch. Rollout must store forward_inputs['action_for_logprob'] for efficient training. "
            "There is no legacy fallback for missing 'action_for_logprob' cache to ensure training logprobs are always computed in the same action space. "
        )


def pack_model_inputs_for_storage(
    model_inputs: dict[str, Any],
    batch_size: int,
) -> dict[str, Any]:
    """Pack variable-length vision tensors into batch-first storage format."""
    packed = dict(model_inputs)
    bsz = int(batch_size)
    if bsz <= 0:
        return packed

    for key, lens_key in _PACKABLE_MODEL_INPUT_KEYS:
        tensor = packed.get(key)
        if not isinstance(tensor, torch.Tensor) or tensor.ndim == 0:
            continue
        if tensor.shape[0] == bsz:
            continue
        if tensor.shape[0] % bsz != 0:
            raise RuntimeError(
                f"Cannot pack '{key}' for rollout storage as batch-first: "
                f"leading_dim={tensor.shape[0]}, batch_size={bsz}. "
                "Expected leading dim to be divisible by batch size."
            )
        per_sample = tensor.shape[0] // bsz
        packed[key] = tensor.reshape(bsz, per_sample, *tensor.shape[1:])
        packed[lens_key] = torch.full(
            (bsz,),
            fill_value=per_sample,
            device=tensor.device,
            dtype=torch.int64,
        )
    return packed


_PAD_2D_KEYS: tuple[str, ...] = (
    "input_ids",
    "attention_mask",
    "position_ids",
    "token_type_ids",
)


def normalize_model_inputs_for_storage(
    model_inputs: dict[str, Any],
    starvla_model: Any,
    rollout_prompt_seq_len: Optional[int],
) -> tuple[dict[str, Any], Optional[int]]:
    """Pad prompt tensors to fixed seq-len so trajectory tensors can stack."""
    input_ids = model_inputs.get("input_ids")
    attention_mask = model_inputs.get("attention_mask")
    if not (
        isinstance(input_ids, torch.Tensor) and isinstance(attention_mask, torch.Tensor)
    ):
        return model_inputs, rollout_prompt_seq_len
    if input_ids.ndim != 2 or attention_mask.ndim != 2:
        return model_inputs, rollout_prompt_seq_len

    bsz = int(input_ids.shape[0])
    seq_len = int(input_ids.shape[1])
    target_len = rollout_prompt_seq_len
    if target_len is None:
        action_cfg = getattr(
            getattr(getattr(starvla_model, "config", None), "framework", None),
            "action_model",
            None,
        )
        for key in (
            "rollout_prompt_seq_len",
            "rollout_prompt_length",
            "prompt_seq_len",
        ):
            value = getattr(action_cfg, key, None) if action_cfg is not None else None
            try:
                value = int(value)
            except (TypeError, ValueError):
                continue
            if value > 0:
                target_len = value
                break
        if target_len is None:
            target_len = int((max(256, seq_len + 64) + 7) // 8 * 8)

    target_len = int(target_len)
    if seq_len == target_len:
        return model_inputs, target_len
    if seq_len > target_len:
        raise RuntimeError(
            "VLM prompt length exceeds fixed rollout prompt length for tensor stacking: "
            f"current={seq_len}, target={target_len}. "
            "Set 'framework.action_model.rollout_prompt_seq_len' in config to a larger value."
        )

    pad_len = target_len - seq_len
    model_cfg = getattr(
        getattr(resolve_vlm_interface(starvla_model), "model", None), "config", None
    )
    try:
        input_pad_id = int(getattr(model_cfg, "pad_token_id", 0) or 0)
    except (TypeError, ValueError):
        input_pad_id = 0

    normalized = dict(model_inputs)
    for key in _PAD_2D_KEYS:
        tensor = normalized.get(key)
        if not (
            isinstance(tensor, torch.Tensor)
            and tensor.ndim == 2
            and tensor.shape[0] == bsz
            and tensor.shape[1] == seq_len
        ):
            continue
        pad_value = input_pad_id if key == "input_ids" else 0
        pad_tensor = torch.full(
            (bsz, pad_len),
            fill_value=pad_value,
            device=tensor.device,
            dtype=tensor.dtype,
        )
        normalized[key] = torch.cat([tensor, pad_tensor], dim=1)

    return normalized, target_len
