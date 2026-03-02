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

from . import action_space as action_space_utils
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
) -> list[dict[str, Any]]:
    """Convert env observations into starVLA 'examples' format."""
    main_images = env_obs["main_images"]
    extra_view_images = env_obs.get("extra_view_images")
    wrist_images = env_obs.get("wrist_images")
    task_desc = env_obs.get("task_descriptions")
    states = env_obs.get("states")

    def to_numpy(x):
        if torch.is_tensor(x):
            return tensor_to_numpy_compatible(x)
        return np.asarray(x)

    def to_pil(img_arr):
        if img_arr.dtype != np.uint8:
            img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
        return Image.fromarray(img_arr)

    def append_views(views: list[Image.Image], arr: Optional[np.ndarray]):
        if arr is None:
            return
        if arr.ndim == 4:
            for v in range(arr.shape[0]):
                views.append(to_pil(arr[v]))
        elif arr.ndim == 3:
            views.append(to_pil(arr))

    main_images_np = to_numpy(main_images)
    extra_view_images_np = (
        to_numpy(extra_view_images) if extra_view_images is not None else None
    )
    wrist_images_np = to_numpy(wrist_images) if wrist_images is not None else None

    examples: list[dict[str, Any]] = []
    for i in range(main_images_np.shape[0]):
        views = [to_pil(main_images_np[i])]
        append_views(
            views, None if extra_view_images_np is None else extra_view_images_np[i]
        )
        append_views(views, None if wrist_images_np is None else wrist_images_np[i])

        sample: dict[str, Any] = {
            "image": views,
            "lang": "" if task_desc is None else str(task_desc[i]),
        }

        if states is not None:
            state_i = states[i]
            state_i = prepare_state_tensor(
                state_i,
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


def build_sampling_param_tensors(
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Build batch-aligned sampling metadata tensors for replay storage."""
    bsz = int(batch_size)
    if bsz <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    def _scalar(value: Any, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(value, dtype=dtype).view(1).repeat(bsz)

    return {
        "do_sample": _scalar(int(do_sample), torch.int64),
        "temperature": _scalar(float(temperature), torch.float32),
        "top_k": _scalar(int(top_k), torch.int64),
        "top_p": _scalar(float(top_p), torch.float32),
    }


def apply_sampling_filters(logits: torch.Tensor, data: dict[str, Any]) -> torch.Tensor:
    """Apply temperature/top-k/top-p filtering to logits."""
    filtered = logits

    temp = get_scalar(data.get("temperature"), default=1.0, cast=float)
    if temp <= 0:
        temp = 1.0
    filtered = filtered / temp

    top_k = get_scalar(data.get("top_k"), default=0, cast=int)
    if top_k > 0:
        k = min(top_k, filtered.size(-1))
        kth = torch.topk(filtered, k=k, dim=-1).values[..., -1]
        neg_inf = torch.tensor(
            float("-inf"), device=filtered.device, dtype=filtered.dtype
        )
        filtered = torch.where(filtered < kth.unsqueeze(-1), neg_inf, filtered)

    top_p = get_scalar(data.get("top_p"), default=1.0, cast=float)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(sorted_probs, dim=-1)

        remove = cumprobs > top_p
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))

        filtered = torch.full_like(filtered, float("-inf"))
        filtered.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)

    return filtered


def collect_tensor_inputs(
    data: dict[str, Any],
    skip_keys: set[str],
    ignored_keys: set[str],
) -> dict[str, torch.Tensor]:
    """Collect model-input tensors while skipping rollout-only keys."""
    skip_all = set(skip_keys) | ignored_keys
    inputs: dict[str, torch.Tensor] = {}
    for key, value in data.items():
        if key in skip_all:
            continue
        if isinstance(value, torch.Tensor):
            inputs[key] = value
    return inputs


def ensure_default_forward_replay_batch(data: dict[str, torch.Tensor]) -> None:
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


def prepare_actions_for_default_forward(
    policy,
    *,
    env_actions: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Convert env actions to model action space and target shape."""
    action = env_actions
    if action.ndim == 2:
        bsz = action.shape[0]
        action = action.view(bsz, policy.num_action_chunks, policy.action_dim)
    elif action.ndim != 3:
        raise ValueError(f"Expected 'action' [B, T*D] or [B,T,D], got {action.shape}")

    action = action.to(device=reference.device, dtype=reference.dtype)
    return action_space_utils.normalize_actions_for_model(
        env_actions=action,
        action_norm_stats=policy._action_norm_stats,
    )


def project_rollout_actions_for_logprob(
    policy,
    *,
    rollout_actions: torch.Tensor,
) -> torch.Tensor:
    """Project rollout actions to the same model space used in training.

    This follows the exact transform chain used by rollout storage + training:
        rollout action -> env action (storage) -> model action (training forward).
    Use this when computing rollout-time ``prev_logprobs`` to ensure old/new
    logprobs are compared in the same action space.
    """
    action_norm_stats = (
        None if policy.disable_action_unnormalization else policy._action_norm_stats
    )

    env_actions_np, _ = action_space_utils.unnormalize_actions_for_env(
        normalized_actions=tensor_to_numpy_compatible(rollout_actions),
        action_norm_stats=action_norm_stats,
        # Avoid duplicate warnings here; the main rollout action path will emit it.
        warned_missing_action_norm_stats=True,
    )
    env_actions_t = torch.from_numpy(env_actions_np).to(
        device=rollout_actions.device,
        dtype=rollout_actions.dtype,
    )
    return action_space_utils.normalize_actions_for_model(
        env_actions=env_actions_t,
        action_norm_stats=policy._action_norm_stats,
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


_PROMPT_LEN_CFG_KEYS: tuple[str, ...] = (
    "rollout_prompt_seq_len",
    "rollout_prompt_length",
    "prompt_seq_len",
)
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
        for key in _PROMPT_LEN_CFG_KEYS:
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
