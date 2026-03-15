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

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


def dict_apply(func, d):
    """
    Apply a function to all values in a dictionary recursively.
    If the value is a dictionary, it will apply the function to its values.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            dict_apply(func, value)
        else:
            d[key] = func(value)
    return d


class Normalizer:
    def __init__(
        self,
        norm_stats: dict[str, dict[str, np.ndarray]],
        from_file: bool = False,
        data_type: str | None = None,
        norm_type: dict[str, str] | None = None,
    ):
        if from_file:
            if data_type == "libero":
                norm_stats["state"]["mean"] = np.array(norm_stats["state"]["mean"][:8])
                norm_stats["state"]["std"] = np.array(norm_stats["state"]["std"][:8])
                norm_stats["actions"]["mean"] = np.array(
                    norm_stats["actions"]["mean"][:7]
                )
                norm_stats["actions"]["std"] = np.array(
                    norm_stats["actions"]["std"][:7]
                )
            elif data_type == "robotwin":
                norm_stats["observation.state"], norm_stats["action"] = {}, {}
                norm_stats["observation.state"]["q01"] = np.array(
                    norm_stats["observation.state.arm.position"]["q01"][:6]
                    + norm_stats["observation.state.effector.position"]["q01"][:1]
                    + norm_stats["observation.state.arm.position"]["q01"][6:]
                    + norm_stats["observation.state.effector.position"]["q01"][1:]
                )
                norm_stats["observation.state"]["q99"] = np.array(
                    norm_stats["observation.state.arm.position"]["q99"][:6]
                    + norm_stats["observation.state.effector.position"]["q99"][:1]
                    + norm_stats["observation.state.arm.position"]["q99"][6:]
                    + norm_stats["observation.state.effector.position"]["q99"][1:]
                )
                norm_stats["action"]["q01"] = np.array(
                    norm_stats["action.arm.position"]["q01"][:6]
                    + norm_stats["action.effector.position"]["q01"][:1]
                    + norm_stats["action.arm.position"]["q01"][6:]
                    + norm_stats["action.effector.position"]["q01"][1:]
                )
                norm_stats["action"]["q99"] = np.array(
                    norm_stats["action.arm.position"]["q99"][:6]
                    + norm_stats["action.effector.position"]["q99"][:1]
                    + norm_stats["action.arm.position"]["q99"][6:]
                    + norm_stats["action.effector.position"]["q99"][1:]
                )
            elif data_type == "customized":
                for key in norm_stats:
                    if isinstance(norm_stats[key], dict):
                        for sub_key in norm_stats[key]:
                            norm_stats[key][sub_key] = np.array(
                                norm_stats[key][sub_key]
                            )
            self.norm_stats = norm_stats
        else:
            self.norm_stats = dict_apply(lambda x: x.astype(np.float32), norm_stats)
        self.norm_type = norm_type or {}
        self.from_file = from_file

    def normalize(self, data: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        normalized_data = {}
        for key, value in data.items():
            if key in self.norm_stats:
                norm_type = self.norm_type.get(key, "identity")
                if norm_type == "meanstd":
                    mean = self.norm_stats[key]["mean"]
                    std = self.norm_stats[key]["std"]
                    normalized_value = (value - mean) / (std + 1e-6)
                elif norm_type == "bounds_99_woclip":
                    low = self.norm_stats[key]["q01"]
                    high = self.norm_stats[key]["q99"]
                    normalized_value = (value - low) / (high - low + 1e-6) * 2.0 - 1.0
                elif norm_type == "std":
                    std = self.norm_stats[key]["std"]
                    normalized_value = value / (std + 1e-6)
                elif norm_type == "minmax":
                    min_val = self.norm_stats[key]["min"]
                    max_val = self.norm_stats[key]["max"]
                    normalized_value = (value - min_val) / (
                        max_val - min_val + 1e-6
                    ) * 2 - 1
                elif norm_type == "identity":
                    normalized_value = value
                else:
                    raise ValueError(
                        f"Unknown normalization type: {norm_type}. Supported types are 'meanstd', 'minmax', and 'identity'."
                    )
                normalized_data[key] = normalized_value
            else:
                # If the key is not in norm_stats, we assume no normalization is needed
                normalized_data[key] = value
        return normalized_data

    def unnormalize(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Unnormalize the given data using stored normalization statistics.

        Args:
            data (Dict[str, np.ndarray]): Dictionary of normalized arrays to unnormalize.

        Returns:
            Dict[str, np.ndarray]: Dictionary of unnormalized arrays.
        """
        unnormalized_data = {}
        for key, value in data.items():
            if key in self.norm_stats:
                norm_type = self.norm_type.get(key, "identity")
                stats = self.norm_stats[key]
                if norm_type == "meanstd":
                    mean = stats["mean"]
                    std = stats["std"]
                    unnormalized_value = value * (std + 1e-6) + mean
                elif norm_type == "bounds_98" or norm_type == "bounds_98_woclip":
                    low = self.norm_stats[key]["q02"]
                    high = self.norm_stats[key]["q98"]
                    unnormalized_value = ((value + 1.0) / 2.0) * (
                        high - low + 1e-6
                    ) + low
                elif norm_type == "bounds_99" or norm_type == "bounds_99_woclip":
                    low = self.norm_stats[key]["q01"]
                    high = self.norm_stats[key]["q99"]
                    unnormalized_value = ((value + 1.0) / 2.0) * (
                        high - low + 1e-6
                    ) + low
                elif norm_type == "std":
                    std = stats["std"]
                    unnormalized_value = value * (std + 1e-6)
                elif norm_type == "minmax":
                    min_val = stats["min"]
                    max_val = stats["max"]
                    # Reverse: (x + 1)/2 * (max-min+eps) + min
                    unnormalized_value = (value + 1) / 2.0 * (
                        max_val - min_val + 1e-6
                    ) + min_val
                elif norm_type == "identity":
                    unnormalized_value = value
                else:
                    raise ValueError(
                        f"Unknown normalization type: {norm_type}. Supported types are 'meanstd', 'minmax', and 'identity'."
                    )
                unnormalized_data[key] = unnormalized_value
            else:
                # If no normalization was applied, return as-is
                unnormalized_data[key] = value
        return unnormalized_data


def resize_with_pad_item(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 3:
        raise ValueError(f"(c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[1:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img.unsqueeze(0),
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def prepare_images(
    config, image_processor, observation: dict[str, Tensor], use_depth_align=False
):
    """Normalize, resize, and pad images and stack them into a tensor.

    Args:
        observation (dict[str, Tensor])

    Returns:
        images (torch.Tensor): (*b, n, c, h, w) images in range [-1.0, 1.0]
        img_masks (torch.Tensor): (*b, n) masks for images, True if image is present, False if missing
    """
    dtype = observation["state"].dtype
    images, img_masks = [], []
    if use_depth_align:
        pil_images = []

    for key in IMAGE_KEYS:
        if key in observation["image"]:
            # resize, pad, and normalize
            img = observation["image"][key]
            assert img.ndim == 3, f"Expected 3D image, got {img.shape}"
            pil_img = img.cpu().numpy()
            if image_processor is None:
                img = img.to(dtype) / 127.5 - 1.0  # to [-1, 1]
                img = resize_with_pad_item(
                    img, *config.resize_imgs_with_padding, pad_value=-1.0
                )
            else:
                img = resize_with_pad_item(
                    img, *config.resize_imgs_with_padding, pad_value=0
                )
                img = image_processor(img)["pixel_values"]
            images.append(img)
            img_masks.append(True)
            if use_depth_align:
                pil_images.append(pil_img)
        else:
            # zero padding
            if image_processor is None:
                img = torch.full_like(img, fill_value=-1.0)
                if use_depth_align:
                    pil_img = torch.full_like(pil_img, fill_value=-1.0)
            else:
                img = np.zeros_like(img)
                if use_depth_align:
                    pil_img = np.zeros_like(pil_img)
            images.append(img)
            if use_depth_align:
                pil_images.append(pil_img)
            img_masks.append(False)
    if isinstance(images[0], torch.Tensor):
        images = torch.stack(images, dim=0)  # (n, c, h, w)
    elif isinstance(images[0], np.ndarray):
        images = torch.from_numpy(np.stack(images, axis=0))  # (n, c, h, w)
    img_masks = torch.tensor(img_masks, dtype=torch.bool)  # (*n)

    if use_depth_align:
        pil_images = torch.from_numpy(np.stack(pil_images, axis=0))  # (n, c, h, w)
    else:
        pil_images = []

    return images, img_masks, pil_images


def prepare_state(config, observation: dict[str, Tensor]):
    """Pad the state to the maximum state dimension.

    Args:
        observation (dict[str, Tensor])

    Returns:
        state (torch.Tensor): (*b, max_state_dim) padded state tensor
    """
    state = observation["state"]
    state = F.pad(state, (0, config.max_state_dim - state.shape[-1]))
    return state


def prepare_action(config, observation: dict[str, Tensor]):
    """Pad the action to the maximum action dimension.

    Args:
        observation (dict[str, Tensor])

    Returns:
        action (torch.Tensor): (*b, n, max_action_dim) padded action tensor
        action_dim (int): the actual dimension of the action before padding
    """
    # ipdb.set_trace()
    action = observation["action"]
    action = F.pad(action, (0, config.max_action_dim - action.shape[-1]))
    return action


def prepare_language(config, language_tokenizer, observation: dict[str, Tensor]):
    """If `prompt` is provided, modify it to PaliGemma format and tokenize it.
    If `lang_tokens` and `lang_masks` are provided, use them directly.

    PaliGemma expects prefix prompts to be formatted as:
    <images> .... <images> <bos> prompt <sep>, where <sep> uses `\\n`.
    So here we format the prompt to start with `<bos>` and end with `\\n`.
    Later, we will concatenate the images and language tokens into a single sequence.

    Args:
        observation (dict[str, Tensor])

    Returns:
        lang_tokens (torch.Tensor): (*b, l) language tokens
        lang_masks (torch.Tensor): (*b, l) masks for language tokens, True if token is present, False if missing
    """
    lang_tokens = observation.get("lang_tokens", None)
    lang_masks = observation.get("lang_masks", None)
    prompt = observation.get("prompt", None)

    # either provide `prompt` or (`lang_tokens`, `lang_masks`)
    if prompt is None and (lang_tokens is None or lang_masks is None):
        raise ValueError(
            "Either 'prompt' or ('lang_tokens', 'lang_masks') must be provided in the observation."
        )

    device = observation["state"].device
    if prompt is not None and (lang_tokens is None or lang_masks is None):
        prompt = [p if p.startswith("<bos>") else f"<bos>{p}" for p in prompt]
        prompt = [p if p.endswith("\n") else f"{p}\n" for p in prompt]
        tokenized_prompt = language_tokenizer.__call__(
            prompt,
            padding="max_length",
            padding_side="right",
            max_length=config.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        lang_masks = tokenized_prompt["attention_mask"].to(
            device=device, dtype=torch.bool
        )
    else:
        lang_tokens = observation["lang_tokens"].to(device=device)
        lang_masks = observation["lang_masks"].to(device=device, dtype=torch.bool)

    return lang_tokens.squeeze(0), lang_masks.squeeze(0)
