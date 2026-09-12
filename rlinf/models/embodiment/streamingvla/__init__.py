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

"""Factory and strict checkpoint loading for StreamingVLA SFT."""

from __future__ import annotations

import dataclasses
import pathlib
from typing import TYPE_CHECKING, Any

import safetensors.torch
import torch
from omegaconf import DictConfig

if TYPE_CHECKING:
    from .streamingvla_action_model import StreamingVLAForSFTActionPrediction


def _load_torch_state_dict(path: pathlib.Path) -> dict[str, torch.Tensor]:
    """Load a trusted RLinf tensor-only checkpoint."""
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict) or not all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in state_dict.items()
    ):
        raise TypeError(f"Expected a tensor state dict in {str(path)!r}.")
    return state_dict


def _convert_openpi_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert the legacy OpenPI layout into StreamingVLA's private layout."""
    if not any(key.startswith("paligemma_with_expert.") for key in state_dict):
        return state_dict
    from rlinf.utils.ckpt_convertor.openpi.openpi_pytorch_to_openpi_rlinf import (
        old_to_new_state_dict,
    )

    return old_to_new_state_dict(state_dict)


def load_streamingvla_checkpoint(
    model: StreamingVLAForSFTActionPrediction, checkpoint_dir: str | pathlib.Path
) -> pathlib.Path:
    """Strictly load an OpenPI base or RLinf/FSDP StreamingVLA checkpoint.

    Args:
        model: Newly constructed StreamingVLA SFT wrapper.
        checkpoint_dir: Directory containing ``model.safetensors`` or an RLinf
            ``full_weights.pt`` file.

    Returns:
        The exact weight file that was loaded.

    Raises:
        FileNotFoundError: No supported weight file exists.
        RuntimeError: The checkpoint has missing, unexpected, or mismatched keys.
    """
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    full_weight_candidates = (
        checkpoint_dir / "model_state_dict" / "full_weights.pt",
        checkpoint_dir / "actor" / "model_state_dict" / "full_weights.pt",
    )
    for path in full_weight_candidates:
        if path.is_file():
            try:
                model.load_state_dict(_load_torch_state_dict(path), strict=True)
            except RuntimeError as error:
                raise RuntimeError(
                    f"Strict StreamingVLA checkpoint load failed for {str(path)!r}."
                ) from error
            return path

    safetensors_path = checkpoint_dir / "model.safetensors"
    if not safetensors_path.is_file():
        raise FileNotFoundError(
            f"No StreamingVLA weights found under {str(checkpoint_dir)!r}."
        )
    state_dict = safetensors.torch.load_file(str(safetensors_path), device="cpu")
    if not state_dict:
        raise ValueError(f"StreamingVLA checkpoint {str(safetensors_path)!r} is empty.")

    wrapper_flags = [key.startswith("svla_model.") for key in state_dict]
    if all(wrapper_flags):
        target: torch.nn.Module = model
    elif any(wrapper_flags):
        raise ValueError(
            "StreamingVLA checkpoint mixes wrapper-prefixed and base-model keys."
        )
    else:
        target = model.svla_model
        state_dict = _convert_openpi_state_dict(state_dict)
    try:
        target.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            f"Strict StreamingVLA checkpoint load failed for {str(safetensors_path)!r}."
        ) from error
    return safetensors_path


def get_model(cfg: DictConfig, torch_dtype: Any = None) -> torch.nn.Module:
    """Build a training-only StreamingVLA model from ``actor.model`` config."""
    del torch_dtype
    from .dataconfig import get_streamingvla_config
    from .streamingvla_action_model import StreamingVLAForSFTActionPrediction

    streaming_cfg = cfg.streamingvla
    if not bool(streaming_cfg.get("use_sfp", False)):
        raise ValueError("actor.model.streamingvla.use_sfp must be true.")
    if bool(streaming_cfg.get("train_expert_only", False)):
        raise ValueError(
            "StreamingVLA PR support is full-parameter SFT only; "
            "train_expert_only must be false."
        )

    train_config = get_streamingvla_config(
        str(streaming_cfg.config_name),
        model_path=str(cfg.model_path),
        data_kwargs=streaming_cfg.get("data", None),
    )
    precision = str(cfg.get("precision", "bf16")).lower()
    dtype = (
        "float32" if precision in {"fp32", "float32", "32", "32-true"} else "bfloat16"
    )
    model_config = dataclasses.replace(
        train_config.model,
        dtype=dtype,
        action_horizon=int(streaming_cfg.get("action_horizon", 10)),
        action_dim=int(streaming_cfg.get("model_action_dim", 32)),
    )
    # RLinf's pinned OpenPI Pi0Config predates this StreamingVLA field.
    object.__setattr__(model_config, "use_sfp", True)

    model = StreamingVLAForSFTActionPrediction(
        model_config,
        sigma=float(streaming_cfg.get("sigma", 0.16)),
        noise_decay=float(streaming_cfg.get("noise_decay", 4.0)),
        require_action_states=bool(streaming_cfg.get("use_action_states", True)),
    )

    import openpi.shared.download as download

    checkpoint_dir = download.maybe_download(str(cfg.model_path))
    load_streamingvla_checkpoint(model, checkpoint_dir)
    return model


__all__ = [
    "get_model",
    "load_streamingvla_checkpoint",
]
