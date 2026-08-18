# Copyright 2026 The RLinf Authors.
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

"""RLT configuration and checkpoint helpers for OpenPI_RLinf."""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

from rlinf.utils.logging import get_logger

logger = get_logger()

FULL_WEIGHTS_CANDIDATES = (
    "actor/model_state_dict/full_weights.pt",
    "model_state_dict/full_weights.pt",
    "full_weights.pt",
)

_FSDP_WRAPPER_PREFIXES = (
    "_fsdp_wrapped_module.",
    "_orig_mod.",
    "module.",
)

_BARE_PI0_PREFIXES = (
    "llm.",
    "img.",
    "action_in_proj.",
    "action_out_proj.",
    "time_mlp_in.",
    "time_mlp_out.",
    "state_proj.",
    "action_time_mlp_in.",
    "action_time_mlp_out.",
    "pointnet.",
)

_OLD_OPENPI_PREFIX = "paligemma_with_expert."


@dataclasses.dataclass(frozen=True)
class OpenPiPytorchRLTConfig:
    """RLT-token knobs shared by the SFT and eval wrappers."""

    use_rlt: bool = False
    rlt_alpha: float = 1.0
    rlt_input_dim: int = 2048
    rlt_embed_dim: int = 2048
    rlt_prefix_seq_len: int = 768
    rlt_num_layers: int = 2
    rlt_num_heads: int = 8
    rlt_mlp_ratio: float = 4.0
    rlt_image_only: bool = True
    rlt_use_mask: bool = False


def build_rlt_config(model_cfg: Any) -> OpenPiPytorchRLTConfig:
    """Build optional RLT-token config from ``actor.model.openpi``."""
    from omegaconf import OmegaConf

    return OpenPiPytorchRLTConfig(
        use_rlt=bool(OmegaConf.select(model_cfg, "use_rlt", default=False)),
        rlt_alpha=float(OmegaConf.select(model_cfg, "rlt_alpha", default=1.0)),
        rlt_input_dim=int(OmegaConf.select(model_cfg, "rlt_input_dim", default=2048)),
        rlt_embed_dim=int(OmegaConf.select(model_cfg, "rlt_embed_dim", default=2048)),
        rlt_prefix_seq_len=int(
            OmegaConf.select(model_cfg, "rlt_prefix_seq_len", default=768)
        ),
        rlt_num_layers=int(OmegaConf.select(model_cfg, "rlt_num_layers", default=2)),
        rlt_num_heads=int(OmegaConf.select(model_cfg, "rlt_num_heads", default=8)),
        rlt_mlp_ratio=float(OmegaConf.select(model_cfg, "rlt_mlp_ratio", default=4.0)),
        rlt_image_only=bool(
            OmegaConf.select(model_cfg, "rlt_image_only", default=True)
        ),
        rlt_use_mask=bool(OmegaConf.select(model_cfg, "rlt_use_mask", default=False)),
    )


def resolve_model_safetensors(model_path: Any) -> pathlib.Path | None:
    """Resolve a base ``model.safetensors`` checkpoint path."""
    path = pathlib.Path(model_path).expanduser()
    if path.is_file() and path.name.endswith(".safetensors"):
        return path
    weights_path = path / "model.safetensors"
    return weights_path if weights_path.exists() else None


def resolve_full_weights(model_path: Any) -> pathlib.Path | None:
    """Resolve an RLinf FSDP ``full_weights.pt`` checkpoint path."""
    path = pathlib.Path(model_path).expanduser()
    if path.is_file() and path.name.endswith(".pt"):
        return path
    for rel_path in FULL_WEIGHTS_CANDIDATES:
        candidate = path / rel_path
        if candidate.exists():
            return candidate
    return None


def _normalize_wrapper_key(key: str) -> str:
    while True:
        for prefix in _FSDP_WRAPPER_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        else:
            return key


def _normalize_wrapper_state_dict(state_dict):
    normalized = {}
    for key, tensor in state_dict.items():
        key = _normalize_wrapper_key(key)
        if key in normalized:
            raise ValueError(
                f"Duplicate checkpoint key after prefix normalization: {key!r}."
            )
        normalized[key] = tensor

    has_wrapper_key = any(
        key.startswith("model.") or key.startswith("rlt_module.") for key in normalized
    )
    if has_wrapper_key:
        return normalized

    if any(key.startswith(_BARE_PI0_PREFIXES) for key in normalized):
        return {f"model.{key}": tensor for key, tensor in normalized.items()}
    return normalized


def load_full_wrapper_weights(wrapper, weights_path, *, expect_rlt: bool) -> None:
    """Load an RLinf full-wrapper checkpoint into an OpenPI_RLinf wrapper."""
    import torch

    from rlinf.utils.ckpt_convertor.openpi._core import as_state_dict

    loaded = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    state_dict = _normalize_wrapper_state_dict(as_state_dict(loaded))
    if expect_rlt and not any(key.startswith("rlt_module.") for key in state_dict):
        raise ValueError(
            "openpi_rlinf RLT checkpoint has no rlt_module.* weights. "
            "Stage2 must consume a Stage1 checkpoint trained with openpi.use_rlt=True."
        )

    incompatible = wrapper.load_state_dict(state_dict, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = list(incompatible.missing_keys)
    matched = len(state_dict) - len(unexpected)
    if matched <= 0:
        raise RuntimeError(
            f"No tensors from {weights_path} matched the openpi_rlinf wrapper. "
            "This usually means the checkpoint is still in the legacy official "
            "OpenPI PyTorch key layout."
        )
    if expect_rlt and any(key.startswith("rlt_module.") for key in missing):
        raise RuntimeError(
            f"RLT checkpoint {weights_path} did not load all rlt_module weights; "
            f"missing={missing[:8]}"
        )

    if missing or unexpected:
        logger.warning(
            "openpi_rlinf: loaded wrapper checkpoint %s with strict=False "
            "(matched=%d missing=%d unexpected=%d)",
            weights_path,
            matched,
            len(missing),
            len(unexpected),
        )
    else:
        logger.info(
            "openpi_rlinf: loaded full wrapper checkpoint from %s", weights_path
        )


def load_base_safetensors(model, safetensors_path) -> None:
    """Load a base checkpoint, accepting new and legacy OpenPI layouts."""
    import safetensors.torch

    state_dict = safetensors.torch.load_file(str(safetensors_path), device="cpu")
    if any(key.startswith(_OLD_OPENPI_PREFIX) for key in state_dict):
        from rlinf.utils.ckpt_convertor.openpi.openpi_pytorch_to_openpi_rlinf import (
            old_to_new_state_dict,
        )

        state_dict = old_to_new_state_dict(state_dict)
        logger.info(
            "openpi_rlinf: converted OpenPI PyTorch checkpoint keys from %s in memory",
            safetensors_path,
        )
    model.load_state_dict(state_dict, strict=True)
