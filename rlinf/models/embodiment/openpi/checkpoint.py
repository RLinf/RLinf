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

"""Checkpoint helpers for OpenPI."""

from __future__ import annotations

import json
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
_OLD_WRAPPER_MODEL_PREFIX = "model."


def _missing_pi0_keys(missing_keys) -> list[str]:
    """Return missing keys that belong to the Pi0 backbone, not extra heads."""
    return [key for key in missing_keys if key.startswith(_BARE_PI0_PREFIXES)]


def resolve_model_safetensors(model_path: Any) -> pathlib.Path | None:
    """Resolve a single-file safetensors checkpoint or its shard index."""
    path = pathlib.Path(model_path).expanduser()
    if path.is_file() and path.name.endswith(
        (".safetensors", ".safetensors.index.json")
    ):
        return path
    for filename in ("model.safetensors", "model.safetensors.index.json"):
        weights_path = path / filename
        if weights_path.is_file():
            return weights_path
    return None


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


def _normalize_key(key: str) -> str:
    while True:
        for prefix in _FSDP_WRAPPER_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        else:
            return key


def _normalize_state_dict(state_dict):
    """Map checkpoint keys onto the inherited-Pi0 layout.

    New checkpoints store bare Pi0 keys (``llm.*``). Older wrapper checkpoints
    stored the same tensors under ``model.llm.*``; strip that prefix. RLT and
    extra heads (``rlt_module.*``, ``value_head.*``, …) stay as-is.
    """
    normalized = {}
    for key, tensor in state_dict.items():
        key = _normalize_key(key)
        if key.startswith(_OLD_WRAPPER_MODEL_PREFIX):
            rest = key[len(_OLD_WRAPPER_MODEL_PREFIX) :]
            # Only strip when the remainder is a Pi0 module (or nested FSDP
            # leftover). Algorithm heads never lived under ``model.``.
            if rest.startswith(_BARE_PI0_PREFIXES) or rest.startswith(
                _FSDP_WRAPPER_PREFIXES
            ):
                key = rest
        if key in normalized:
            raise ValueError(
                f"Duplicate checkpoint key after prefix normalization: {key!r}."
            )
        normalized[key] = tensor
    return normalized


def _convert_official_openpi_keys(state_dict, source) -> dict:
    """Rewrite ``paligemma_with_expert.*`` keys onto the inherited-Pi0 layout.

    Extra heads (``value_head.*``, ``rlt_module.*``, …) are kept as-is.
    """
    if not any(key.startswith(_OLD_OPENPI_PREFIX) for key in state_dict):
        return state_dict

    from rlinf.utils.ckpt_convertor.openpi.openpi_pytorch_to_openpi import (
        old_to_new_state_dict,
    )

    converted = old_to_new_state_dict(state_dict)
    for key, tensor in state_dict.items():
        if key.startswith(_OLD_OPENPI_PREFIX) or key in converted:
            continue
        converted[key] = tensor
    logger.info(
        "openpi: converted OpenPI PyTorch checkpoint keys from %s in memory",
        source,
    )
    return converted


def load_full_weights(model, weights_path, *, expect_rlt: bool) -> None:
    """Load an RLinf ``full_weights.pt`` checkpoint into a Pi0 (or subclass)."""
    import torch

    from rlinf.utils.ckpt_convertor.openpi._core import as_state_dict

    loaded = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    state_dict = _convert_official_openpi_keys(
        _normalize_state_dict(as_state_dict(loaded)),
        weights_path,
    )
    if expect_rlt and not any(key.startswith("rlt_module.") for key in state_dict):
        raise ValueError(
            "openpi RLT checkpoint has no rlt_module.* weights. "
            "Stage2 must consume a Stage1 checkpoint trained with openpi.use_rlt=True."
        )

    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = list(incompatible.missing_keys)
    matched = len(state_dict) - len(unexpected)
    if matched <= 0:
        raise RuntimeError(
            f"No tensors from {weights_path} matched the openpi model. "
            "This usually means the checkpoint is still in the legacy official "
            "OpenPI PyTorch key layout."
        )
    missing_pi0 = _missing_pi0_keys(missing)
    if missing_pi0:
        raise RuntimeError(
            f"Pi0 tensors missing from {weights_path}: {missing_pi0[:8]}"
        )
    if expect_rlt and any(key.startswith("rlt_module.") for key in missing):
        raise RuntimeError(
            f"RLT checkpoint {weights_path} did not load all rlt_module weights; "
            f"missing={missing[:8]}"
        )

    if missing or unexpected:
        logger.warning(
            "openpi: loaded checkpoint %s with strict=False "
            "(matched=%d missing=%d unexpected=%d)",
            weights_path,
            matched,
            len(missing),
            len(unexpected),
        )
    else:
        logger.info("openpi: loaded full checkpoint from %s", weights_path)


def _load_safetensors_state_dict(safetensors_path: pathlib.Path) -> dict:
    """Load all indexed shards before any cross-tensor layout conversion."""
    import safetensors.torch

    if not safetensors_path.name.endswith(".safetensors.index.json"):
        return safetensors.torch.load_file(str(safetensors_path), device="cpu")

    index = json.loads(safetensors_path.read_text())
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if (
        not isinstance(weight_map, dict)
        or not weight_map
        or not all(
            isinstance(key, str)
            and isinstance(shard, str)
            and shard.endswith(".safetensors")
            for key, shard in weight_map.items()
        )
    ):
        raise ValueError(
            f"Expected a non-empty tensor-to-safetensors weight_map in "
            f"{safetensors_path}; found {weight_map!r}."
        )

    state_dict = {}
    for shard in sorted(set(weight_map.values())):
        shard_path = safetensors_path.parent / shard
        if not shard_path.is_file():
            raise FileNotFoundError(
                f"Safetensors shard {shard_path} referenced by "
                f"{safetensors_path} not found."
            )
        shard_state = safetensors.torch.load_file(str(shard_path), device="cpu")
        expected = {key for key, filename in weight_map.items() if filename == shard}
        missing = sorted(expected - shard_state.keys())
        unexpected = sorted(shard_state.keys() - expected)
        if missing or unexpected:
            raise ValueError(
                f"Safetensors shard {shard_path} does not match "
                f"{safetensors_path}: missing={missing[:8]}, "
                f"unexpected={unexpected[:8]}."
            )
        state_dict.update(shard_state)
    return state_dict


def load_base_safetensors(model, safetensors_path) -> None:
    """Load single-file or indexed shards in new or legacy OpenPI layouts."""
    safetensors_path = pathlib.Path(safetensors_path)
    state_dict = _convert_official_openpi_keys(
        _load_safetensors_state_dict(safetensors_path),
        safetensors_path,
    )
    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if not hasattr(model, "value_head"):
        unused_value_keys = [key for key in unexpected if key.startswith("value_head.")]
        if unused_value_keys:
            logger.info(
                "openpi: ignoring %d value_head tensors from %s; "
                "the target model has no value head",
                len(unused_value_keys),
                safetensors_path,
            )
            unexpected = [
                key for key in unexpected if not key.startswith("value_head.")
            ]
    if unexpected:
        raise RuntimeError(
            f"Unexpected keys loading {safetensors_path}: {unexpected[:8]}"
        )
    missing_pi0 = _missing_pi0_keys(incompatible.missing_keys)
    if missing_pi0:
        raise RuntimeError(
            f"Pi0 tensors missing from {safetensors_path}: {missing_pi0[:8]}"
        )
    if incompatible.missing_keys:
        logger.info(
            "openpi: loaded base safetensors %s; leaving %d extra module "
            "tensors randomly initialized (%s...)",
            safetensors_path,
            len(incompatible.missing_keys),
            incompatible.missing_keys[:4],
        )
