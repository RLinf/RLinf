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

from __future__ import annotations

from typing import Any

from rlinf.config import torch_dtype_from_precision
from rlinf.utils.logging import get_logger

logger = get_logger()

_FULL_WEIGHTS_CANDIDATES = (
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


def _resolve_model_safetensors(model_path: Any):
    import pathlib

    path = pathlib.Path(model_path).expanduser()
    if path.is_file() and path.name.endswith(".safetensors"):
        return path
    weights_path = path / "model.safetensors"
    return weights_path if weights_path.exists() else None


def _resolve_full_weights(model_path: Any):
    import pathlib

    path = pathlib.Path(model_path).expanduser()
    if path.is_file() and path.name.endswith(".pt"):
        return path
    for rel_path in _FULL_WEIGHTS_CANDIDATES:
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
        key.startswith("model.") or key.startswith("rlt_module.")
        for key in normalized
    )
    if has_wrapper_key:
        return normalized

    if any(key.startswith(_BARE_PI0_PREFIXES) for key in normalized):
        return {f"model.{key}": tensor for key, tensor in normalized.items()}
    return normalized


def _load_full_wrapper_weights(wrapper, weights_path, *, expect_rlt: bool) -> None:
    import torch

    from rlinf.utils.ckpt_convertor.openpi._core import as_state_dict

    loaded = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    state_dict = _normalize_wrapper_state_dict(as_state_dict(loaded))
    if expect_rlt and not any(key.startswith("rlt_module.") for key in state_dict):
        raise ValueError(
            "openpi_pytorch RLT checkpoint has no rlt_module.* weights. "
            "Stage2 must consume a Stage1 checkpoint trained with openpi.use_rlt=True."
        )

    incompatible = wrapper.load_state_dict(state_dict, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = list(incompatible.missing_keys)
    matched = len(state_dict) - len(unexpected)
    if matched <= 0:
        raise RuntimeError(
            f"No tensors from {weights_path} matched the openpi_pytorch wrapper. "
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
            "openpi_pytorch: loaded wrapper checkpoint %s with strict=False "
            "(matched=%d missing=%d unexpected=%d)",
            weights_path,
            matched,
            len(missing),
            len(unexpected),
        )
    else:
        logger.info(
            "openpi_pytorch: loaded full wrapper checkpoint from %s", weights_path
        )


def get_model(cfg: Any, torch_dtype: Any = None) -> Any:
    """Build an OpenPI PyTorch Pi0.5 model from ``actor.model`` config.

    ``cfg.model_path`` may point at either a new-format base checkpoint
    containing ``model.safetensors`` or an RLinf FSDP SFT checkpoint containing
    ``full_weights.pt``. Model shape comes from YAML; no checkpoint
    ``config.json`` is read. ``cfg.openpi.task`` selects the SFT, eval, or RL
    wrapper around the shared Pi0 core.
    """
    import pathlib

    import safetensors.torch
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi_pytorch.pi0_model import gemma as pi0_gemma
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0_config import Pi0Config
    from rlinf.models.embodiment.openpi_pytorch.utils.model_builders import (
        _build_eval_model,
        _build_rl_model,
        _build_sft_model,
    )

    model_cfg = cfg.openpi
    target_dtype = (
        torch_dtype
        if torch_dtype is not None
        else torch_dtype_from_precision(cfg.precision)
    )

    model_path = pathlib.Path(cfg.model_path).expanduser()
    safetensors_path = _resolve_model_safetensors(model_path)
    full_weights_path = _resolve_full_weights(model_path)
    if safetensors_path is None and full_weights_path is None:
        raise FileNotFoundError(
            "openpi_pytorch checkpoint not found. Expected either "
            f"{model_path}/model.safetensors or one of "
            f"{[str(model_path / rel) for rel in _FULL_WEIGHTS_CANDIDATES]}."
        )

    pi0_kwargs = {
        "pi05": True,
        "action_horizon": int(cfg.num_action_chunks),
        "action_dim": int(model_cfg.model_action_dim),
        "paligemma_variant": str(model_cfg.paligemma_variant),
        "action_expert_variant": str(model_cfg.action_expert_variant),
        "dtype": "bfloat16",
        "pcd": False,
    }
    discrete_state_input = OmegaConf.select(
        model_cfg, "discrete_state_input", default=None
    )
    if discrete_state_input is not None:
        pi0_kwargs["discrete_state_input"] = bool(discrete_state_input)
    max_token_len = OmegaConf.select(model_cfg, "max_token_len", default=None)
    if max_token_len is not None:
        pi0_kwargs["max_token_len"] = int(max_token_len)

    pi0_config = Pi0Config(**pi0_kwargs)
    model = pi0_config.create()
    if safetensors_path is not None and full_weights_path is None:
        state_dict = safetensors.torch.load_file(str(safetensors_path), device="cpu")
        model.load_state_dict(state_dict, strict=True)
    n_params = sum(param.numel() for param in model.parameters())
    if target_dtype is not None:
        model = model.to(target_dtype)

    num_steps = int(cfg.num_steps)
    action_chunk = int(cfg.num_action_chunks)
    action_env_dim = int(cfg.action_dim)

    task = OmegaConf.select(model_cfg, "task", default=None)
    if task is None:
        raise ValueError(
            "actor.model.openpi.task is required: set it to 'sft', 'rl', or "
            "'eval' to pick the concrete OpenPI PyTorch model variant."
        )
    task = str(task).lower()

    if task == "eval":
        wrapper = _build_eval_model(
            cfg,
            model_cfg,
            model,
            num_steps=num_steps,
            action_chunk=action_chunk,
            action_env_dim=action_env_dim,
        )
    elif task == "sft":
        wrapper = _build_sft_model(
            model_cfg,
            model,
            num_steps=num_steps,
            action_env_dim=action_env_dim,
        )
    elif task == "rl":
        paligemma_width = pi0_gemma.get_config(pi0_config.paligemma_variant).width
        wrapper = _build_rl_model(
            cfg,
            model_cfg,
            model,
            num_steps=num_steps,
            action_chunk=action_chunk,
            action_env_dim=action_env_dim,
            paligemma_width=paligemma_width,
        )
    else:
        raise ValueError(
            f"actor.model.openpi.task={task!r} is not supported; "
            "use 'eval', 'sft', or 'rl'."
        )

    if full_weights_path is not None:
        _load_full_wrapper_weights(
            wrapper,
            full_weights_path,
            expect_rlt=bool(OmegaConf.select(model_cfg, "use_rlt", default=False)),
        )

    source = full_weights_path if full_weights_path is not None else safetensors_path
    logger.info(
        "openpi_pytorch[%s]: loaded %s (%.2fB params) from %s "
        "precision=%s num_steps=%s",
        task,
        pi0_config,
        n_params / 1e9,
        source,
        cfg.precision,
        num_steps,
    )
    return wrapper
