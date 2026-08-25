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
from rlinf.models.embodiment.openpi_rlinf.utils.rlt_utils import (
    FULL_WEIGHTS_CANDIDATES,
    load_base_safetensors,
    load_full_wrapper_weights,
    resolve_full_weights,
    resolve_model_safetensors,
)
from rlinf.utils.logging import get_logger

logger = get_logger()


def get_model(cfg: Any, torch_dtype: Any = None) -> Any:
    """Build an OpenPI PyTorch Pi0/Pi0.5 model from ``actor.model`` config.

    ``cfg.model_path`` may point at either a new-format base checkpoint
    containing ``model.safetensors`` or an RLinf FSDP SFT checkpoint containing
    ``full_weights.pt``. Model shape comes from YAML; no checkpoint
    ``config.json`` is read. ``cfg.openpi.task`` selects the SFT, eval, or RL
    wrapper around the shared Pi0 core.
    """
    import pathlib

    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi_rlinf.pi0_model import gemma as pi0_gemma
    from rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0_config import Pi0Config
    from rlinf.models.embodiment.openpi_rlinf.utils.model_builders import (
        _build_eval_model,
        _build_rl_model,
        _build_sft_model,
    )

    model_cfg = cfg.openpi
    # Existing Pi0.5 templates predate the explicit switch, so preserve their
    # behavior by default. Pi0 templates set this field to False explicitly.
    pi05 = bool(OmegaConf.select(cfg, "pi05", default=True))
    target_dtype = (
        torch_dtype
        if torch_dtype is not None
        else torch_dtype_from_precision(cfg.precision)
    )

    model_path = pathlib.Path(cfg.model_path).expanduser()
    safetensors_path = resolve_model_safetensors(model_path)
    full_weights_path = resolve_full_weights(model_path)
    if safetensors_path is None and full_weights_path is None:
        raise FileNotFoundError(
            "openpi_rlinf checkpoint not found. Expected either "
            f"{model_path}/model.safetensors or one of "
            f"{[str(model_path / rel) for rel in FULL_WEIGHTS_CANDIDATES]}."
        )

    pi0_kwargs = {
        "pi05": pi05,
        "action_horizon": int(cfg.num_action_chunks),
        "action_dim": int(model_cfg.model_action_dim),
        "paligemma_variant": str(model_cfg.paligemma_variant),
        "action_expert_variant": str(model_cfg.action_expert_variant),
        "dtype": cfg.get("precision", "bfloat16"),
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
        load_base_safetensors(model, safetensors_path)
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
        load_full_wrapper_weights(
            wrapper,
            full_weights_path,
            expect_rlt=bool(OmegaConf.select(model_cfg, "use_rlt", default=False)),
        )

    source = full_weights_path if full_weights_path is not None else safetensors_path
    logger.info(
        "openpi_rlinf[%s]: loaded %s (%.2fB params) from %s precision=%s num_steps=%s",
        task,
        pi0_config,
        n_params / 1e9,
        source,
        cfg.precision,
        num_steps,
    )
    return wrapper
