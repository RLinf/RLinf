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
from rlinf.models.embodiment.openpi_rlinf.checkpoint import (
    FULL_WEIGHTS_CANDIDATES,
    load_base_safetensors,
    load_full_weights,
    resolve_full_weights,
    resolve_model_safetensors,
)
from rlinf.models.embodiment.openpi_rlinf.rlt_config import build_rlt_config
from rlinf.utils.logging import get_logger

logger = get_logger()


def get_model(cfg: Any, torch_dtype: Any = None) -> Any:
    """Build an OpenPI PyTorch Pi0/Pi0.5 model from ``actor.model`` config.

    ``cfg.model_path`` may point at either a new-format base checkpoint
    containing ``model.safetensors`` or an RLinf FSDP SFT checkpoint containing
    ``full_weights.pt``. Model shape comes from YAML; no checkpoint
    ``config.json`` is read. ``cfg.openpi.task`` selects ``Pi0`` (SFT) or a
    task subclass; omitted ``task`` defaults to SFT.
    """
    import pathlib

    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi_rlinf.pi0 import Pi0
    from rlinf.models.embodiment.openpi_rlinf.pi0_config import Pi0Config

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
    if full_weights_path is not None and safetensors_path is not None:
        logger.warning(
            "openpi_rlinf: both %s and %s exist; loading full_weights (RLinf "
            "resume) and ignoring model.safetensors. Remove the leftover file "
            "if that is not what you intended.",
            full_weights_path,
            safetensors_path,
        )

    action_horizon, action_chunk = _resolve_action_horizon_and_chunk(cfg, model_cfg)
    action_env_dim = int(
        OmegaConf.select(model_cfg, "action_env_dim", default=cfg.action_dim)
    )
    num_steps = OmegaConf.select(model_cfg, "num_steps", default=None)
    num_steps = int(num_steps) if num_steps is not None else int(cfg.num_steps)

    pi0_kwargs = {
        "pi05": pi05,
        "action_horizon": action_horizon,
        "action_dim": int(model_cfg.model_action_dim),
        "paligemma_variant": str(model_cfg.paligemma_variant),
        "action_expert_variant": str(model_cfg.action_expert_variant),
        "dtype": _pi0_dtype_name(target_dtype),
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
    rlt_cfg = build_rlt_config(model_cfg)
    runtime = {
        "num_steps": num_steps,
        "action_env_dim": action_env_dim,
        "action_chunk": action_chunk,
    }

    task = OmegaConf.select(model_cfg, "task", default="sft")
    task = str(task).lower() if task is not None else "sft"

    if task == "sft":
        model = Pi0(
            pi0_config,
            num_steps=num_steps,
            action_env_dim=action_env_dim,
            action_chunk=action_chunk,
            rlt_cfg=rlt_cfg,
        )
    elif task == "eval":
        from rlinf.models.embodiment.openpi_rlinf.tasks.eval import Pi0Eval

        config_name = _require_config_name(model_cfg, task)
        model = Pi0Eval(
            pi0_config,
            **runtime,
            config_name=config_name,
            state_indices=OmegaConf.select(model_cfg, "state_indices", default=None),
            rlt_cfg=rlt_cfg,
            rtc_enabled=bool(OmegaConf.select(model_cfg, "rtc_enabled", default=False)),
            rtc_guidance_mode=str(
                OmegaConf.select(model_cfg, "rtc_guidance_mode", default="approx")
            ),
            rtc_guidance_clip=float(
                OmegaConf.select(model_cfg, "rtc_guidance_clip", default=5.0)
            ),
        )
        _install_transforms(model, cfg, config_name)
    elif task == "rl":
        from rlinf.models.embodiment.openpi_rlinf.tasks.rl import Pi0RL, Pi0RLConfig

        config_name = _require_config_name(model_cfg, task)
        noise_logvar = OmegaConf.select(
            model_cfg, "noise_logvar_range", default=[0.08, 0.16]
        )
        rl_cfg = Pi0RLConfig(
            add_value_head=bool(OmegaConf.select(cfg, "add_value_head", default=False)),
            noise_method=str(
                OmegaConf.select(model_cfg, "noise_method", default="flow_ode")
            ),
            noise_level=float(OmegaConf.select(model_cfg, "noise_level", default=0.0)),
            noise_logvar_range=tuple(float(x) for x in noise_logvar),
            joint_logprob=bool(
                OmegaConf.select(model_cfg, "joint_logprob", default=False)
            ),
            ignore_last=bool(OmegaConf.select(model_cfg, "ignore_last", default=False)),
            value_after_vlm=bool(
                OmegaConf.select(model_cfg, "value_after_vlm", default=False)
            ),
            value_vlm_mode=str(
                OmegaConf.select(model_cfg, "value_vlm_mode", default="mean_token")
            ),
            detach_critic_input=bool(
                OmegaConf.select(model_cfg, "detach_critic_input", default=False)
            ),
            chunk_critic_input=bool(
                OmegaConf.select(model_cfg, "chunk_critic_input", default=False)
            ),
            train_expert_only=bool(
                OmegaConf.select(model_cfg, "train_expert_only", default=False)
            ),
            is_nft=bool(OmegaConf.select(model_cfg, "is_nft", default=False)),
            config_name=config_name,
        )
        model = Pi0RL(
            pi0_config,
            **runtime,
            rl_cfg=rl_cfg,
            config_name=config_name,
            state_indices=OmegaConf.select(model_cfg, "state_indices", default=None),
        )
        _install_transforms(model, cfg, config_name)
    elif task == "dagger":
        from rlinf.models.embodiment.openpi_rlinf.tasks.dagger import Pi0DAgger

        config_name = _require_config_name(model_cfg, task)
        model = Pi0DAgger(
            pi0_config,
            **runtime,
            config_name=config_name,
            state_indices=OmegaConf.select(model_cfg, "state_indices", default=None),
        )
        _install_transforms(model, cfg, config_name)
    elif task == "dsrl":
        from rlinf.models.embodiment.openpi_rlinf.tasks.dsrl import (
            Pi0DSRL,
            Pi0DSRLConfig,
        )

        config_name = _require_config_name(model_cfg, task)
        train_expert_only = OmegaConf.select(
            model_cfg, "train_expert_only", default=None
        )
        if train_expert_only is not None and not bool(train_expert_only):
            logger.warning(
                "openpi.train_expert_only=False is ignored for task=dsrl; "
                "the Pi0 decoder is always frozen."
            )
        hidden = OmegaConf.select(
            model_cfg, "dsrl_hidden_dims", default=[128, 128, 128]
        )
        dsrl_cfg = Pi0DSRLConfig(
            state_dim=int(OmegaConf.select(model_cfg, "dsrl_state_dim", default=8)),
            action_noise_dim=int(
                OmegaConf.select(model_cfg, "dsrl_action_noise_dim", default=32)
            ),
            num_q_heads=int(
                OmegaConf.select(model_cfg, "dsrl_num_q_heads", default=10)
            ),
            image_latent_dim=int(
                OmegaConf.select(model_cfg, "dsrl_image_latent_dim", default=64)
            ),
            state_latent_dim=int(
                OmegaConf.select(model_cfg, "dsrl_state_latent_dim", default=64)
            ),
            hidden_dims=tuple(int(x) for x in hidden),
        )
        model = Pi0DSRL(
            pi0_config,
            **runtime,
            config_name=config_name,
            state_indices=OmegaConf.select(model_cfg, "state_indices", default=None),
            dsrl_cfg=dsrl_cfg,
        )
        _install_transforms(model, cfg, config_name)
    else:
        raise ValueError(
            f"actor.model.openpi.task={task!r} is not supported; "
            "use 'eval', 'sft', 'rl', 'dagger', or 'dsrl'."
        )

    if full_weights_path is not None:
        load_full_weights(
            model,
            full_weights_path,
            expect_rlt=task in ("sft", "eval")
            and bool(OmegaConf.select(model_cfg, "use_rlt", default=False)),
        )
    elif safetensors_path is not None:
        load_base_safetensors(model, safetensors_path)

    _freeze_after_load(model, task)
    _apply_openpi_param_dtypes(model, target_dtype)

    n_params = sum(param.numel() for param in model.parameters())
    source = full_weights_path if full_weights_path is not None else safetensors_path
    logger.info(
        "openpi_rlinf[%s]: loaded %s (%.2fB params) from %s precision=%s "
        "num_steps=%s action_horizon=%s action_chunk=%s",
        task,
        pi0_config,
        n_params / 1e9,
        source,
        cfg.precision,
        num_steps,
        action_horizon,
        action_chunk,
    )
    return model


def _freeze_after_load(model, task: str) -> None:
    """Freeze after weights load so ``requires_grad`` is the post-load truth."""
    if task == "rl" and getattr(model, "rl_cfg", None) is not None:
        if model.rl_cfg.train_expert_only:
            frozen = model.freeze_vlm()
            logger.info(
                "openpi_rlinf[rl]: train_expert_only=True; froze %d parameter "
                "tensors (SigLIP + gemma expert-0) after weight load",
                frozen,
            )
        return
    if task == "dsrl":
        frozen = model.freeze_vlm(freeze_action_expert=True)
        logger.info(
            "openpi_rlinf[dsrl]: froze %d Pi0 parameter tensors after weight "
            "load; SAC heads remain trainable",
            frozen,
        )


def _apply_openpi_param_dtypes(model, target_dtype) -> None:
    """Apply OpenPI's selective mixed precision after checkpoint load.

    YAML ``precision: null`` (OpenPI default) and ``bf16`` both mean: Gemma /
    SigLIP weights in bf16, RMSNorm + vision stem + action/value heads in fp32.
    A global ``model.to(bf16)`` would also cast those fp32 islands and is the
    source of higher PPO KL versus OpenPI. ``fp32`` keeps the whole net in
    fp32 (SFT).
    """
    import torch

    if target_dtype is None or target_dtype == torch.bfloat16:
        model.to_bfloat16_for_selected_params("bfloat16")
    elif target_dtype == torch.float32:
        model.to_bfloat16_for_selected_params("float32")
    else:
        model.to(target_dtype)

    sample_keys = (
        "img.stem.weight",
        "img.pos_embedding",
        "llm.layers.0.pre_attention_norms.0.scale",
        "llm.layers.0.attn.q_proj.0.weight",
        "action_out_proj.weight",
        "state_proj.weight",
        "value_head.mlp.0.weight",
    )
    named = dict(model.named_parameters())
    parts = [f"{key}={named[key].dtype}" for key in sample_keys if key in named]
    if parts:
        logger.info("openpi_rlinf param dtypes: %s", ", ".join(parts))


def _pi0_dtype_name(target_dtype) -> str:
    """Map the factory compute dtype onto ``Pi0Config.dtype`` (embed dtype)."""
    import torch

    if target_dtype is None or target_dtype == torch.bfloat16:
        return "bfloat16"
    if target_dtype == torch.float32:
        return "float32"
    if target_dtype == torch.float16:
        return "float16"
    raise ValueError(
        f"Unsupported openpi_rlinf dtype {target_dtype!r}; "
        "expected float32, float16, or bfloat16."
    )


def _resolve_action_horizon_and_chunk(cfg, model_cfg) -> tuple[int, int]:
    """Split model horizon from the env-executed chunk.

    ``num_action_chunks`` / ``openpi.action_chunk`` is the env interface.
    The network horizon is ``openpi.action_horizon`` when set, otherwise the
    official OpenPI ``TrainConfig`` for ``config_name``, otherwise
    ``num_action_chunks``.
    """
    from omegaconf import OmegaConf

    chunk = OmegaConf.select(model_cfg, "action_chunk", default=None)
    action_chunk = int(chunk) if chunk is not None else int(cfg.num_action_chunks)

    yaml_horizon = OmegaConf.select(model_cfg, "action_horizon", default=None)
    if yaml_horizon is not None:
        return int(yaml_horizon), action_chunk

    config_name = str(OmegaConf.select(model_cfg, "config_name", default="") or "")
    if config_name:
        from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

        official_horizon = int(
            get_openpi_config(
                config_name,
                model_path=cfg.model_path,
                data_kwargs=_resolve_data_kwargs(cfg),
            ).model.action_horizon
        )
        return official_horizon, action_chunk

    return int(cfg.num_action_chunks), action_chunk


def _require_config_name(model_cfg, task: str) -> str:
    from omegaconf import OmegaConf

    config_name = str(OmegaConf.select(model_cfg, "config_name", default=""))
    if not config_name:
        raise ValueError(
            f"actor.model.openpi.config_name is required for task={task!r} "
            "(it selects the upstream openpi TrainConfig)."
        )
    return config_name


def _resolve_data_kwargs(cfg):
    from omegaconf import OmegaConf

    data_kwargs = OmegaConf.select(cfg, "openpi_data", default=None)
    if data_kwargs is not None:
        data_kwargs = OmegaConf.to_container(data_kwargs, resolve=True)
    return data_kwargs


def _install_transforms(model, cfg, config_name: str):
    from rlinf.models.embodiment.openpi_rlinf.transforms.pipeline import (
        build_openpi_transforms,
    )

    input_transforms, output_transforms = build_openpi_transforms(
        cfg.model_path, config_name, data_kwargs=_resolve_data_kwargs(cfg)
    )
    model.setup_transforms(input_transforms, output_transforms)
    return model
