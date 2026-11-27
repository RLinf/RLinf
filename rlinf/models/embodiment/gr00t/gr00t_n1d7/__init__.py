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

from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.utils.logging import get_logger

from .npu_patches import apply_npu_patches, restore_npu_patches


def _limit_sft_logits(_module, args, kwargs):
    # The action head consumes hidden states, not the language-model logits.
    return args, {**kwargs, "logits_to_keep": 1}


def _resolve_sft_checkpoint(model_path: Path) -> tuple[Path, Path] | None:
    """Return ``(checkpoint_dir, full_weights_path)`` for an FSDP SFT export."""
    if (
        (model_path / "model.safetensors").is_file()
        or (model_path / "model.safetensors.index.json").is_file()
        or any(model_path.glob("model-*.safetensors"))
    ):
        return None
    for checkpoint_dir in (model_path / "actor", model_path):
        full_weights_path = checkpoint_dir / "model_state_dict" / "full_weights.pt"
        if full_weights_path.is_file():
            return checkpoint_dir, full_weights_path
    return None


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
    from transformers import AutoConfig, AutoModel

    AutoConfig.register("Gr00tN1d7", Gr00tN1d7Config)
    AutoModel.register(Gr00tN1d7Config, Gr00tN1d7)

    # PPO rollout and actor use different batch sizes. BF16 reduced-precision
    # GEMM reductions can change frozen backbone features and policy densities.
    # This process-wide setting persists for all CUDA BF16 GEMMs in the worker.
    if torch_dtype == torch.bfloat16:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    logger = get_logger()
    logger.info(
        "Successfully registered custom architecture Gr00tN1d7, authentication passed!"
    )

    from rlinf.utils.patcher import Patcher

    Patcher.clear()
    Patcher.add_patch(
        "gr00t.data.embodiment_tags.EmbodimentTag",
        "rlinf.models.embodiment.gr00t.embodiment_tags.EmbodimentTag",
    )
    # Register the accelerator-specific patches (a no-op off Ascend) before
    # applying, and restore the process-global ones once construction finishes.
    npu_patch_state = apply_npu_patches(Patcher)
    try:
        Patcher.apply()
        from gr00t.data.embodiment_tags import EmbodimentTag

        from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (
            GR00T_N1_7_ForRLActionPrediction,
        )
        from rlinf.models.embodiment.gr00t.utils import replace_dropout_with_identity

        embodiment_tag_by_cfg = {
            "libero_sim": EmbodimentTag.LIBERO_SIM,  # N1.7 specific change
            "libero_panda": EmbodimentTag.LIBERO_PANDA,
            "libero_franka": EmbodimentTag.LIBERO_FRANKA,
            "isaaclab_franka": EmbodimentTag.ISAACLAB_FRANKA,
            "maniskill_widowx": EmbodimentTag.MANISKILL_WIDOWX,
            "robocasa_panda_omron": EmbodimentTag.ROBOCASA_PANDA_OMRON,
            "gr1": EmbodimentTag.GR1,
            "behavior_r1_pro": EmbodimentTag.BEHAVIOR_R1_PRO,
            "new_embodiment": EmbodimentTag.NEW_EMBODIMENT,
            "so101": EmbodimentTag.NEW_EMBODIMENT,
            "so100": EmbodimentTag.NEW_EMBODIMENT,
        }
        emb_tag = embodiment_tag_by_cfg.get(cfg.embodiment_tag)
        if emb_tag is None:
            raise ValueError(
                f"Invalid or unsupported embodiment tag: {cfg.embodiment_tag}. "
                f"Supported tags are: {list(embodiment_tag_by_cfg.keys())}."
            )

        model_path = Path(cfg.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        sft_checkpoint = _resolve_sft_checkpoint(model_path)
        config_dir = sft_checkpoint[0] if sft_checkpoint is not None else model_path
        config = Gr00tN1d7Config.from_pretrained(str(config_dir))
        training_mode = cfg.get("training_mode", "rl")
        if training_mode not in {"rl", "sft"}:
            raise ValueError(
                f"Expected GR00T training_mode 'rl' or 'sft', got {training_mode!r}"
            )
        init_backbone_from_config = cfg.get("init_backbone_from_config", False)
        if init_backbone_from_config and training_mode != "sft":
            raise ValueError("Config-only GR00T backbone initialization is SFT-only")
        if training_mode == "sft":
            if cfg.rl_head_config.add_value_head:
                raise ValueError(
                    "GR00T SFT requires rl_head_config.add_value_head=False"
                )
            for name in (
                "tune_llm",
                "tune_visual",
                "tune_projector",
                "tune_diffusion_model",
                "tune_vlln",
            ):
                setattr(config, name, bool(cfg.sft[name]))

        _action_dim = cfg.get("action_dim")
        if _action_dim is not None:
            config.action_dim = _action_dim

        backbone_model_path = OmegaConf.select(cfg, "backbone_model_path", default=None)

        if sft_checkpoint is not None:
            checkpoint_dir, full_weights_path = sft_checkpoint
            model = GR00T_N1_7_ForRLActionPrediction(
                config=config,
                local_model_path=str(checkpoint_dir),
                backbone_model_path=backbone_model_path,
                compute_dtype=torch_dtype,
                embodiment_tag=emb_tag,
                denoising_steps=cfg.denoising_steps,
                output_action_chunks=cfg.num_action_chunks,
                obs_converter_type=cfg.obs_converter_type,
                rl_head_config=cfg.rl_head_config,
                init_backbone_from_config=(backbone_model_path is not None),
            )
            model.load_state_dict(torch.load(full_weights_path, map_location="cpu"))
            model.eval()
        else:
            loaded = GR00T_N1_7_ForRLActionPrediction.from_pretrained(
                config=config,
                local_model_path=str(model_path),
                pretrained_model_name_or_path=str(model_path),
                backbone_model_path=backbone_model_path,
                torch_dtype=torch_dtype,
                embodiment_tag=emb_tag,
                denoising_steps=cfg.denoising_steps,
                output_action_chunks=cfg.num_action_chunks,
                obs_converter_type=cfg.obs_converter_type,
                rl_head_config=cfg.rl_head_config,
                init_backbone_from_config=init_backbone_from_config,
                output_loading_info=init_backbone_from_config,
            )
            if init_backbone_from_config:
                model, loading_info = loaded
                if (
                    loading_info["missing_keys"]
                    or loading_info["mismatched_keys"]
                    or loading_info["error_msgs"]
                ):
                    raise ValueError(
                        f"Incomplete GR00T SFT checkpoint restore: {loading_info}"
                    )
                model.checkpoint_loading_info = loading_info
                logger.info("GR00T SFT whole-checkpoint loading info: %s", loading_info)
            else:
                model = loaded
    finally:
        restore_npu_patches(Patcher, npu_patch_state)

    model.to(torch_dtype)
    if training_mode == "sft":
        model.backbone.model.register_forward_pre_hook(
            _limit_sft_logits, with_kwargs=True
        )
    if cfg.rl_head_config.add_value_head and hasattr(model.action_head, "value_head"):
        model.action_head.value_head._init_weights()

    if cfg.rl_head_config.disable_dropout:
        replace_dropout_with_identity(model)

    return model
