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

import logging
import sys
import types
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

# Monkey-patch: inject a stub for rlinf.envs.libero.asset_paths so that
# env workers can import it even when the file is absent from the container.
# This avoids modifying the official rlinf/envs/libero/libero_env.py.
_OLD_ASSET_PATHS = sys.modules.get("rlinf.envs.libero.asset_paths")
if _OLD_ASSET_PATHS is None:
    _stub = types.ModuleType("rlinf.envs.libero.asset_paths")

    def _noop(*args, **kwargs):
        pass

    _stub.apply_standard_libero_env_vars = _noop
    sys.modules["rlinf.envs.libero.asset_paths"] = _stub


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
    from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
    from transformers import AutoConfig, AutoModel

    AutoConfig.register("Gr00tN1d6", Gr00tN1d6Config)
    AutoModel.register(Gr00tN1d6Config, Gr00tN1d6)
    logging.info(
        "Successfully registered custom architecture Gr00tN1d6, authentication passed!"
    )
    import rlinf.hybrid_engines.fsdp.strategy.fsdp as fsdp_strategy

    if not hasattr(fsdp_strategy, "_is_gr00t_patched"):
        orig_policy = fsdp_strategy.get_fsdp_wrap_policy

        def custom_fsdp_wrap_policy(
            module, config=None, is_lora=False, model_type=None
        ):
            import functools

            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

            target_keywords = [
                "DecoderLayer",
                "EncoderLayer",
                "DiTBlock",
                "NoiseNet",
                "ValueHead",
                "ActionHead",
                "Timestep",
            ]
            found_classes = set()
            for name, mod in module.named_modules():
                cname = mod.__class__.__name__
                if any(key in cname for key in target_keywords):
                    found_classes.add(mod.__class__)

            if found_classes:
                logging.info(
                    "\n  FSDP Slicer: %s\n", [c.__name__ for c in found_classes]
                )
                return functools.partial(
                    transformer_auto_wrap_policy, transformer_layer_cls=found_classes
                )

            return orig_policy(module, config, is_lora, model_type)

        fsdp_strategy.get_fsdp_wrap_policy = custom_fsdp_wrap_policy
        fsdp_strategy._is_gr00t_patched = True
    from rlinf.utils.patcher import Patcher

    Patcher.clear()
    Patcher.add_patch(
        "gr00t.data.embodiment_tags.EmbodimentTag",
        "rlinf.models.embodiment.gr00t_n1d6.embodiment_tags.EmbodimentTag",
    )
    Patcher.add_patch(
        "gr00t.data.embodiment_tags.EMBODIMENT_TAG_MAPPING",
        "rlinf.models.embodiment.gr00t_n1d6.embodiment_tags.EMBODIMENT_TAG_MAPPING",
    )
    Patcher.apply()

    from gr00t.data.embodiment_tags import EmbodimentTag

    from rlinf.models.embodiment.gr00t_n1d6.gr00t_action_model import (
        GR00T_N1_6_ForRLActionPrediction,
    )
    from rlinf.models.embodiment.gr00t_n1d6.utils import replace_dropout_with_identity

    is_sft_model = cfg.get("model_type") == "gr00t_n1d6_sft"
    use_official_libero_panda = bool(
        OmegaConf.select(cfg, "use_official_libero_panda", default=is_sft_model)
    )

    if cfg.embodiment_tag == "libero_panda":
        emb_tag = (
            EmbodimentTag.LIBERO_PANDA
            if use_official_libero_panda
            else EmbodimentTag.ROBOCASA_PANDA_OMRON
        )
    elif cfg.embodiment_tag in [
        "libero_franka",
        "isaaclab_franka",
        "maniskill_widowx",
        "robocasa_panda_omron",
    ]:
        emb_tag = EmbodimentTag.ROBOCASA_PANDA_OMRON
    elif cfg.embodiment_tag == "gr1":
        emb_tag = EmbodimentTag.GR1
    elif cfg.embodiment_tag == "behavior_r1_pro":
        emb_tag = EmbodimentTag.BEHAVIOR_R1_PRO
    else:
        raise ValueError(
            f"Invalid or unsupported embodiment tag: {cfg.embodiment_tag}. "
            f"Supported tags are: ['behavior_r1_pro', 'gr1', 'robocasa_panda_omron', 'libero_panda']."
        )

    model_path = Path(cfg.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    if cfg.get("model_type") == "gr00t_n1d6_sft":
        from .gr00t_n1d6_sft_model import GR00T_N1D6_SFT_Model

        model_cls = GR00T_N1D6_SFT_Model
    else:
        model_cls = GR00T_N1_6_ForRLActionPrediction

    config = Gr00tN1d6Config.from_pretrained(str(model_path))
    _action_dim = cfg.get("action_dim")
    if _action_dim is not None:
        config.action_dim = _action_dim

    processor_path = OmegaConf.select(cfg, "processor_path", default=None)

    model = model_cls.from_pretrained(
        config=config,
        local_model_path=str(model_path),
        pretrained_model_name_or_path=str(model_path),
        torch_dtype=torch_dtype,
        embodiment_tag=emb_tag,
        denoising_steps=cfg.denoising_steps,
        output_action_chunks=cfg.num_action_chunks,
        obs_converter_type=cfg.obs_converter_type,
        rl_head_config=cfg.rl_head_config,
        processor_path=processor_path,
    )

    model.to(torch_dtype)
    if cfg.rl_head_config.add_value_head and hasattr(model.action_head, "value_head"):
        # reinitialize the value head after model loading
        model.action_head.value_head._init_weights()

    if cfg.rl_head_config.disable_dropout:
        replace_dropout_with_identity(model)

    return model


def patch_fsdp_rollout_state_dict():
    """Patch EmbodiedFSDPActor.get_rollout_state_dict to use full_state_dict=True."""
    try:
        from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

        def _patched_get_rollout_state_dict(self) -> dict:
            return self.get_model_state_dict(cpu_offload=False, full_state_dict=True)

        EmbodiedFSDPActor.get_rollout_state_dict = _patched_get_rollout_state_dict
        logging.info(
            "[GR00T patch] EmbodiedFSDPActor.get_rollout_state_dict patched: "
            "full_state_dict=True for multi-GPU FSDP weight sync safety"
        )
    except Exception as e:
        logging.warning("[GR00T patch] Failed to patch get_rollout_state_dict: %s", e)
