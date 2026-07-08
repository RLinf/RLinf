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

_SUPPORTED_EMBODIMENT_TAGS = (
    "behavior_r1_pro",
    "gr1",
    "robocasa_panda_omron",
    "libero_panda",
    "libero_franka",
    "isaaclab_franka",
    "maniskill_widowx",
    "new_embodiment",
    "so101",
    "so100",
)


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
    from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
    from transformers import AutoConfig, AutoModel

    AutoConfig.register("Gr00tN1d6", Gr00tN1d6Config)
    AutoModel.register(Gr00tN1d6Config, Gr00tN1d6)

    from rlinf.utils.patcher import Patcher

    Patcher.clear()
    Patcher.add_patch(
        "gr00t.data.embodiment_tags.EmbodimentTag",
        "rlinf.models.embodiment.gr00t.embodiment_tags.EmbodimentTag",
    )
    Patcher.add_patch(
        "gr00t.data.embodiment_tags.EMBODIMENT_TAG_MAPPING",
        "rlinf.models.embodiment.gr00t.embodiment_tags.EMBODIMENT_TAG_MAPPING",
    )
    Patcher.apply()

    from gr00t.data.embodiment_tags import EmbodimentTag

    from rlinf.models.embodiment.gr00t.gr00t_n1d6.gr00t_action_model import (
        GR00T_N1_6_ForRLActionPrediction,
    )
    from rlinf.models.embodiment.gr00t.utils import replace_dropout_with_identity

    embodiment_tag_by_cfg = {
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
            f"Supported tags are: {list(_SUPPORTED_EMBODIMENT_TAGS)}."
        )

    model_path = Path(cfg.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    config = Gr00tN1d6Config.from_pretrained(str(model_path))
    _action_dim = cfg.get("action_dim")
    if _action_dim is not None:
        config.action_dim = _action_dim

    processor_path = OmegaConf.select(cfg, "processor_path", default=None)

    model = GR00T_N1_6_ForRLActionPrediction.from_pretrained(
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
        model.action_head.value_head._init_weights()

    if cfg.rl_head_config.disable_dropout:
        replace_dropout_with_identity(model)

    return model
