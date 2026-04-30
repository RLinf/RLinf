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

import torch
from omegaconf import DictConfig
from pathlib import Path


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    from transformers import AutoConfig, AutoModel
    from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
    from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
    AutoConfig.register("Gr00tN1d6", Gr00tN1d6Config)
    AutoModel.register(Gr00tN1d6Config, Gr00tN1d6)
    print("Successfully registered custom architecture Gr00tN1d6, authentication passed!")
    import rlinf.hybrid_engines.fsdp.strategy.fsdp as fsdp_strategy
    
    if not hasattr(fsdp_strategy, "_is_gr00t_patched"):
        orig_policy = fsdp_strategy.get_fsdp_wrap_policy
        
        def custom_fsdp_wrap_policy(module, config=None, is_lora=False, model_type=None):
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
            import functools
            
            target_keywords = ["DecoderLayer", "EncoderLayer", "DiTBlock", "NoiseNet", "ValueHead", "ActionHead", "Timestep"]
            found_classes = set()
            for name, mod in module.named_modules():
                cname = mod.__class__.__name__
                if any(key in cname for key in target_keywords):
                    found_classes.add(mod.__class__)
            
            if found_classes:
                print(f"\n  FSDP Slicer: {[c.__name__ for c in found_classes]}\n")
                return functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=found_classes)
            
            return orig_policy(module, config, is_lora, model_type)

        fsdp_strategy.get_fsdp_wrap_policy = custom_fsdp_wrap_policy
        fsdp_strategy._is_gr00t_patched = True
    from rlinf.utils.patcher import Patcher

    Patcher.clear()
    Patcher.add_patch(
        "gr00t.data.embodiment_tags.EmbodimentTag",
        "rlinf.models.embodiment.gr00t_1_6.embodiment_tags.EmbodimentTag",
    )
    Patcher.add_patch(
        "gr00t.data.embodiment_tags.EMBODIMENT_TAG_MAPPING",
        "rlinf.models.embodiment.gr00t_1_6.embodiment_tags.EMBODIMENT_TAG_MAPPING",
    )
    Patcher.apply()

    from gr00t.data.embodiment_tags import EmbodimentTag
    from rlinf.models.embodiment.gr00t_1_6.gr00t_action_model import (
        GR00T_N1_6_ForRLActionPrediction,
    )
    from rlinf.models.embodiment.gr00t_1_6.utils import replace_dropout_with_identity

    if cfg.embodiment_tag in ["libero_franka", "libero_panda"]:
        emb_tag = EmbodimentTag.LIBERO_PANDA
    elif cfg.embodiment_tag in ["isaaclab_franka", "maniskill_widowx", "robocasa_panda_omron"]:
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

    if cfg.get("model_type") == "gr00t_1_6_sft":
        from .gr00t_16_sft_model import GR00T_1_6_SFT_Model
        model_cls = GR00T_1_6_SFT_Model
    else:
        model_cls = GR00T_N1_6_ForRLActionPrediction

    model = model_cls.from_pretrained(
        local_model_path=str(model_path),
        pretrained_model_name_or_path=str(model_path),
        # model_path,
        torch_dtype=torch_dtype,
        embodiment_tag=emb_tag,
        processor_path=cfg.get("processor_path", None),
        denoising_steps=cfg.denoising_steps,
        output_action_chunks=cfg.num_action_chunks,
        obs_converter_type=cfg.obs_converter_type,
        tune_visual=False,
        tune_llm=False,
        rl_head_config=cfg.rl_head_config,
        # weight_syncer_cfg=cfg.get("weight_syncer_cfg", None),
    )

    if cfg.rl_head_config.get("add_value_head", False) and hasattr(
        model.action_head, "value_head"
    ):
        # The value head is absent from SFT checkpoints. Reinitialize it explicitly
        # after from_pretrained() so missing-key initialization cannot leave NaNs.
        value_head = model.action_head.value_head.float()
        value_head._init_weights("relu")
        final_layer = value_head.mlp[-1]
        torch.nn.init.zeros_(final_layer.weight)
        if final_layer.bias is not None:
            torch.nn.init.zeros_(final_layer.bias)

    model.to(torch_dtype)

    if cfg.rl_head_config.disable_dropout:
        replace_dropout_with_identity(model)

    return model