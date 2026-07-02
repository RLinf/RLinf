# Copyright 2025 The RLinf Authors.
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

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass
from omegaconf import DictConfig


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    from pathlib import Path

    from rlinf.utils.patcher import Patcher

    npu_available = hasattr(torch, "npu") and torch.npu.is_available()

    Patcher.clear()
    Patcher.add_patch(
        "gr00t.data.embodiment_tags.EmbodimentTag",
        "rlinf.models.embodiment.gr00t.embodiment_tags.EmbodimentTag",
    )
    Patcher.add_patch(
        "gr00t.data.embodiment_tags.EMBODIMENT_TAG_MAPPING",
        "rlinf.models.embodiment.gr00t.embodiment_tags.EMBODIMENT_TAG_MAPPING",
    )
    if npu_available:
        Patcher.add_patch(
            "transformers.models.qwen3.modeling_qwen3.apply_rotary_pos_emb",
            "rlinf.models.embodiment.gr00t.gr00t_n1d5.npu_patches.npu_apply_rotary_pos_emb",
        )
        Patcher.add_patch(
            "transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm.forward",
            "rlinf.models.embodiment.gr00t.gr00t_n1d5.npu_patches.npu_rmsnorm_forward",
        )
        # RADIO queries CUDA capability unconditionally during import. On Ascend,
        # provide its minimum accepted value as a compatibility sentinel.
        Patcher.add_patch(
            "torch.cuda.get_device_capability",
            "rlinf.models.embodiment.gr00t.gr00t_n1d5.npu_patches.get_radio_compatible_cuda_capability_on_npu",
        )
        Patcher.skip_import("flash_attn")
    Patcher.apply()

    from gr00t.experiment.data_config import load_data_config

    from rlinf.models.embodiment.gr00t.gr00t_n1d5.gr00t_action_model import (
        GR00T_N1_5_ForRLActionPrediction,
    )
    from rlinf.models.embodiment.gr00t.utils import replace_dropout_with_identity

    if cfg.embodiment_tag == "libero_franka" or cfg.embodiment_tag == "isaaclab_franka":
        data_config = load_data_config(
            "rlinf.models.embodiment.gr00t.gr00t_n1d5.modality_config:LiberoFrankaDataConfig"
        )
    elif cfg.embodiment_tag == "maniskill_widowx":
        data_config = load_data_config(
            "rlinf.models.embodiment.gr00t.gr00t_n1d5.modality_config:ManiskillWidowXDataConfig"
        )
    else:
        raise ValueError(f"Invalid embodiment tag: {cfg.embodiment_tag}")
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    # The transformer rigisteration is done in gr00t/model/gr00t_n1.py
    model_path = Path(cfg.model_path)
    if not model_path.exists():
        # raise error or it triggers auto download from hf(It's cool but we don't have internet connection.)
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    autoconfig_cls = None
    original_autoconfig_descriptor = None
    if npu_available:
        from transformers import AutoConfig

        autoconfig_cls = AutoConfig
        original_autoconfig_descriptor = AutoConfig.__dict__["from_pretrained"]
        original_autoconfig_from_pretrained = AutoConfig.from_pretrained

        def _from_pretrained_and_clear_flash_attn_stub(cls, *args, **kwargs):
            config = original_autoconfig_from_pretrained(*args, **kwargs)
            if getattr(config, "model_type", None) == "eagle_2_5_vl":
                Patcher.clear_stub_import("flash_attn")
            return config

        AutoConfig.from_pretrained = classmethod(
            _from_pretrained_and_clear_flash_attn_stub
        )

    try:
        model = GR00T_N1_5_ForRLActionPrediction.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            embodiment_tag=cfg.embodiment_tag,  # This tag determines the state encoder and action head to use
            modality_config=modality_config,
            modality_transform=modality_transform,
            denoising_steps=cfg.denoising_steps,
            output_action_chunks=cfg.num_action_chunks,
            obs_converter_type=cfg.obs_converter_type,  # TODO(lx): unify the embodiment data format and obs converter
            tune_visual=False,
            tune_llm=False,
            rl_head_config=cfg.rl_head_config,
        )
    finally:
        if autoconfig_cls is not None:
            autoconfig_cls.from_pretrained = original_autoconfig_descriptor
        if npu_available:
            Patcher.clear_stub_import("flash_attn")
    model.to(torch_dtype)
    if cfg.rl_head_config.add_value_head:
        # reinitialize the value head after model loading, or there are nan values in the value head after model loading.
        model.action_head.value_head._init_weights()

    if cfg.rl_head_config.disable_dropout:
        replace_dropout_with_identity(model)

    return model
