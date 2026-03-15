# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_loader/loader.py

from abc import ABC

import torch
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    PreTrainedModel,
)
from transformers.modeling_utils import no_init_weights

from ..utils import logging
from ..utils.import_utils import is_torch_npu_available, is_vescale_available
from .module_utils import init_empty_weights, load_model_weights
from .registry import get_registry

logger = logging.get_logger(__name__)


class BaseModelLoader(ABC):
    def __init__(self):
        pass

    def load_model(self, model_config, **kwargs):
        raise NotImplementedError


class HuggingfaceLoader(BaseModelLoader):
    def __init__(self):
        super().__init__()

    def load_model(self, init_kwargs: dict, **kwargs):
        model_config = init_kwargs["config"]
        architecture = _get_model_arch_from_config(model_config)

        if (
            type(model_config) in AutoModelForVision2Seq._model_mapping.keys()
        ):  # assume built-in models
            load_class = AutoModelForVision2Seq
        elif (
            "ForCausalLM" in architecture
            and type(model_config) in AutoModelForCausalLM._model_mapping.keys()
        ):
            load_class = AutoModelForCausalLM
        else:
            load_class = AutoModel

        init_device = kwargs.pop("init_device", "cuda")
        weights_path = kwargs.pop("weights_path", None)
        empty_init = kwargs.pop("empty_init", False)

        logger.info_rank0(
            f"Loading model from Huggingface modeling.\n"
            f"init_device: {init_device}\n"
            f"empty_init: {empty_init}\n"
            f"weights_path: {weights_path}"
        )

        if weights_path is None:  # init empty model from config
            if is_torch_npu_available() and init_device == "cuda":
                init_device = "npu"
            if init_device == "meta":
                with torch.device(init_device), no_init_weights():
                    logger.info_rank0(
                        "Init empty model on meta device from config without init_weights."
                    )
                    model = load_class.from_config(**init_kwargs)
            else:
                with torch.device(init_device):
                    logger.info_rank0("Init empty model from config.")
                    model = load_class.from_config(**init_kwargs)
        else:
            if is_vescale_available() and init_device == "meta":
                from vescale.initialize.meta_init import meta_device_init

                with meta_device_init():
                    model = load_class.from_config(**init_kwargs)
            else:
                with init_empty_weights(), no_init_weights():
                    model = load_class.from_config(**init_kwargs)
            if not empty_init:
                load_model_weights(model, weights_path, init_device)

        return model


class CustomizedModelingLoader(BaseModelLoader):
    def __init__(self, model_cls: PreTrainedModel):
        super().__init__()
        self.model_cls = model_cls  # model class from code_path

    def load_model(self, init_kwargs: dict, **kwargs):
        init_kwargs.pop("trust_remote_code", True)

        init_device = kwargs.pop("init_device", "cuda")
        weights_path = kwargs.pop("weights_path", None)
        empty_init = kwargs.pop("empty_init", False)
        vlm_repo_id = kwargs.pop("vlm_repo_id", None)
        enable_expert_vision = kwargs.pop("enable_expert_vision", False)
        expert_vision_path = kwargs.pop("expert_vision_path", None)
        post_training = kwargs.pop("post_training", False)
        adanorm_time = kwargs.pop("adanorm_time", False)
        incremental_training = kwargs.pop("incremental_training", False)
        depth_incremental_training = kwargs.pop("depth_incremental_training", False)
        norm_qkv = kwargs.pop("norm_qkv", False)

        logger.info_rank0(
            f"Loading model from customized modeling.\n"
            f"init_device: {init_device}\n"
            f"empty_init: {empty_init}\n"
            f"weights_path: {weights_path}"
        )

        if weights_path is None:  # init empty model from config
            if is_torch_npu_available() and init_device == "cuda":
                init_device = "npu"
            if init_device == "meta":
                with torch.device(init_device), no_init_weights():
                    logger.info_rank0(
                        "Init empty model on meta device from config without init_weights."
                    )
                    model = self.model_cls._from_config(**init_kwargs)
            else:
                with torch.device(init_device):
                    logger.info_rank0("Init empty model from config.")
                    model = self.model_cls._from_config(**init_kwargs)
        else:
            load_vlm_only = False
            if is_vescale_available() and init_device == "meta":
                from vescale.initialize.meta_init import meta_device_init

                with meta_device_init():
                    model = self.model_cls._from_config(**init_kwargs)
            else:
                with init_empty_weights(), no_init_weights():
                    if (
                        self.model_cls.__name__ == "PI0Policy"
                        and "modeling_pi0" in self.model_cls.__module__
                    ):
                        model = self.model_cls(
                            config=init_kwargs["config"],
                            tokenizer_path=init_kwargs["config"].tokenizer_path,
                        ).to(init_kwargs["torch_dtype"])
                        if vlm_repo_id is not None:
                            load_vlm_only = True
                    elif (
                        self.model_cls.__name__ == "LingbotVlaPolicy"
                        and "modeling_lingbot_vla" in self.model_cls.__module__
                    ):
                        model = self.model_cls(
                            config=init_kwargs["config"],
                            tokenizer_path=init_kwargs["config"].tokenizer_path,
                        ).to(init_kwargs["torch_dtype"])
                        if vlm_repo_id is not None and incremental_training:
                            load_vlm_only = True
                    else:
                        model = self.model_cls._from_config(**init_kwargs)

            if not empty_init:
                load_model_weights(
                    model,
                    weights_path,
                    init_device,
                    load_vlm_only=load_vlm_only,
                    enable_expert_vision=enable_expert_vision,
                    expert_vision_path=expert_vision_path,
                    post_training=post_training,
                    incremental_training=incremental_training,
                    depth_incremental_training=depth_incremental_training,
                    norm_qkv=norm_qkv,
                    adanorm_time=adanorm_time,
                )

            # we should tie embeddings after loading weights because init_empty_weights() leads to untied weights,
            if getattr(model.config, "tie_word_embeddings", True):
                try:
                    input_embeddings = model.get_input_embeddings()
                    output_embeddings = model.get_output_embeddings()
                    output_embeddings._parameters["weight"] = (
                        input_embeddings._parameters["weight"]
                    )
                except Exception as e:
                    logger.info_rank0(f"Failed to tie embeddings: {e}")

        return model


def _get_model_arch_from_config(model_config):
    arch_name = model_config.architectures
    if isinstance(arch_name, list):
        arch_name = arch_name[0]
    return arch_name


def get_loader(model_config, force_use_huggingface):
    if isinstance(model_config, PI0Config):
        if "qwen" not in model_config.tokenizer_path.lower():
            model_arch = "PI0Policy"
        elif "qwen2" in model_config.tokenizer_path.lower():
            model_arch = "LingbotVlaPolicy"
    else:
        model_arch = _get_model_arch_from_config(
            model_config
        )  # Qwen2VLForConditionalGeneration
    loader = HuggingfaceLoader()
    if not force_use_huggingface:
        model_registry = get_registry()
        if model_arch in model_registry.supported_models:
            model_cls = model_registry.get_model_cls_from_model_arch(model_arch)
            loader = CustomizedModelingLoader(model_cls=model_cls)

    return loader
