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


import os
from dataclasses import dataclass
from typing import List, Optional

import torch
import yaml
from omegaconf.dictconfig import DictConfig
from transformers import AutoConfig


@dataclass
class ConvertorConfig:
    # from HuggingFace config and ckpt_cfg
    load_path: str = None
    save_path: str = None
    num_layers: int = None
    num_attn_heads: int = None
    num_query_groups: int = None
    head_dim: int = None
    tie_word_embeddings: bool = False
    model_type: str = None
    model: str = None

    # from default_args.yaml
    attn_type: str = None
    use_q_lora: bool = None
    use_qkv_bias: bool = None
    use_qk_norm: bool = None
    mlp_type: str = None
    num_experts: Optional[int] = None
    use_shared_experts: bool = False
    hf_share_experts_prefix: Optional[str] = None
    use_expert_bias: bool = False
    use_shared_experts_gate: bool = False
    first_dense: int = 0

    # from ckpt_cfg
    iteration: int = -1
    process_num: int = 1
    tp_size: int = 1
    tpe_size: int = 1
    ep_size: int = 1
    pp_size: int = 1
    schedular: str = "1f1b"
    pp_stages: Optional[List[int]] = None

    # from runtime parameters
    use_gpu_num: int = 0
    use_gpu_index: Optional[List[int]] = None

    # precision
    linear_trans: str = "auto"
    layernorm_trans: str = "auto"
    router_trans: str = "auto"

    # special for mg
    te_ln_linear_qkv: bool = False
    te_ln_linear_mlp_fc1: bool = False
    te_ln_add_extra_state: bool = False


def load_convertor_config(hf_ckpt_path: str, ckpt_cfg: DictConfig) -> ConvertorConfig:
    hf_config = AutoConfig.from_pretrained(hf_ckpt_path, trust_remote_code=True)
    convertor_config = ConvertorConfig()
    convertor_config.load_path = hf_ckpt_path

    convertor_config.num_layers = hf_config.num_hidden_layers
    convertor_config.num_attn_heads = hf_config.num_attention_heads
    # num_key_value_heads is gqa/mqa num_groups
    num_kv_heads = getattr(
        hf_config, "num_key_value_heads", hf_config.num_attention_heads
    )
    convertor_config.num_query_groups = getattr(
        hf_config, "num_query_groups", num_kv_heads
    )

    convertor_config.head_dim = getattr(hf_config, "head_dim", None)
    if convertor_config.head_dim is None:
        convertor_config.head_dim = (
            hf_config.hidden_size // hf_config.num_attention_heads
        )

    assert (
        convertor_config.num_query_groups is not None
        and convertor_config.num_query_groups > 0
    ), "num_query_groups must be specified and greater than 0."
    assert (
        convertor_config.num_attn_heads is not None
        and convertor_config.num_attn_heads > 0
    ), "num_attn_heads must be specified and greater than 0."

    if ckpt_cfg.use_gpu_num is None:
        convertor_config.use_gpu_num = torch.cuda.device_count()
    else:
        convertor_config.use_gpu_num = ckpt_cfg.use_gpu_num
    if ckpt_cfg.use_gpu_index is not None:
        assert len(ckpt_cfg.use_gpu_index) == convertor_config.use_gpu_num, (
            "use_gpu_index length must match use_gpu_num"
        )

    if ckpt_cfg.model_type is None:
        convertor_config.model_type = hf_config.model_type
    else:
        convertor_config.model_type = ckpt_cfg.model_type

    script_path = os.path.dirname(os.path.abspath(__file__))
    with open(f"{script_path}/default_args.yaml") as default_args_file:
        default_args = yaml.safe_load(default_args_file)

    model_defaults = default_args["explict_model"]
    model_type_defaults = default_args["model_type"]

    assert ckpt_cfg.model is not None, "model must be specified in config file."

    convertor_config.model = ckpt_cfg.model
    assert convertor_config.model in model_defaults, (
        f"Model {convertor_config.model} not found in supported list."
    )
    convertor_config.model_type = model_defaults[convertor_config.model]["model_type"]
    for key, value in model_defaults[convertor_config.model].items():
        setattr(convertor_config, key, value)

    if ckpt_cfg.model_type is not None:
        assert (
            convertor_config.model_type
            == model_defaults[convertor_config.model]["model_type"]
        ), (
            f"Model type {convertor_config.model_type} does not match expected {model_defaults[convertor_config.model]['model_type']} for model {convertor_config.model}."
        )

    assert convertor_config.model_type in model_type_defaults, (
        f"Model type {convertor_config.model_type} not found in supported list."
    )

    for key, value in model_type_defaults[convertor_config.model_type].items():
        setattr(convertor_config, key, value)

    assert convertor_config.attn_type is not None

    if convertor_config.attn_type == "mla":
        assert (
            convertor_config.use_q_lora is None
            and convertor_config.num_attn_heads is None
            and convertor_config.num_query_groups is None
        )
    elif convertor_config.attn_type == "gqa":
        assert convertor_config.use_q_lora is None
        if convertor_config.use_qkv_bias is None:
            convertor_config.use_qkv_bias = False
        if convertor_config.use_qk_norm is None:
            pass
    convertor_config.process_num = ckpt_cfg.process_num
    convertor_config.save_path = ckpt_cfg.save_path
    convertor_config.tp_size = ckpt_cfg.tensor_model_parallel_size
    convertor_config.pp_size = ckpt_cfg.pipeline_model_parallel_size

    if convertor_config.mlp_type == "moe":
        if convertor_config.use_shared_experts is None:
            convertor_config.use_shared_experts = False
        if convertor_config.use_shared_experts:
            if convertor_config.use_expert_bias is None:
                convertor_config.use_expert_bias = False
            if convertor_config.use_shared_experts_gate is None:
                convertor_config.use_shared_experts_gate = False
            assert convertor_config.hf_share_experts_prefix is not None

        if convertor_config.first_dense is None:
            convertor_config.first_dense = 0
        if convertor_config.num_experts is None:
            convertor_config.num_experts = getattr(hf_config, "num_experts", None)
        assert convertor_config.num_experts is not None
    else:
        assert convertor_config.num_experts in (0, None)
        convertor_config.num_experts = 0

    if convertor_config.tie_word_embeddings is None:
        convertor_config.tie_word_embeddings = getattr(
            hf_config, "tie_word_embeddings", False
        )

    if convertor_config.num_layers is None:
        convertor_config.num_layers = getattr(hf_config, "num_hidden_layers", None)
    assert (
        convertor_config.num_layers is not None and convertor_config.num_layers > 0
    ), "num_layers must be specified and greater than 0."

    return convertor_config
