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

# convert hf model to middle_file
# middle_file:
#   params not in decoder: io.safetensors
#   params in decoder layer x: x.safetensors
#   use megatron name and style (fuse glu fc1)

import gc
import os

import safetensors.torch
import torch

from .config import ConvertorConfig
from .utils.fp8_utils import dict_push
from .utils.safetensors_loader import STLoaderLazy
from .utils.tensor_operations import CopyEquals, MergeGlu, MergeQKV, Operation


def get_megatron_iteration(convert_config: ConvertorConfig) -> int:
    if convert_config.iteration != -1:
        iteration = convert_config.iteration
    else:
        load_ckpt_ite_file = (
            f"{convert_config.load_path}/latest_checkpointed_iteration.txt"
        )
        if not os.path.exists(load_ckpt_ite_file):
            iteration = "release"
        else:
            with open(load_ckpt_ite_file, "r") as f:
                iteration = f.read()

    if iteration == "release":
        return -1
    else:
        return int(iteration)


class Save(Operation):
    def __init__(self, buffer_dict: dict, name: str, src: Operation):
        super().__init__()
        self.buffer_dict = buffer_dict
        self.name = name
        self.src = src

    def execute(self):
        tensor = self.src.execute()
        dict_push(self.buffer_dict, self.name, tensor)


class DictSaver:
    def __init__(self, buffer_dict=None, prefix=""):
        super().__init__()
        if buffer_dict is None:
            buffer_dict = {}
        self.buffer_dict = buffer_dict
        self.prefix = prefix

    def sub_saver(self, suffix):
        return DictSaver(self.buffer_dict, f"{self.prefix}{suffix}")

    def save(self, name: str, src: Operation):
        return Save(self.buffer_dict, f"{self.prefix}{name}", src)

    def dump_file(self, save_layer_path):
        safetensors.torch.save_file(self.buffer_dict, save_layer_path)
        with open(f"{save_layer_path}.done", "w") as f:
            f.write("done")


def convert_layer_loadsave(
    convert_config: ConvertorConfig,
    hfst_loader: STLoaderLazy,
    layer_idx: int,
    save_layer_path: str,
) -> None:
    saver = DictSaver()
    convert_layer(convert_config, hfst_loader, saver, layer_idx)
    saver.dump_file(save_layer_path)


def convert_layer(
    convert_config: ConvertorConfig,
    hfst_loader: STLoaderLazy,
    saver: DictSaver,
    layer_idx: int,
):
    num_layers = convert_config.num_layers
    moe = convert_config.mlp_type == "moe" and layer_idx >= convert_config.first_dense

    linear_trans = convert_config.linear_trans
    layernorm_trans = convert_config.layernorm_trans
    router_trans = convert_config.router_trans

    should_load_prefix: set[str] = set()
    actual_load_prefix: set[str] = set()
    model_strategy_map = {}
    layer_strategy_map = {}
    expert_strategy_map = {}

    if layer_idx == num_layers:
        should_load_prefix.update(
            (k for k in hfst_loader.keys() if k.startswith("model.embed_tokens."))
        )
        model_strategy_map.update(
            {
                "embedding.word_embeddings.weight": (
                    "copy",
                    linear_trans,
                    "model.embed_tokens.weight",
                ),
            }
        )
    elif layer_idx == num_layers + 1:
        should_load_prefix.update(
            (
                k
                for k in hfst_loader.keys()
                if (
                    k.startswith("model.")
                    and not k.startswith("model.layers.")
                    and not k.startswith("model.embed_tokens.")
                )
                or not k.startswith("model.")
            )
        )
        model_strategy_map.update(
            {
                "decoder.final_layernorm.weight": (
                    "copy",
                    layernorm_trans,
                    "model.norm.weight",
                ),
            }
        )
        if not convert_config.tie_word_embeddings:
            model_strategy_map.update(
                {
                    "output_layer.weight": ("copy", linear_trans, "lm_head.weight"),
                }
            )
    else:
        should_load_prefix.update(
            (
                k
                for k in hfst_loader.keys()
                if k.startswith(f"model.layers.{layer_idx}.")
            )
        )
        if convert_config.attn_type == "mla":
            layer_strategy_map.update(
                {
                    "input_layernorm.weight": (
                        "copy",
                        layernorm_trans,
                        "input_layernorm.weight",
                    ),
                    "self_attention.kv_a_layernorm.weight": (
                        "copy",
                        layernorm_trans,
                        "self_attn.kv_a_layernorm.weight",
                    ),
                    "self_attention.linear_kv_a_proj.weight": (
                        "copy",
                        linear_trans,
                        "self_attn.kv_a_proj_with_mqa.weight",
                    ),
                    "self_attention.linear_kv_b_proj.weight": (
                        "copy",
                        linear_trans,
                        "self_attn.kv_b_proj.weight",
                    ),
                    "self_attention.linear_proj.weight": (
                        "copy",
                        linear_trans,
                        "self_attn.o_proj.weight",
                    ),
                }
            )
            if convert_config.use_q_lora:
                layer_strategy_map.update(
                    {
                        "self_attention.linear_q_a_proj.weight": (
                            "copy",
                            linear_trans,
                            "self_attn.q_a_proj.weight",
                        ),
                        "self_attention.linear_q_b_proj.weight": (
                            "copy",
                            linear_trans,
                            "self_attn.q_b_proj.weight",
                        ),
                    }
                )
            else:
                layer_strategy_map.update(
                    {
                        "self_attention.linear_q_proj.weight": (
                            "copy",
                            linear_trans,
                            "self_attn.q_proj.weight",
                        ),
                    }
                )
            if convert_config.use_qk_norm:
                if convert_config.use_q_lora:
                    layer_strategy_map.update(
                        {
                            "self_attention.q_a_layernorm.weight": (
                                "copy",
                                layernorm_trans,
                                "self_attn.q_a_layernorm.weight",
                            ),
                        }
                    )
                layer_strategy_map.update(
                    {
                        "self_attention.k_layernorm.weight": (
                            "copy",
                            layernorm_trans,
                            "self_attn.k_a_layernorm.weight",
                        ),
                    }
                )
        elif convert_config.attn_type == "gqa":
            layer_strategy_map.update(
                {
                    "input_layernorm.weight": (
                        "copy",
                        layernorm_trans,
                        "input_layernorm.weight",
                    ),
                    "self_attention.linear_qkv.weight": (
                        "qkvw",
                        linear_trans,
                        "self_attn.q_proj.weight",
                        "self_attn.k_proj.weight",
                        "self_attn.v_proj.weight",
                    ),
                    "self_attention.linear_proj.weight": (
                        "copy",
                        linear_trans,
                        "self_attn.o_proj.weight",
                    ),
                }
            )
            if convert_config.use_qkv_bias:
                layer_strategy_map.update(
                    {
                        "self_attention.linear_qkv.bias": (
                            "qkvb",
                            linear_trans,
                            "self_attn.q_proj.bias",
                            "self_attn.k_proj.bias",
                            "self_attn.v_proj.bias",
                        ),
                    }
                )
            if convert_config.use_qk_norm:
                layer_strategy_map.update(
                    {
                        "self_attention.q_layernorm.weight": (
                            "copy",
                            layernorm_trans,
                            "self_attn.q_norm.weight",
                        ),
                        "self_attention.k_layernorm.weight": (
                            "copy",
                            layernorm_trans,
                            "self_attn.k_norm.weight",
                        ),
                    }
                )
        else:
            raise NotImplementedError(
                f"attn-type not support for {convert_config.attn_type}"
            )

        if not moe:
            layer_strategy_map.update(
                {
                    "pre_mlp_layernorm.weight": (
                        "copy",
                        linear_trans,
                        "post_attention_layernorm.weight",
                    ),
                    "mlp.linear_fc1.weight": (
                        "glu",
                        linear_trans,
                        "mlp.up_proj.weight",
                        "mlp.gate_proj.weight",
                    ),
                    "mlp.linear_fc2.weight": (
                        "copy",
                        linear_trans,
                        "mlp.down_proj.weight",
                    ),
                }
            )
        else:
            layer_strategy_map.update(
                {
                    "pre_mlp_layernorm.weight": (
                        "copy",
                        layernorm_trans,
                        "post_attention_layernorm.weight",
                    ),
                    "mlp.router.weight": ("copy", router_trans, "mlp.gate.weight"),
                }
            )
            expert_strategy_map.update(
                {
                    "linear_fc1.weight": (
                        "glu",
                        linear_trans,
                        "up_proj.weight",
                        "gate_proj.weight",
                    ),
                    "linear_fc2.weight": ("copy", linear_trans, "down_proj.weight"),
                }
            )
            if convert_config.use_expert_bias:
                layer_strategy_map.update(
                    {
                        "mlp.router.expert_bias": (
                            "copy",
                            router_trans,
                            "mlp.gate.e_score_correction_bias",
                        ),
                    }
                )
            if convert_config.use_shared_experts:
                shared_prefix = convert_config.hf_shared_experts_prefix
                layer_strategy_map.update(
                    {
                        "mlp.shared_expert.linear_fc1.weight": (
                            "glu",
                            linear_trans,
                            f"{shared_prefix}.up_proj.weight",
                            f"{shared_prefix}.gate_proj.weight",
                        ),
                        "mlp.shared_expert.linear_fc2.weight": (
                            "copy",
                            linear_trans,
                            f"{shared_prefix}.down_proj.weight",
                        ),
                    }
                )
                if convert_config.use_shared_experts_gate:
                    layer_strategy_map.update(
                        {
                            "mlp.shared_expert_gate.weight": (
                                "copy",
                                linear_trans,
                                "mlp.shared_expert_gate.weight",
                            ),
                        }
                    )

    operations: list[Operation] = []

    for mg_name, (opstr, dtype_trans, *hf_names) in model_strategy_map.items():
        actual_load_prefix.update((hf_name for hf_name in hf_names))
        if opstr == "copy":
            (hf_name,) = hf_names
            operations.append(
                saver.save(mg_name, hfst_loader.load(hf_name, dtype_trans))
            )
        elif opstr == "copy_equal":
            operations.append(
                saver.save(
                    mg_name,
                    CopyEquals(
                        [hfst_loader.load(hf_name, dtype_trans) for hf_name in hf_names]
                    ),
                )
            )
        else:
            assert False

    loader_decoder = hfst_loader.sub_loader(f"model.layers.{layer_idx}.")
    saver_decoder = saver.sub_saver(f"decoder.layers.{layer_idx}.")
    for mg_name, (opstr, dtype_trans, *hf_names) in layer_strategy_map.items():
        actual_load_prefix.update(
            (f"model.layers.{layer_idx}.{hf_name}" for hf_name in hf_names)
        )
        if opstr == "copy":
            (hf_name,) = hf_names
            operations.append(
                saver_decoder.save(mg_name, loader_decoder.load(hf_name, dtype_trans))
            )
        elif opstr == "glu":
            (hf_fc1_name, hf_gate_name) = hf_names
            operations.append(
                saver_decoder.save(
                    mg_name,
                    MergeGlu(
                        loader_decoder.load(hf_gate_name, dtype_trans),
                        loader_decoder.load(hf_fc1_name, dtype_trans),
                    ),
                )
            )
        elif opstr == "qkvw":
            (hf_q_name, hf_k_name, hf_v_name) = hf_names
            num_query_groups, num_attention_heads, head_dim = (
                convert_config.num_query_groups,
                convert_config.num_attention_heads,
                convert_config.head_dim,
            )
            operations.append(
                saver_decoder.save(
                    mg_name,
                    MergeQKV(
                        loader_decoder.load(hf_q_name, dtype_trans),
                        loader_decoder.load(hf_k_name, dtype_trans),
                        loader_decoder.load(hf_v_name, dtype_trans),
                        num_query_groups,
                        num_attention_heads,
                        head_dim,
                        "w",
                    ),
                )
            )
        elif opstr == "qkvb":
            (hf_q_name, hf_k_name, hf_v_name) = hf_names
            num_query_groups, num_attention_heads, head_dim = (
                convert_config.num_query_groups,
                convert_config.num_attention_heads,
                convert_config.head_dim,
            )
            operations.append(
                saver_decoder.save(
                    mg_name,
                    MergeQKV(
                        loader_decoder.load(hf_q_name, dtype_trans),
                        loader_decoder.load(hf_k_name, dtype_trans),
                        loader_decoder.load(hf_v_name, dtype_trans),
                        num_query_groups,
                        num_attention_heads,
                        head_dim,
                        "b",
                    ),
                )
            )
        else:
            assert False

    for expert_index in range(convert_config.num_experts):
        loader_expert = loader_decoder.sub_loader(f"mlp.experts.{expert_index}.")
        saver_expert = saver_decoder.sub_saver(
            f"mlp.experts.local_experts.{expert_index}."
        )
        for mg_name, (opstr, dtype_trans, *hf_names) in expert_strategy_map.items():
            actual_load_prefix.update(
                (
                    f"model.layers.{layer_idx}.mlp.experts.{expert_index}.{hf_name}"
                    for hf_name in hf_names
                )
            )
            if opstr == "copy":
                (hf_name,) = hf_names
                operations.append(
                    saver_expert.save(mg_name, loader_expert.load(hf_name, dtype_trans))
                )
            elif opstr == "glu":
                (hf_fc1_name, hf_gate_name) = hf_names
                operations.append(
                    saver_expert.save(
                        mg_name,
                        MergeGlu(
                            loader_expert.load(hf_gate_name, dtype_trans),
                            loader_expert.load(hf_fc1_name, dtype_trans),
                        ),
                    )
                )
            else:
                assert False

    should_load_but = should_load_prefix.difference(actual_load_prefix)
    actual_load_but = actual_load_prefix.difference(should_load_prefix)
    if should_load_but or actual_load_but:
        raise RuntimeError(
            f"Layer {layer_idx}: should_load_but: {should_load_but}; actual_load_but: {actual_load_but}"
        )
    for op in operations:
        op.execute()

    gc.collect()
    if Operation.global_device != "cpu":
        torch.cuda.empty_cache()
