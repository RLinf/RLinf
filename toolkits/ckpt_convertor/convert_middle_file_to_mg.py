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

# convert mg model from middle_file
# middle_file:
#   params not in decoder: io.safetensors
#   params in decoder layer x: x.safetensors
#   use megatron name and style (fuse glu fc1)

import gc
import os
import shutil
from collections import OrderedDict

import torch

from toolkits.ckpt_convertor.config import ConvertorConfig

from .utils.fp8_utils import dict_push
from .utils.mg_moe_groupgemm import moe_seq_to_group, moe_seq_to_te_group
from .utils.safetensors_loader import STLoader, STLoaderLazy
from .utils.tensor_operations import Operation, SplitTpTpe


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


def get_hetero_pp_rank(
    convert_config: ConvertorConfig, num_layers: int, layer_idx: int
) -> tuple[int, int, int]:
    if convert_config.schedular == "1f1b":
        if convert_config.pp_stages is not None:
            pp_stages = convert_config.pp_stages
            assert len(pp_stages) == convert_config.pp_size
            assert sum(pp_stages) == num_layers
            for pp_rank in range(1000):
                local_num_layers = pp_stages[pp_rank]
                layer_offset = sum(([0] + pp_stages)[: pp_rank + 1])
                if layer_offset <= layer_idx < layer_offset + local_num_layers:
                    return pp_rank, layer_idx - layer_offset, local_num_layers
        else:
            assert num_layers % convert_config.pp_size == 0, (
                "cannot split layers to pp. please use --pp-satges"
            )
            local_num_layers = int(num_layers // convert_config.pp_size)
            if layer_idx == 0:
                return 0, 0, 0
            pp_rank = layer_idx // local_num_layers
            local_layer = layer_idx % local_num_layers
            return int(pp_rank), int(local_layer), local_num_layers
    elif convert_config.schedular == "dualpipev":
        if convert_config.pp_stages is not None:
            ppvpp_stages = convert_config.pp_stages
            assert len(ppvpp_stages) == convert_config.pp_size * 2
            assert sum(ppvpp_stages) == num_layers
            # reverse vpp rank 1
            ppvpp_stages = (
                ppvpp_stages[: convert_config.pp_size]
                + ppvpp_stages[: convert_config.pp_size - 1 : -1]
            )
            for ppvpp_rank in range(1000):
                local_num_layers = ppvpp_stages[ppvpp_rank]
                layer_offset = sum(([0] + ppvpp_stages)[: ppvpp_rank + 1])
                if layer_offset <= layer_idx < layer_offset + local_num_layers:
                    return ppvpp_rank, layer_idx - layer_offset, local_num_layers
        else:
            assert num_layers % (convert_config.pp_size * 2) == 0, (
                "cannot split layers to vpp. please use --pp-satges"
            )
            local_num_layers = int(num_layers // convert_config.pp_size // 2)
            if layer_idx == 0:
                return 0, 0, 0
            ppvpp_rank = layer_idx // local_num_layers
            local_layer = layer_idx % local_num_layers
            return int(ppvpp_rank), int(local_layer), local_num_layers
    elif convert_config.schedular == "vpp":
        raise NotImplementedError()
    else:
        raise ValueError()


class Save(Operation):
    def __init__(self, full_checkpoint, name, strategy, src, pp_rank, model_key_vpp):
        super().__init__()
        self.full_checkpoint = full_checkpoint
        self.name = name
        self.strategy = strategy
        self.src = src
        self.pp_rank = pp_rank
        self.model_key_vpp = model_key_vpp

    @staticmethod
    def gen_checkpoint_name(pp_rank, ep_rank_input):
        target_tp = Operation.global_tp
        target_tpe = Operation.global_tpe
        target_ep = Operation.global_ep
        target_pp = Operation.global_pp

        target_size_wopp = max(target_tpe * target_ep, target_tp)
        for model_rank in range(target_size_wopp):
            tp_rank = model_rank % target_tp
            tpe_rank = model_rank % target_tpe
            if ep_rank_input is not None:
                if ep_rank_input != (model_rank // target_tpe) % target_ep:
                    continue
                ep_rank = ep_rank_input
            else:
                ep_rank = (model_rank // target_tpe) % target_ep

            if target_tpe > target_tp:
                assert False, (
                    f"target_tpe > target_tp: {target_tpe} > {target_tp} megatron can't load ckpt"
                )
            else:
                if target_pp == 1:
                    key = f"mp_rank_{tp_rank:02d}"
                else:
                    key = f"mp_rank_{tp_rank:02d}_{pp_rank:03d}"
                if target_ep > 1:
                    key += f"_{ep_rank:03d}"
                yield key, tp_rank, tpe_rank

    def execute(self):
        value = self.src.execute()
        if self.strategy == "copy":
            for key, tp_rank, tpe_rank in Save.gen_checkpoint_name(self.pp_rank, None):
                if key in self.full_checkpoint:
                    dict_push(
                        self.full_checkpoint[key][self.model_key_vpp], self.name, value
                    )
        elif self.strategy == "tp":
            for key, tp_rank, tpe_rank in Save.gen_checkpoint_name(self.pp_rank, None):
                if key in self.full_checkpoint:
                    dict_push(
                        self.full_checkpoint[key][self.model_key_vpp],
                        self.name,
                        value[tp_rank],
                    )
        elif isinstance(self.strategy, int):  # ep
            ep_rank = self.strategy
            for key, tp_rank, tpe_rank in Save.gen_checkpoint_name(
                self.pp_rank, ep_rank
            ):
                if key in self.full_checkpoint:
                    dict_push(
                        self.full_checkpoint[key][self.model_key_vpp],
                        self.name,
                        value[tpe_rank],
                    )
        else:
            assert False


class CKPTSaver:
    def __init__(self, pp_rank, full_checkpoint, prefix="", save_dir=None):
        super().__init__()
        self.pp_rank = pp_rank
        self.prefix = prefix
        self.save_dir = save_dir
        self.full_checkpoint = full_checkpoint

    @staticmethod
    def reset_ckpt_name(save_dir, pp_rank, vpp_size=None):
        full_checkpoint = {}
        for key, *_ in Save.gen_checkpoint_name(pp_rank, None):
            if os.path.exists(f"{save_dir}/{key}.done"):
                continue
            if os.path.exists(f"{save_dir}/{key}"):
                shutil.rmtree(f"{save_dir}/{key}")
            checkpoint = OrderedDict()
            if vpp_size is None:
                checkpoint["model"] = OrderedDict()
            else:
                for i in range(vpp_size):
                    checkpoint[f"model{i}"] = OrderedDict()
            full_checkpoint[key] = checkpoint
        return full_checkpoint

    def save_ckpts(self):
        for k, v in self.full_checkpoint.items():
            CKPTSaver.save_ckpt_one(self.save_dir, k, v)
        self.full_checkpoint.clear()

    @staticmethod
    def save_ckpt_one(save_dir, name, save_ckpt):
        save_path = os.path.join(save_dir, name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, "model_optim_rng.pt")
        torch.save(save_ckpt, save_path)
        with open(f"{save_dir}/{name}.done", "w") as f:
            f.write("done")

    def sub_saver(self, suffix=""):
        return CKPTSaver(
            self.pp_rank, self.full_checkpoint, prefix=f"{self.prefix}{suffix}"
        )

    def save(self, model_key_vpp, name, strategy, src) -> Save:
        return Save(
            self.full_checkpoint,
            f"{self.prefix}{name}",
            strategy,
            src,
            self.pp_rank,
            model_key_vpp,
        )


def merge_checkpoint_dict(full_checkpoints, to_cpu=False):
    all_ckpts = {}
    for ckpt_one in full_checkpoints:
        for ckpt_key, ckpt_value in ckpt_one.items():
            if ckpt_key not in all_ckpts:
                all_ckpts[ckpt_key] = {}
            for model_key, model_value in ckpt_value.items():
                if model_key not in all_ckpts[ckpt_key]:
                    all_ckpts[ckpt_key][model_key] = {}
                for key in list(model_value.keys()):
                    assert key not in all_ckpts[ckpt_key][model_key]
                    value = model_value.pop(key)
                    if to_cpu:
                        value = value.to("cpu")
                    all_ckpts[ckpt_key][model_key][key] = value
        gc.collect()
        if Operation.global_device != "cpu":
            torch.cuda.empty_cache()
    return all_ckpts


def convert_layer_load(
    convert_config: ConvertorConfig,
    layer_info: tuple[str, int, int, int],
    pp_rank: int,
    full_checkpoint: list,
):
    pp_saver = CKPTSaver(pp_rank, full_checkpoint)
    model_key_vpp, layer, local_layer, local_num_layer = layer_info
    st_loader = STLoaderLazy.from_file(
        [f"{convert_config.load_path}/{layer}.safetensors"], check_done=True
    )
    st_loader_pre, st_loader_post = None, None
    if layer == 0:
        st_loader_pre = STLoaderLazy.from_file(
            [f"{convert_config.load_path}/pre.safetensors"], check_done=True
        )
    if layer == convert_config.num_layers - 1:
        st_loader_post = STLoaderLazy.from_file(
            [f"{convert_config.load_path}/post.safetensors"], check_done=True
        )

    convert_layer(
        convert_config,
        st_loader,
        st_loader_pre,
        st_loader_post,
        pp_saver,
        model_key_vpp,
        layer,
        local_layer,
    )
    return pp_saver.full_checkpoint


def convert_layer(
    convert_config: ConvertorConfig,
    st_loader: STLoader,
    st_loader_pre: STLoader,
    st_loader_post: STLoader,
    saver: CKPTSaver,
    model_key_vpp: str,
    layer_idx: int,
    local_layer: int,
) -> None:
    num_layers = convert_config.num_layers
    moe = convert_config.mlp_type == "moe" and layer_idx >= convert_config.first_dense
    num_experts = convert_config.num_experts
    num_local_experts = convert_config.num_experts // convert_config.ep_size

    linear_trans = convert_config.linear_trans
    layernorm_trans = convert_config.layernorm_trans
    router_trans = convert_config.router_trans

    should_load_prefix: set[str] = set()
    actual_load_prefix: set[str] = set()
    pre_strategy_map = {}
    post_strategy_map = {}
    layer_strategy_map = {}
    expert_strategy_map = {}

    if layer_idx == 0:
        should_load_prefix.update(st_loader_pre.keys())
        pre_strategy_map.update(
            {
                "embedding.word_embeddings.weight": ("dense_fc1", linear_trans),
            }
        )

    if layer_idx == num_layers - 1:
        should_load_prefix.update(st_loader_post.keys())
        post_strategy_map.update(
            {
                "decoder.final_layernorm.weight": ("copy", layernorm_trans),
            }
        )
        if not convert_config.tie_word_embeddings:
            post_strategy_map.update(
                {
                    "output_layer.weight": ("dense_fc1", linear_trans),
                }
            )

    should_load_prefix.update(st_loader.keys())
    if convert_config.attn_type == "mla":
        layer_strategy_map.update(
            {
                "input_layernorm.weight": ("copy", layernorm_trans),
                "self_attention.linear_kv_a_proj.weight": ("copy", linear_trans),
                "self_attention.linear_kv_b_proj.weight": ("dense_fc1", linear_trans),
                "self_attention.linear_proj.weight": ("dense_fc2", linear_trans),
            }
        )
        if convert_config.use_q_lora:
            layer_strategy_map.update(
                {
                    "self_attention.linear_q_a_proj.weight": ("copy", linear_trans),
                    "self_attention.linear_q_b_proj.weight": (
                        "dense_fc1",
                        linear_trans,
                    ),
                }
            )
        else:
            layer_strategy_map.update(
                {
                    "self_attention.linear_q_proj.weight": ("dense_fc1", linear_trans),
                }
            )
        if convert_config.use_qk_norm:
            if convert_config.use_q_lora:
                layer_strategy_map.update(
                    {
                        "self_attention.q_a_layernorm.weight": (
                            "copy",
                            layernorm_trans,
                        ),
                    }
                )
            layer_strategy_map.update(
                {
                    "self_attention.kv_a_layernorm.weight": ("copy", layernorm_trans),
                }
            )
    elif convert_config.attn_type == "gqa":
        if convert_config.te_ln_linear_qkv:
            layer_strategy_map.update(
                {
                    "input_layernorm.weight": (
                        "rename_copy:self_attention.linear_qkv.layer_norm_weight",
                        layernorm_trans,
                    ),
                }
            )
        else:
            layer_strategy_map.update(
                {
                    "input_layernorm.weight": ("copy", layernorm_trans),
                }
            )
        layer_strategy_map.update(
            {
                "self_attention.linear_qkv.weight": ("qkv_w", linear_trans),
                "self_attention.linear_proj.weight": ("dense_fc2", linear_trans),
            }
        )
        if convert_config.use_qkv_bias:
            layer_strategy_map.update(
                {
                    "self_attention.linear_qkv.bias": ("qkv_b", linear_trans),
                }
            )
        if convert_config.use_qk_norm:
            layer_strategy_map.update(
                {
                    "self_attention.q_layernorm.weight": ("copy", layernorm_trans),
                    "self_attention.k_layernorm.weight": ("copy", layernorm_trans),
                }
            )
    else:
        raise NotImplementedError(
            f"attn-type not support for {convert_config.attn_type}"
        )

    if not moe:
        if convert_config.te_ln_linear_mlp_fc1:
            layer_strategy_map.update(
                {
                    "pre_mlp_layernorm.weight": (
                        "rename_copy:mlp.linear_fc1.layer_norm_weight",
                        linear_trans,
                    ),
                }
            )
        else:
            layer_strategy_map.update(
                {
                    "pre_mlp_layernorm.weight": ("copy", layernorm_trans),
                }
            )
        layer_strategy_map.update(
            {
                "mlp.linear_fc1.weight": ("dense_fc1_glu", linear_trans),
                "mlp.linear_fc2.weight": ("dense_fc2", linear_trans),
            }
        )
    else:
        layer_strategy_map.update(
            {
                "pre_mlp_layernorm.weight": ("copy", layernorm_trans),
                "mlp.router.weight": ("copy", router_trans),
            }
        )
        expert_strategy_map.update(
            {
                "linear_fc1.weight": ("moe_fc1_glu", linear_trans),
                "linear_fc2.weight": ("moe_fc2", linear_trans),
            }
        )
        if convert_config.use_expert_bias:
            layer_strategy_map.update(
                {
                    "mlp.router.expert_bias": ("copy", router_trans),
                }
            )
        if convert_config.use_shared_experts:
            layer_strategy_map.update(
                {
                    "mlp.shared_expert.linear_fc1.weight": (
                        "dense_fc1_glu",
                        linear_trans,
                    ),
                    "mlp.shared_expert.linear_fc2.weight": ("dense_fc2", linear_trans),
                }
            )
            if convert_config.use_shared_experts_gate:
                layer_strategy_map.update(
                    {
                        "mlp.shared_expert_gate.weight": ("copy", router_trans),
                    }
                )

    operations: list[Operation] = []

    actual_load_prefix.update(pre_strategy_map.keys())
    for mg_name, (opstr, dtype_trans) in pre_strategy_map.items():
        if opstr == "copy":
            operations.append(
                saver.save(
                    model_key_vpp,
                    mg_name,
                    "copy",
                    st_loader_pre.load(mg_name, dtype_trans),
                )
            )
        elif opstr.startswith("dense_"):
            operations.append(
                saver.save(
                    model_key_vpp,
                    mg_name,
                    "tp",
                    SplitTpTpe(st_loader_pre.load(mg_name, dtype_trans), opstr),
                )
            )
        else:
            assert False

    actual_load_prefix.update(post_strategy_map.keys())
    for mg_name, (opstr, dtype_trans) in post_strategy_map.items():
        if opstr == "copy":
            operations.append(
                saver.save(
                    model_key_vpp,
                    mg_name,
                    "copy",
                    st_loader_post.load(mg_name, dtype_trans),
                )
            )
        elif opstr.startswith("dense_"):
            operations.append(
                saver.save(
                    model_key_vpp,
                    mg_name,
                    "tp",
                    SplitTpTpe(st_loader_post.load(mg_name, dtype_trans), opstr),
                )
            )
        else:
            assert False

    actual_load_prefix.update(
        (f"decoder.layers.{layer_idx}.{k}" for k in layer_strategy_map.keys())
    )
    loader_decoder = st_loader.sub_loader(f"decoder.layers.{layer_idx}.")
    saver_decoder = saver.sub_saver(f"decoder.layers.{local_layer}.")
    for mg_name, (opstr, dtype_trans) in layer_strategy_map.items():
        if opstr == "copy":
            operations.append(
                saver_decoder.save(
                    model_key_vpp,
                    mg_name,
                    "copy",
                    loader_decoder.load(mg_name, dtype_trans),
                )
            )
        elif opstr.startswith("rename_copy:"):
            real_name = opstr[len("rename_copy:") :]
            operations.append(
                saver_decoder.save(
                    model_key_vpp,
                    real_name,
                    "copy",
                    loader_decoder.load(mg_name, dtype_trans),
                )
            )
        elif opstr.startswith("dense_"):
            operations.append(
                saver_decoder.save(
                    model_key_vpp,
                    mg_name,
                    "tp",
                    SplitTpTpe(loader_decoder.load(mg_name, dtype_trans), opstr),
                )
            )
        elif opstr.startswith("qkv_"):
            operations.append(
                saver_decoder.save(
                    model_key_vpp,
                    mg_name,
                    "tp",
                    SplitTpTpe(loader_decoder.load(mg_name, dtype_trans), "dense_fc1"),
                )
            )
        else:
            assert False

    for globa_expert_index in range(num_experts):
        ep_rank = globa_expert_index // num_local_experts
        local_expert_index = globa_expert_index % num_local_experts

        actual_load_prefix.update(
            (
                f"decoder.layers.{layer_idx}.mlp.experts.local_experts.{globa_expert_index}.{k}"
                for k in expert_strategy_map.keys()
            )
        )
        loader_expert = loader_decoder.sub_loader(
            f"mlp.experts.local_experts.{globa_expert_index}."
        )
        saver_expert = saver_decoder.sub_saver(
            f"mlp.experts.local_experts.{local_expert_index}."
        )
        for mg_name, (opstr, dtype_trans) in expert_strategy_map.items():
            if opstr.startswith("moe_"):
                operations.append(
                    saver_expert.save(
                        model_key_vpp,
                        mg_name,
                        ep_rank,
                        SplitTpTpe(loader_expert.load(mg_name, dtype_trans), opstr),
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

    if convert_config.grouped_gemm == "te":
        for model_key in saver.full_checkpoint:
            state_dict = saver.full_checkpoint[model_key]["model"]
            moe_seq_to_te_group(state_dict)
    elif convert_config.grouped_gemm == "legacy":
        for model_key in saver.full_checkpoint:
            state_dict = saver.full_checkpoint[model_key]["model"]
            moe_seq_to_group(state_dict, num_local_experts, glu=True)
    elif convert_config.grouped_gemm is not None:
        assert False, (
            f"now megatron grouped_gemm {convert_config.grouped_gemm} not supported, please use te_grouped_gemm"
        )

    gc.collect()
    if Operation.global_device != "cpu":
        torch.cuda.empty_cache()
