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

# convert mg model to middle_file
# middle_file:
#   params not in decoder: io.safetensors
#   params in decoder layer x: x.safetensors
#   use megatron name and style (fuse glu fc1)

import gc
import os
import time
from argparse import ArgumentParser
from concurrent.futures import Future, ProcessPoolExecutor

import safetensors.torch
import torch
import yaml
from tqdm import tqdm
from utils.fp8_utils import dict_push
from utils.mg_loader import MGLoaderGroupLazy
from utils.mp_utils import get_device_initializer, single_thread_init
from utils.tensor_operations import MergeTpTpe, Operation

torch.set_num_threads(32)


def find_only_directory(path):
    """
    Finds the single directory directly under the given path.

    Args:
        path (str): The path to search within.

    Returns:
        str or None: The full path of the single directory found, or None if
                     zero or multiple directories are present.
    """
    if not os.path.isdir(path):
        print(f"Error: '{path}' is not a valid directory.")
        return None

    subdirectories = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            subdirectories.append(item_path)

    if len(subdirectories) == 1:
        return subdirectories[0]

    return None


def get_args():
    def strtobool(x: str):
        if x.lower() in ["true", "1", "yes", "y"]:
            return True
        if x.lower() in ["false", "0", "no", "n"]:
            return False
        raise ValueError()

    parser = ArgumentParser()
    parser.add_argument(
        "--load-path", type=str, required=True, help="Path to megatron model"
    )
    parser.add_argument(
        "--save-path", type=str, required=True, help="Path to middle file"
    )
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--iteration", type=int, default=-1)

    # attn
    parser.add_argument("--attn-type", type=str, choices=["gqa", "mla"], default=None)
    parser.add_argument("--use-q-lora", type=strtobool, default=None)
    parser.add_argument("--use-qkv-bias", type=strtobool, default=None)
    parser.add_argument("--use-qk-norm", type=strtobool, default=None)

    # mlp
    parser.add_argument("--mlp-type", type=str, choices=["dense", "moe"], default=None)
    parser.add_argument("--num-experts", type=int, default=None)
    parser.add_argument("--use-shared-experts", type=strtobool, default=None)
    parser.add_argument("--use-expert-bias", type=strtobool, default=None)
    parser.add_argument("--use-shared-experts-gate", type=strtobool, default=None)
    parser.add_argument("--first-dense", type=int, default=None)

    # other network structure
    parser.add_argument("--tie-word-embeddings", type=strtobool, default=None)

    # distributed
    parser.add_argument("--use-gpu-num", type=int, default=None)
    parser.add_argument("--use-gpu-index", type=int, nargs="*", default=None)
    parser.add_argument("--process-num", type=int, default=None)

    # precision
    parser.add_argument(
        "--router-trans",
        type=str,
        choices=["auto", "bf16_bf16", "bf16_fp32", "fp32_bf16", "fp32_fp32"],
        default="auto",
    )
    parser.add_argument(
        "--linear-trans",
        type=str,
        choices=["auto", "fp8_fp8", "fp8_bf16", "bf16_fp8", "bf16_bf16"],
        default="auto",
    )
    parser.add_argument(
        "--layernorm-trans",
        type=str,
        choices=["auto", "fp8_fp8", "fp8_bf16", "bf16_fp8", "bf16_bf16"],
        default="auto",
    )

    # models
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model-type", type=str, default=None)

    # fusion
    parser.add_argument("--te-ln-linear-qkv", type=strtobool, default=False)
    parser.add_argument("--te-ln-linear-mlp_fc1", type=strtobool, default=False)
    parser.add_argument("--te-extra-state-check-none", type=strtobool, default=False)

    # parallism
    parser.add_argument("--tp-size", type=int, default=1, required=True)
    parser.add_argument("--pp-size", type=int, default=1, required=True)
    parser.add_argument("--tpe-size", type=int, default=1, required=False)
    parser.add_argument("--ep-size", type=int, default=1, required=False)
    parser.add_argument(
        "--schedular", type=str, choices=["1f1b", "vpp", "dualpipev"], default="1f1b"
    )
    parser.add_argument("--pp-stages", type=int, nargs="*", default=None)

    # log specific
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="disable tqdm progress bar output if True",
    )

    args = parser.parse_args()
    args.load_path = find_only_directory(args.load_path)

    if args.use_gpu_num is None:
        args.use_gpu_num = torch.cuda.device_count()
        print(f"use-gpu-num is None, so default use {args.use_gpu_num} gpus")
    if args.use_gpu_index is not None:
        assert len(args.use_gpu_index) == args.use_gpu_num

    script_path = os.path.dirname(os.path.abspath(__file__))
    with open(f"{script_path}/default_args.yaml") as stream:
        defaults_values = yaml.safe_load(stream)
    model_defaults = defaults_values["explict_model"]
    model_type_defaults = defaults_values["model_type"]

    if args.model is not None:
        assert args.model in model_defaults, "model not in default_args.yaml!"
        for k, v in model_defaults[args.model].items():
            if hasattr(args, k) and getattr(args, k) is None:
                setattr(args, k, v)

    if args.model_type is not None:
        assert args.model_type in model_type_defaults, (
            "model_type not in default_args.yaml!"
        )
        for k, v in model_type_defaults[args.model_type].items():
            if hasattr(args, k) and getattr(args, k) is None:
                setattr(args, k, v)

    assert args.attn_type is not None
    if args.attn_type == "mla":
        assert args.use_q_lora is not None
        assert args.num_query_groups is None
        if args.use_qk_norm is None:
            args.use_qk_norm = True
    elif args.attn_type == "gqa":
        assert args.use_q_lora is None
        if args.use_qkv_bias is None:
            args.use_qkv_bias = False
        if args.use_qk_norm is None:
            args.use_qk_norm = False

    assert args.mlp_type is not None
    if args.mlp_type == "moe":
        if args.use_shared_experts is None:
            args.use_shared_experts = False
        if args.use_shared_experts:
            if args.use_expert_bias is None:
                args.use_expert_bias = False
            if args.use_shared_experts_gate is None:
                args.use_shared_experts_gate = False
        if args.first_dense is None:
            args.first_dense = 0
        assert args.num_experts is not None
    else:
        assert args.num_experts in (0, None)
        args.num_experts = 0

    assert args.num_layers is not None

    return args


def get_extra_state_check_none(extra_state):
    if get_extra_state_check_none.extra_state_none is None:
        import pickle

        state_serialized = bytearray(pickle.dumps(None))
        state_serialized = torch.frombuffer(state_serialized, dtype=torch.uint8)
        get_extra_state_check_none.extra_state_none = state_serialized
    return extra_state is None or torch.equal(
        extra_state, get_extra_state_check_none.extra_state_none
    )


get_extra_state_check_none.extra_state_none = None


def get_megatron_iteration(args):
    if args.iteration != -1:
        iteration = args.iteration
    else:
        if args.schedular == "1f1b":
            model_key_vpp = "model"
        else:
            model_key_vpp = "model0"
        mg_loader = MGLoaderGroupLazy.from_path(args.load_path, 0)
        if "iteration" in mg_loader.keys(model_key_vpp):
            iteration = str(mg_loader.get_keys_value("iteration"))
        else:
            iteration = "release"

    with open(f"{args.save_path}/latest_checkpointed_iteration.txt", "w") as f:
        f.write(iteration)

    if iteration == "release":
        return -1
    else:
        return int(iteration)


def get_hetero_pp_rank(args, num_layers, layer_idx):
    if args.schedular == "1f1b":
        if args.pp_stages is not None:
            pp_stages = args.pp_stages
            assert len(pp_stages) == args.pp_size
            assert sum(pp_stages) == num_layers
            for pp_rank in range(1000):
                local_num_layers = pp_stages[pp_rank]
                layer_offset = sum(([0] + pp_stages)[: pp_rank + 1])
                if layer_offset <= layer_idx < layer_offset + local_num_layers:
                    return pp_rank, layer_idx - layer_offset, local_num_layers
        else:
            assert num_layers % args.pp_size == 0, (
                "cannot split layers to pp. please use --pp-satges"
            )
            local_num_layers = int(num_layers // args.pp_size)
            if layer_idx == 0:
                return 0, 0, 0
            pp_rank = layer_idx // local_num_layers
            local_layer = layer_idx % local_num_layers
            return int(pp_rank), int(local_layer), local_num_layers
    elif args.schedular == "dualpipev":
        if args.pp_stages is not None:
            ppvpp_stages = args.pp_stages
            assert len(ppvpp_stages) == args.pp_size * 2
            assert sum(ppvpp_stages) == num_layers
            # reverse vpp rank 1
            ppvpp_stages = (
                ppvpp_stages[: args.pp_size] + ppvpp_stages[: args.pp_size - 1 : -1]
            )
            for ppvpp_rank in range(1000):
                local_num_layers = ppvpp_stages[ppvpp_rank]
                layer_offset = sum(([0] + ppvpp_stages)[: ppvpp_rank + 1])
                if layer_offset <= layer_idx < layer_offset + local_num_layers:
                    return ppvpp_rank, layer_idx - layer_offset, local_num_layers
        else:
            assert num_layers % (args.pp_size * 2) == 0, (
                "cannot split layers to vpp. please use --pp-satges"
            )
            local_num_layers = int(num_layers // args.pp_size // 2)
            if layer_idx == 0:
                return 0, 0, 0
            ppvpp_rank = layer_idx // local_num_layers
            local_layer = layer_idx % local_num_layers
            return int(ppvpp_rank), int(local_layer), local_num_layers
    elif args.schedular == "vpp":
        raise NotImplementedError()
    else:
        raise ValueError()


class Save(Operation):
    def __init__(self, buffer_dict: dict, name: str, src: Operation):
        super().__init__()
        self.buffer_dict = buffer_dict
        self.name = name
        self.src = src

    def execute(self):
        tensor: torch.Tensor = self.src.execute()
        tensor = tensor.to(device="cpu")
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
        self.buffer_dict.clear()


def convert_layer_loadsave(
    args, pp_rank, model_key_vpp, save_layer_path, layer_idx, local_layer
):
    mg_loader = MGLoaderGroupLazy.from_path(args.load_path, pp_rank)
    saver = DictSaver()
    convert_layer(args, mg_loader, saver, model_key_vpp, layer_idx, local_layer)
    saver.dump_file(save_layer_path)


def filter_keys_output_embedding(k):
    return k.startswith("embedding.")


def filter_keys_output_local_layer(k, local_layer):
    return k.startswith(f"decoder.layers.{local_layer}.")


def filter_keys_output(k):
    return (
        k.startswith("model.")
        and not k.startswith("model.layers.")
        and not k.startswith("model.norm.")
    ) or (
        not k.startswith("lm_head.")
        and not k.startswith("embedding.")
        and not k.startswith("decoder.layers.")
    )


def convert_layer(
    args,
    mg_loader: MGLoaderGroupLazy,
    saver: DictSaver,
    model_key_vpp,
    layer_idx,
    local_layer,
):
    num_layers = args.num_layers
    moe = args.mlp_type == "moe" and layer_idx >= args.first_dense
    num_local_experts = args.num_experts // args.ep_size

    linear_trans = args.linear_trans
    layernorm_trans = args.layernorm_trans
    router_trans = args.router_trans

    should_load_prefix: set[str] = set()
    actual_load_prefix: set[str] = set()
    model_strategy_map = {}
    layer_strategy_map = {}
    expert_strategy_map = {}

    mg_all_keys = mg_loader.keys(model_key_vpp)
    if layer_idx == num_layers:
        assert local_layer is None
        should_load_prefix.update(
            (
                k
                for k in mg_loader.keys(model_key_vpp)
                if filter_keys_output_embedding(k)
            )
        )
        model_strategy_map.update(
            {
                "embedding.word_embeddings.weight": ("dense_fc1", linear_trans),
            }
        )
    elif layer_idx == num_layers + 1:
        assert local_layer is None
        should_load_prefix.update(
            (k for k in mg_loader.keys(model_key_vpp) if filter_keys_output(k))
        )
        model_strategy_map.update(
            {
                "decoder.final_layernorm.weight": ("copy", layernorm_trans),
            }
        )
        if not args.tie_word_embeddings:
            model_strategy_map.update(
                {
                    "output_layer.weight": ("dense_fc1", linear_trans),
                }
            )
    else:
        assert local_layer is not None
        should_load_prefix.update(
            (
                k
                for k in mg_loader.keys(model_key_vpp)
                if filter_keys_output_local_layer(k, local_layer)
            )
        )
        if args.attn_type == "mla":
            layer_strategy_map.update(
                {
                    "input_layernorm.weight": ("copy", layernorm_trans),
                    "self_attention.linear_kv_a_proj.weight": ("copy", linear_trans),
                    "self_attention.linear_kv_b_proj.weight": (
                        "dense_fc1",
                        linear_trans,
                    ),
                    "self_attention.linear_proj.weight": ("dense_fc2", linear_trans),
                }
            )
            if args.use_q_lora:
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
                        "self_attention.linear_q_proj.weight": (
                            "dense_fc1",
                            linear_trans,
                        ),
                    }
                )
            if args.use_qk_norm:
                if args.use_q_lora:
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
                        "self_attention.kv_a_layernorm.weight": (
                            "copy",
                            layernorm_trans,
                        ),
                    }
                )
        elif args.attn_type == "gqa":
            if args.te_ln_linear_qkv:
                layer_strategy_map.update(
                    {
                        "self_attention.linear_qkv.layer_norm_weight": (
                            "rename_copy:input_layernorm.weight",
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
            if args.use_qkv_bias:
                layer_strategy_map.update(
                    {
                        "self_attention.linear_qkv.bias": ("qkv_b", linear_trans),
                    }
                )
            if args.use_qk_norm:
                layer_strategy_map.update(
                    {
                        "self_attention.q_layernorm.weight": ("copy", layernorm_trans),
                        "self_attention.k_layernorm.weight": ("copy", layernorm_trans),
                    }
                )
        else:
            raise NotImplementedError(f"attn-type not support for {args.attn_type}")

        if not moe:
            if args.te_ln_linear_mlp_fc1:
                layer_strategy_map.update(
                    {
                        "mlp.linear_fc1.layer_norm_weight": (
                            "rename_copy:pre_mlp_layernorm.weight",
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
            if args.use_expert_bias:
                layer_strategy_map.update(
                    {
                        "mlp.router.expert_bias": ("copy", router_trans),
                    }
                )
            if args.use_shared_experts:
                layer_strategy_map.update(
                    {
                        "mlp.shared_expert.linear_fc1.weight": (
                            "dense_fc1_glu",
                            linear_trans,
                        ),
                        "mlp.shared_expert.linear_fc2.weight": (
                            "dense_fc2",
                            linear_trans,
                        ),
                    }
                )
                if args.use_shared_experts_gate:
                    layer_strategy_map.update(
                        {
                            "mlp.shared_expert_gate.weight": ("copy", router_trans),
                        }
                    )

    operations: list[Operation] = []

    actual_load_prefix.update(model_strategy_map.keys())
    for mg_name, (opstr, dtype_trans) in model_strategy_map.items():
        if opstr == "copy" or opstr.startswith("dense_"):
            operations.append(
                saver.save(
                    mg_name,
                    MergeTpTpe(
                        mg_loader.load(model_key_vpp, mg_name, dtype_trans), opstr
                    ),
                )
            )
        else:
            assert False

    actual_load_prefix.update(
        (f"decoder.layers.{local_layer}.{k}" for k in layer_strategy_map.keys())
    )
    loader_decoder = mg_loader.sub_loader(f"decoder.layers.{local_layer}.")
    saver_decoder = saver.sub_saver(f"decoder.layers.{layer_idx}.")
    for mg_name, (opstr, dtype_trans) in layer_strategy_map.items():
        if opstr.startswith("rename_copy:"):
            mg_name_new = opstr.split(":")[1]
            operations.append(
                saver_decoder.save(
                    mg_name_new,
                    MergeTpTpe(
                        loader_decoder.load(model_key_vpp, mg_name, dtype_trans), "copy"
                    ),
                )
            )
        elif opstr == "copy" or opstr.startswith("dense_"):
            operations.append(
                saver_decoder.save(
                    mg_name,
                    MergeTpTpe(
                        loader_decoder.load(model_key_vpp, mg_name, dtype_trans), opstr
                    ),
                )
            )
        elif opstr.startswith("qkv_"):
            operations.append(
                saver_decoder.save(
                    mg_name,
                    MergeTpTpe(
                        loader_decoder.load(model_key_vpp, mg_name, dtype_trans),
                        "dense_fc1",
                    ),
                )
            )
        else:
            assert False

    for globa_expert_index in range(args.num_experts):
        ep_rank = globa_expert_index // num_local_experts
        local_expert_index = globa_expert_index % num_local_experts

        actual_load_prefix.update(
            (
                f"decoder.layers.{local_layer}.mlp.experts.local_experts.{local_expert_index}.{k}"
                for k in expert_strategy_map.keys()
            )
        )
        loader_expert = loader_decoder.sub_loader(
            f"mlp.experts.local_experts.{local_expert_index}.", need_ep_rank=ep_rank
        )
        saver_expert = saver_decoder.sub_saver(
            f"mlp.experts.local_experts.{globa_expert_index}."
        )
        for mg_name, (opstr, dtype_trans) in expert_strategy_map.items():
            if opstr.startswith("moe_"):
                operations.append(
                    saver_expert.save(
                        mg_name,
                        MergeTpTpe(
                            loader_expert.load(model_key_vpp, mg_name, dtype_trans),
                            opstr,
                        ),
                    )
                )
            else:
                assert False

    if args.te_extra_state_check_none:
        should_load_prefix_new = set()
        for key in should_load_prefix:
            if not key.endswith("._extra_state"):
                should_load_prefix_new.add(key)
            else:
                if key not in mg_all_keys:
                    loaded = [
                        i.execute()
                        for i in mg_loader.load(model_key_vpp, key, "not_tensor")
                    ]
                    assert all(get_extra_state_check_none(i) for i in loaded), (
                        f"extra_state is not None for {key}"
                    )
        should_load_prefix = should_load_prefix_new

    should_load_but = should_load_prefix.difference(actual_load_prefix)
    actual_load_but = actual_load_prefix.difference(should_load_prefix)
    if should_load_but or actual_load_but:
        raise RuntimeError(
            f"Layer {layer_idx}: should_load_but: {should_load_but}; actual_load_but: {actual_load_but}"
        )

    if layer_idx == num_layers:
        desc = "Before decoder"
    elif layer_idx == num_layers + 1:
        desc = "After decoder"
    else:
        desc = f"Layer {layer_idx}"
    if args.disable_tqdm:
        for op in operations:
            op.execute()
    else:
        for op in tqdm(operations, desc=desc, position=Operation.local_idx + 1):
            op.execute()

    gc.collect()
    if Operation.global_device != "cpu":
        torch.cuda.empty_cache()


def main():
    args = get_args()
    time_start = time.time()

    Operation.global_tp = args.tp_size
    Operation.global_tpe = args.tpe_size
    Operation.global_ep = args.ep_size
    Operation.global_pp = args.pp_size

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    args.iteration = get_megatron_iteration(args)

    pp_rank_to_layers = [[] for _ in range(Operation.global_pp)]
    for layer_idx in range(int(args.num_layers)):
        ppvpp_rank, local_layer, local_num_layer = get_hetero_pp_rank(
            args, args.num_layers, layer_idx
        )
        assert local_layer is not None
        if args.schedular == "1f1b":
            pp_rank = ppvpp_rank
            model_key_vpp = "model"
        elif args.schedular == "vpp":
            if ppvpp_rank < Operation.global_pp:
                pp_rank = ppvpp_rank
                model_key_vpp = "model0"
            else:
                pp_rank = ppvpp_rank - Operation.global_pp
                model_key_vpp = "model1"
        elif args.schedular == "dualpipev":
            if ppvpp_rank < Operation.global_pp:
                pp_rank = ppvpp_rank
                model_key_vpp = "model0"
            else:
                pp_rank = Operation.global_pp * 2 - 1 - ppvpp_rank
                model_key_vpp = "model1"
        else:
            assert False
        pp_rank_to_layers[pp_rank].append((model_key_vpp, layer_idx, local_layer))

    if args.schedular == "1f1b":
        pre_layer = ("model", args.num_layers, None)
        post_layer = ("model", args.num_layers + 1, None)
        pp_rank_to_layers[0] = [pre_layer, *pp_rank_to_layers[0]]
        pp_rank_to_layers[-1].append(post_layer)
    elif args.schedular == "vpp":
        pre_layer = ("model0", args.num_layers, None)
        post_layer = ("model1", args.num_layers + 1, None)
        pp_rank_to_layers[0] = [pre_layer, *pp_rank_to_layers[0]]
        pp_rank_to_layers[-1].append(post_layer)
    elif args.schedular == "dualpipev":
        pre_layer = ("model0", args.num_layers, None)
        post_layer = ("model1", args.num_layers + 1, None)
        pp_rank_to_layers[0] = [pre_layer, *pp_rank_to_layers[0], post_layer]
    else:
        assert False

    convert_layers = []
    for pp_rank in range(Operation.global_pp):
        if not args.disable_tqdm:
            tqdm.write(f"converting pp_rank {pp_rank}")

        for model_key_vpp, layer_idx, local_layer in pp_rank_to_layers[pp_rank]:
            run_covert = True

            assert args.save_path is not None
            if layer_idx == args.num_layers:
                assert local_layer is None
                save_layer_path = f"{args.save_path}/pre.safetensors"
            elif layer_idx == args.num_layers + 1:
                assert local_layer is None
                save_layer_path = f"{args.save_path}/post.safetensors"
            else:
                assert local_layer is not None
                save_layer_path = f"{args.save_path}/{layer_idx}.safetensors"
            if os.path.exists(save_layer_path):
                if os.path.exists(f"{save_layer_path}.done"):
                    run_covert = False
                    continue
                else:
                    os.remove(save_layer_path)

            if run_covert:
                convert_layers.append(
                    (pp_rank, model_key_vpp, save_layer_path, layer_idx, local_layer)
                )

    if args.process_num > 1:
        with ProcessPoolExecutor(
            args.process_num, initializer=get_device_initializer(args)
        ) as mp_exec:
            handles: list[Future] = []
            for (
                pp_rank,
                model_key_vpp,
                save_layer_path,
                layer_idx,
                local_layer,
            ) in convert_layers:
                handles.append(
                    (
                        layer_idx,
                        mp_exec.submit(
                            convert_layer_loadsave,
                            args,
                            pp_rank,
                            model_key_vpp,
                            save_layer_path,
                            layer_idx,
                            local_layer,
                        ),
                    )
                )
            for layer_idx, t in handles:
                exp = t.exception()
                if exp is not None:
                    mp_exec.shutdown(True, cancel_futures=True)
                    raise exp
                t.result()
                if not args.disable_tqdm:
                    tqdm.write(f"layer converted {layer_idx}")
    else:
        # run in one thread
        single_thread_init(args)
        for (
            pp_rank,
            model_key_vpp,
            save_layer_path,
            layer_idx,
            local_layer,
        ) in convert_layers:
            convert_layer_loadsave(
                args, pp_rank, model_key_vpp, save_layer_path, layer_idx, local_layer
            )
            if not args.disable_tqdm:
                tqdm.write(f"layer converted {layer_idx}")
    print(
        f"convert megatron format to middle file finished! time_elapse: {time.time() - time_start:0.3f}s"
    )


if __name__ == "__main__":
    main()
