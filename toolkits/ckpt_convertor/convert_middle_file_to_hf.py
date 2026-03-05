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
import json
import os
import time
from argparse import ArgumentParser
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Union

import safetensors.torch
import torch
import yaml
from tqdm import tqdm
from utils.fp8_utils import dict_push
from utils.mp_utils import get_device_initializer, single_thread_init
from utils.safetensors_loader import STLoaderLazy
from utils.tensor_operations import Operation, SplitGlu, SplitQKV

torch.set_num_threads(32)


def get_args():
    def strtobool(x: str):
        if x.lower() in ["true", "1", "yes", "y"]:
            return True
        if x.lower() in ["false", "0", "no", "n"]:
            return False
        raise ValueError()

    parser = ArgumentParser()
    parser.add_argument(
        "--load-path", type=str, required=True, help="Path to middle file"
    )
    parser.add_argument(
        "--save-path", type=str, required=True, help="Path to huggingface model"
    )
    parser.add_argument("--num-layers", type=int, default=None)

    # attn
    parser.add_argument("--attn-type", type=str, choices=["gqa", "mla"], default=None)
    parser.add_argument("--use-q-lora", type=strtobool, default=None)
    parser.add_argument("--use-qkv-bias", type=strtobool, default=None)
    parser.add_argument("--use-qk-norm", type=strtobool, default=None)
    parser.add_argument("--num-attention-heads", type=int, default=None)
    parser.add_argument("--num-query-groups", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=None)

    # mlp
    parser.add_argument("--mlp-type", type=str, choices=["dense", "moe"], default=None)
    parser.add_argument("--num-experts", type=int, default=None)
    parser.add_argument("--use-shared-experts", type=strtobool, default=None)
    parser.add_argument(
        "--hf-shared-experts-prefix",
        choices=["mlp.shared_expert", "mlp.shared_experts"],
        default=None,
    )
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

    # log specific
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        default=False,
        help="disable tqdm progress bar output if True",
    )

    args = parser.parse_args()

    if args.use_gpu_num is None:
        args.use_gpu_num = torch.cuda.device_count()
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
        if args.use_qk_norm is None:
            args.use_qk_norm = True
        assert args.use_q_lora is not None
        assert args.num_attention_heads is None
        assert args.num_query_groups is None
    elif args.attn_type == "gqa":
        assert args.use_q_lora is None
        if args.use_qkv_bias is None:
            args.use_qkv_bias = False
        if args.use_qk_norm is None:
            args.use_qk_norm = False
        assert args.num_attention_heads is not None
        assert args.num_query_groups is not None

    assert args.mlp_type is not None
    if args.mlp_type == "moe":
        if args.use_shared_experts is None:
            args.use_shared_experts = False
        if args.use_shared_experts:
            if args.use_expert_bias is None:
                args.use_expert_bias = False
            if args.use_shared_experts_gate is None:
                args.use_shared_experts_gate = False
            assert args.hf_shared_experts_prefix is not None
        if args.first_dense is None:
            args.first_dense = 0
        assert args.num_experts is not None
    else:
        assert args.num_experts in (0, None)
        args.num_experts = 0

    assert args.num_layers is not None

    return args


class Save(Operation):
    def __init__(self, buffer_dict: dict, names: Union[str, list[str]], src: Operation):
        super().__init__()
        self.buffer_dict = buffer_dict
        self.names = names
        self.src = src

    def execute(self):
        tensors = self.src.execute()
        if isinstance(self.names, str):
            dict_push(self.buffer_dict, self.names, tensors)
        else:
            assert isinstance(self.names, (tuple, list)) and isinstance(
                tensors, (tuple, list)
            )
            assert len(self.names) == len(tensors)
            for n, t in zip(self.names, tensors):
                dict_push(self.buffer_dict, n, t)


class HFSTSaver:
    def __init__(self, buffer_dict=None, prefix=""):
        super().__init__()
        if buffer_dict is None:
            buffer_dict = {}
        self.buffer_dict = buffer_dict
        self.prefix = prefix

    def sub_saver(self, suffix):
        return HFSTSaver(self.buffer_dict, f"{self.prefix}{suffix}")

    def save(self, names: Union[str, list[str]], src: Operation):
        if isinstance(names, str):
            full_names = f"{self.prefix}{names}"
        else:
            full_names = [f"{self.prefix}{i}" for i in names]
        return Save(self.buffer_dict, full_names, src)

    def dump_file(self, save_path, save_layer_filename):
        save_layer_path = f"{save_path}/{save_layer_filename}.safetensors"
        safetensors.torch.save_file(self.buffer_dict, save_layer_path)
        with open(f"{save_path}/{save_layer_filename}.index_map", "w") as f:
            index_map = dict.fromkeys(
                self.buffer_dict, f"{save_layer_filename}.safetensors"
            )
            json.dump(index_map, f, indent=4)
        with open(f"{save_layer_path}.done", "w") as f:
            f.write("done")
        self.buffer_dict.clear()


def convert_layer_loadsave(args, layer_idx, save_layer_filename):
    if layer_idx == args.num_layers:
        load_layer_path = f"{args.load_path}/pre.safetensors"
    elif layer_idx == args.num_layers + 1:
        load_layer_path = f"{args.load_path}/post.safetensors"
    else:
        load_layer_path = f"{args.load_path}/{layer_idx}.safetensors"
    mfst_loader = STLoaderLazy.from_file([load_layer_path])
    saver = HFSTSaver()
    convert_layer(args, mfst_loader, saver, layer_idx)
    saver.dump_file(args.save_path, save_layer_filename)


def convert_layer(args, mfst_loader: STLoaderLazy, saver: HFSTSaver, layer_idx):
    num_layers = args.num_layers
    moe = args.mlp_type == "moe" and layer_idx >= args.first_dense

    linear_trans = args.linear_trans
    layernorm_trans = args.layernorm_trans
    router_trans = args.router_trans

    should_load_prefix: set[str] = set()
    actual_load_prefix: set[str] = set()
    model_strategy_map = {}
    layer_strategy_map = {}
    expert_strategy_map = {}

    if layer_idx == num_layers:
        should_load_prefix.update(
            (k for k in mfst_loader.keys() if k.startswith("embedding."))
        )
        model_strategy_map.update(
            {
                "model.embed_tokens.weight": (
                    "copy",
                    linear_trans,
                    "embedding.word_embeddings.weight",
                ),
            }
        )
    elif layer_idx == num_layers + 1:
        should_load_prefix.update(
            (
                k
                for k in mfst_loader.keys()
                if (
                    k.startswith("model.")
                    and not k.startswith("model.layers.")
                    and not k.startswith("model.norm.")
                )
                or not k.startswith("lm_head.")
            )
        )
        model_strategy_map.update(
            {
                "model.norm.weight": (
                    "copy",
                    layernorm_trans,
                    "decoder.final_layernorm.weight",
                ),
            }
        )
        if not args.tie_word_embeddings:
            model_strategy_map.update(
                {
                    "lm_head.weight": ("copy", linear_trans, "output_layer.weight"),
                }
            )
    else:
        should_load_prefix.update(
            (
                k
                for k in mfst_loader.keys()
                if k.startswith(f"decoder.layers.{layer_idx}.")
            )
        )
        if args.attn_type == "mla":
            layer_strategy_map.update(
                {
                    "input_layernorm.weight": (
                        "copy",
                        layernorm_trans,
                        "input_layernorm.weight",
                    ),
                    "self_attn.kv_a_proj_with_mqa.weight": (
                        "copy",
                        linear_trans,
                        "self_attention.linear_kv_a_proj.weight",
                    ),
                    "self_attn.kv_b_proj.weight": (
                        "copy",
                        linear_trans,
                        "self_attention.linear_kv_b_proj.weight",
                    ),
                    "self_attn.o_proj.weight": (
                        "copy",
                        linear_trans,
                        "self_attention.linear_proj.weight",
                    ),
                }
            )
            if args.use_q_lora:
                layer_strategy_map.update(
                    {
                        "self_attn.q_a_layernorm.weight": (
                            "copy",
                            layernorm_trans,
                            "self_attention.q_a_layernorm.weight",
                        ),
                        "self_attn.q_a_proj.weight": (
                            "copy",
                            linear_trans,
                            "self_attention.linear_q_a_proj.weight",
                        ),
                        "self_attn.q_b_proj.weight": (
                            "copy",
                            linear_trans,
                            "self_attention.linear_q_b_proj.weight",
                        ),
                    }
                )
            else:
                layer_strategy_map.update(
                    {
                        "self_attn.q_proj.weight": (
                            "copy",
                            linear_trans,
                            "self_attention.linear_q_proj.weight",
                        ),
                    }
                )
            if args.use_qk_norm:
                if args.use_q_lora:
                    layer_strategy_map.update(
                        {
                            "self_attn.q_a_layernorm.weight": (
                                "copy",
                                layernorm_trans,
                                "self_attention.q_a_layernorm.weight",
                            ),
                        }
                    )
                layer_strategy_map.update(
                    {
                        "self_attn.kv_a_layernorm.weight": (
                            "copy",
                            layernorm_trans,
                            "self_attention.kv_a_layernorm.weight",
                        ),
                    }
                )
        elif args.attn_type == "gqa":
            layer_strategy_map.update(
                {
                    "input_layernorm.weight": (
                        "copy",
                        layernorm_trans,
                        "input_layernorm.weight",
                    ),
                    (
                        "self_attn.q_proj.weight",
                        "self_attn.k_proj.weight",
                        "self_attn.v_proj.weight",
                    ): ("qkvw", linear_trans, "self_attention.linear_qkv.weight"),
                    "self_attn.o_proj.weight": (
                        "copy",
                        linear_trans,
                        "self_attention.linear_proj.weight",
                    ),
                }
            )
            if args.use_qkv_bias:
                layer_strategy_map.update(
                    {
                        (
                            "self_attn.q_proj.bias",
                            "self_attn.k_proj.bias",
                            "self_attn.v_proj.bias",
                        ): ("qkvb", linear_trans, "self_attention.linear_qkv.bias"),
                    }
                )
            if args.use_qk_norm:
                layer_strategy_map.update(
                    {
                        "self_attn.q_norm.weight": (
                            "copy",
                            layernorm_trans,
                            "self_attention.q_layernorm.weight",
                        ),
                        "self_attn.k_norm.weight": (
                            "copy",
                            layernorm_trans,
                            "self_attention.k_layernorm.weight",
                        ),
                    }
                )
        else:
            raise NotImplementedError(f"attn-type not support for {args.attn_type}")

        if not moe:
            layer_strategy_map.update(
                {
                    "post_attention_layernorm.weight": (
                        "copy",
                        linear_trans,
                        "pre_mlp_layernorm.weight",
                    ),
                    ("mlp.gate_proj.weight", "mlp.up_proj.weight"): (
                        "glu",
                        linear_trans,
                        "mlp.linear_fc1.weight",
                    ),
                    "mlp.down_proj.weight": (
                        "copy",
                        linear_trans,
                        "mlp.linear_fc2.weight",
                    ),
                }
            )
        else:
            layer_strategy_map.update(
                {
                    "post_attention_layernorm.weight": (
                        "copy",
                        layernorm_trans,
                        "pre_mlp_layernorm.weight",
                    ),
                    "mlp.gate.weight": ("copy", router_trans, "mlp.router.weight"),
                }
            )
            expert_strategy_map.update(
                {
                    ("gate_proj.weight", "up_proj.weight"): (
                        "glu",
                        linear_trans,
                        "linear_fc1.weight",
                    ),
                    "down_proj.weight": ("copy", linear_trans, "linear_fc2.weight"),
                }
            )
            if args.use_expert_bias:
                layer_strategy_map.update(
                    {
                        "mlp.gate.e_score_correction_bias": (
                            "copy",
                            router_trans,
                            "mlp.router.expert_bias",
                        ),
                    }
                )
            if args.use_shared_experts:
                shared_prefix = args.hf_shared_experts_prefix
                layer_strategy_map.update(
                    {
                        (
                            f"{shared_prefix}.gate_proj.weight",
                            f"{shared_prefix}.up_proj.weight",
                        ): ("glu", linear_trans, "mlp.shared_expert.linear_fc1.weight"),
                        f"{shared_prefix}.down_proj.weight": (
                            "copy",
                            linear_trans,
                            "mlp.shared_expert.linear_fc2.weight",
                        ),
                    }
                )
                if args.use_shared_experts_gate:
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

    actual_load_prefix.update(
        (mg_name for mg_name in (i[2] for i in model_strategy_map.values()))
    )
    for hf_names, (opstr, dtype_trans, mg_name) in model_strategy_map.items():
        if opstr == "copy":
            operations.append(
                saver.save(hf_names, mfst_loader.load(mg_name, dtype_trans))
            )
        else:
            assert False

    actual_load_prefix.update(
        (
            f"decoder.layers.{layer_idx}.{mg_name}"
            for mg_name in (i[2] for i in layer_strategy_map.values())
        )
    )
    loader_decoder = mfst_loader.sub_loader(f"decoder.layers.{layer_idx}.")
    saver_decoder = saver.sub_saver(f"model.layers.{layer_idx}.")
    for hf_names, (opstr, dtype_trans, mg_name) in layer_strategy_map.items():
        if opstr == "copy":
            operations.append(
                saver_decoder.save(hf_names, loader_decoder.load(mg_name, dtype_trans))
            )
        elif opstr == "glu":
            operations.append(
                saver_decoder.save(
                    hf_names, SplitGlu(loader_decoder.load(mg_name, dtype_trans))
                )
            )
        elif opstr == "qkvw":
            num_query_groups, num_attention_heads, head_dim = (
                args.num_query_groups,
                args.num_attention_heads,
                args.head_dim,
            )
            operations.append(
                saver_decoder.save(
                    hf_names,
                    SplitQKV(
                        loader_decoder.load(mg_name, dtype_trans),
                        num_query_groups,
                        num_attention_heads,
                        head_dim,
                        "w",
                    ),
                )
            )
        elif opstr == "qkvb":
            num_query_groups, num_attention_heads, head_dim = (
                args.num_query_groups,
                args.num_attention_heads,
                args.head_dim,
            )
            operations.append(
                saver_decoder.save(
                    hf_names,
                    SplitQKV(
                        loader_decoder.load(mg_name, dtype_trans),
                        num_query_groups,
                        num_attention_heads,
                        head_dim,
                        "b",
                    ),
                )
            )
        else:
            assert False

    for expert_index in range(args.num_experts):
        actual_load_prefix.update(
            (
                f"decoder.layers.{layer_idx}.mlp.experts.local_experts.{expert_index}.{mg_name}"
                for mg_name in (i[2] for i in expert_strategy_map.values())
            )
        )
        loader_expert = loader_decoder.sub_loader(
            f"mlp.experts.local_experts.{expert_index}."
        )
        saver_expert = saver_decoder.sub_saver(f"mlp.experts.{expert_index}.")
        for hf_names, (opstr, dtype_trans, mg_name) in expert_strategy_map.items():
            if opstr == "copy":
                operations.append(
                    saver_expert.save(
                        hf_names, loader_expert.load(mg_name, dtype_trans)
                    )
                )
            elif opstr == "glu":
                operations.append(
                    saver_expert.save(
                        hf_names, SplitGlu(loader_expert.load(mg_name, dtype_trans))
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

    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    convert_layers = []
    for layer_idx in range(int(args.num_layers + 2)):
        run_covert = True
        # loader and saver per layer

        assert args.save_path is not None
        save_layer_filename = f"model-{layer_idx + 1:05d}-of-{args.num_layers + 2:05d}"
        save_layer_path = f"{args.save_path}/{save_layer_filename}.safetensors"
        if os.path.exists(save_layer_path):
            if os.path.exists(f"{save_layer_path}.done"):
                run_covert = False
            else:
                os.remove(f"{args.save_path}/{save_layer_filename}.index_map")
                os.remove(save_layer_path)

        if run_covert:
            convert_layers.append((layer_idx, save_layer_filename))

    if args.process_num > 1:
        with ProcessPoolExecutor(
            args.process_num, initializer=get_device_initializer(args)
        ) as mp_exec:
            handles: list[Future] = []
            for layer_idx, save_layer_filename in convert_layers:
                handles.append(
                    (
                        layer_idx,
                        mp_exec.submit(
                            convert_layer_loadsave, args, layer_idx, save_layer_filename
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
        for layer_idx, save_layer_filename in convert_layers:
            convert_layer_loadsave(args, layer_idx, save_layer_filename)
            if not args.disable_tqdm:
                tqdm.write(f"layer converted {layer_idx}")
    index_map_path = f"{args.save_path}/model.safetensors.index.json"
    if not os.path.exists(f"{index_map_path}.done"):
        full_index_map = {}
        for layer_idx in range(int(args.num_layers + 2)):
            save_layer_filename = (
                f"model-{layer_idx + 1:05d}-of-{args.num_layers + 2:05d}"
            )
            with open(f"{args.save_path}/{save_layer_filename}.index_map", "r") as f:
                index_map = json.load(f)
                exist_keys = set(full_index_map.keys()).intersection(index_map.keys())
                assert len(exist_keys) == 0, f"exist_keys: {exist_keys}"
                full_index_map.update(index_map)
        with open(index_map_path, "w") as f:
            json.dump({"metadata": {}, "weight_map": full_index_map}, f, indent=4)
        with open(f"{index_map_path}.done", "w") as f:
            f.write("done")
        for layer_idx in range(int(args.num_layers + 2)):
            save_layer_filename = (
                f"model-{layer_idx + 1:05d}-of-{args.num_layers + 2:05d}"
            )
            os.remove(f"{args.save_path}/{save_layer_filename}.index_map")
    print(f"all allclose! time_elapse: {time.time() - time_start:0.3f}s")


if __name__ == "__main__":
    main()
