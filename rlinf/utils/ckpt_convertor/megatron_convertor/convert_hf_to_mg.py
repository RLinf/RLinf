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
from concurrent.futures import Future, ProcessPoolExecutor
from copy import deepcopy

import torch
from omegaconf.dictconfig import DictConfig

from .config import ConvertorConfig, load_convertor_config
from .convert_hf_to_middle_file import (
    convert_layer_loadsave,
)
from .convert_hf_to_middle_file import (
    get_megatron_iteration as get_megatron_iteration_from_hf,
)
from .convert_middle_file_to_mg import (
    CKPTSaver,
    convert_layer_load,
    get_hetero_pp_rank,
    merge_checkpoint_dict,
)
from .convert_middle_file_to_mg import (
    get_megatron_iteration as get_megatron_iteration_to_mg,
)
from .utils import (
    Operation,
    STLoaderLazy,
    get_device_initializer,
    single_thread_init,
)


class OperationAwareDeviceInitializer:
    def __init__(self, convert_config):
        self.convert_config = convert_config
        self.original_initializer = get_device_initializer(convert_config)

    def __call__(self):
        self.original_initializer()

        Operation.global_tp = self.convert_config.tp_size
        Operation.global_tpe = self.convert_config.tpe_size
        Operation.global_ep = self.convert_config.ep_size
        Operation.global_pp = self.convert_config.pp_size


def hf_to_middle_file(convert_config: ConvertorConfig) -> None:
    """
    Convert a Hugging Face checkpoint to a middle file format.

    Args:
        convert_config (ConvertorConfig): Configuration for the conversion process.
    """
    iteration = get_megatron_iteration_from_hf(convert_config)

    if not os.path.exists(convert_config.save_path):
        os.makedirs(convert_config.save_path)
    if iteration == -1:
        with open(
            f"{convert_config.save_path}/latest_checkpointed_iteration.txt", "w"
        ) as f:
            f.write("release")
    else:
        with open(
            f"{convert_config.save_path}/latest_checkpointed_iteration.txt", "w"
        ) as f:
            f.write(str(iteration))

    hfst_loader = STLoaderLazy.from_path(convert_config.load_path)
    convert_layers = []
    for layer_idx in range(int(convert_config.num_layers + 2)):
        run_covert = True

        # loader and saver per layer
        assert convert_config.save_path is not None
        if layer_idx == convert_config.num_layers:
            save_layer_path = f"{convert_config.save_path}/pre.safetensors"
        elif layer_idx == convert_config.num_layers + 1:
            save_layer_path = f"{convert_config.save_path}/post.safetensors"
        else:
            save_layer_path = f"{convert_config.save_path}/{layer_idx}.safetensors"
        if os.path.exists(save_layer_path):
            if os.path.exists(f"{save_layer_path}.done"):
                run_covert = False
            else:
                os.remove(save_layer_path)

        if run_covert:
            convert_layers.append((layer_idx, save_layer_path))

    if convert_config.process_num > 1:
        with ProcessPoolExecutor(
            convert_config.process_num,
            initializer=get_device_initializer(convert_config),
        ) as mp_exec:
            handles: list[Future] = []
            for layer_idx, save_layer_path in convert_layers:
                handles.append(
                    (
                        layer_idx,
                        mp_exec.submit(
                            convert_layer_loadsave,
                            convert_config,
                            hfst_loader,
                            layer_idx,
                            save_layer_path,
                        ),
                    )
                )
            for layer_idx, t in handles:
                exp = t.exception()
                if exp is not None:
                    mp_exec.shutdown(True, cancel_futures=True)
                    raise exp
                t.result()
    else:
        # run in one thread
        single_thread_init(convert_config)
        for layer_idx, save_layer_path in convert_layers:
            convert_layer_loadsave(
                convert_config, hfst_loader, layer_idx, save_layer_path
            )


def middle_file_to_mg(convert_config: ConvertorConfig) -> None:
    Operation.global_tp = convert_config.tp_size
    Operation.global_tpe = convert_config.tpe_size
    Operation.global_ep = convert_config.ep_size
    Operation.global_pp = convert_config.pp_size

    iteration = get_megatron_iteration_to_mg(convert_config)

    if not os.path.exists(convert_config.save_path):
        os.makedirs(convert_config.save_path)
    if iteration == -1:
        with open(
            f"{convert_config.save_path}/latest_checkpointed_iteration.txt", "w"
        ) as f:
            f.write("release")
        release_dir = os.path.join(convert_config.save_path, "release")
    else:
        with open(
            f"{convert_config.save_path}/latest_checkpointed_iteration.txt", "w"
        ) as f:
            f.write(str(iteration))
        release_dir = os.path.join(convert_config.save_path, f"iter_{iteration:07d}")

    if not os.path.exists(release_dir):
        os.makedirs(release_dir)

    if iteration == -1:
        save_dir = os.path.join(convert_config.save_path, "release")
    else:
        save_dir = os.path.join(convert_config.save_path, f"iter_{iteration:07d}")

    if convert_config.schedular == "1f1b":
        vpp_size = None
    elif convert_config.schedular == "dualpipev":
        vpp_size = 2
    else:
        raise ValueError(f"Unsupported schedular: {convert_config.schedular}")
    pp_rank_to_layers = {i: [] for i in range(Operation.global_pp)}
    for layer in range(int(convert_config.num_layers)):
        ppvpp_rank, local_layer, local_num_layer = get_hetero_pp_rank(
            convert_config, convert_config.num_layers, layer
        )
        if convert_config.schedular == "1f1b":
            pp_rank = ppvpp_rank
            model_key_vpp = "model"
        elif convert_config.schedular == "dualpipev":
            if ppvpp_rank < Operation.global_pp:
                pp_rank = ppvpp_rank
                model_key_vpp = "model0"
            else:
                pp_rank = Operation.global_pp * 2 - 1 - ppvpp_rank
                model_key_vpp = "model1"
        else:
            raise ValueError(f"Unsupported schedular: {convert_config.schedular}")
        pp_rank_to_layers[pp_rank].append(
            (
                model_key_vpp,
                layer,
                local_layer,
                local_num_layer,
            )
        )

    if convert_config.process_num > 1:
        operation_aware_initializer = OperationAwareDeviceInitializer(convert_config)
        with (
            ProcessPoolExecutor(convert_config.process_num) as saver_exec,
            ProcessPoolExecutor(
                convert_config.process_num, initializer=operation_aware_initializer
            ) as mp_exec,
        ):
            spliter_handles = []
            saver_handles = []
            nums_pp_rank = {}
            full_checkpoint_pp_rank = {}
            for pp_rank in range(Operation.global_pp):
                full_checkpoint = CKPTSaver.reset_ckpt_name(save_dir, pp_rank, vpp_size)
                if len(full_checkpoint) == 0:
                    continue
                full_checkpoint_pp_rank[pp_rank] = []
                for layer_info in pp_rank_to_layers[pp_rank]:
                    if pp_rank not in nums_pp_rank:
                        nums_pp_rank[pp_rank] = 0
                    nums_pp_rank[pp_rank] += 1
                    spliter_handles.append(
                        (
                            pp_rank,
                            mp_exec.submit(
                                convert_layer_load,
                                convert_config,
                                layer_info,
                                pp_rank,
                                full_checkpoint,
                            ),
                        )
                    )
            for pp_rank, t in spliter_handles:
                exp = t.exception()
                if exp is not None:
                    mp_exec.shutdown(True, cancel_futures=True)
                    saver_exec.shutdown(True, cancel_futures=True)
                    raise exp
                full_checkpoint_pp_rank[pp_rank].append(t.result())
                nums_pp_rank[pp_rank] -= 1
                if nums_pp_rank[pp_rank] == 0:
                    all_ckpts = merge_checkpoint_dict(
                        full_checkpoint_pp_rank.pop(pp_rank),
                        convert_config.use_gpu_num > 0,
                    )
                    for i, (ckpt_key, ckpt_value) in enumerate(all_ckpts.items()):
                        if convert_config.te_ln_add_extra_state is not None:
                            if convert_config.te_ln_add_extra_state == "none":
                                extra_state = None
                            elif (
                                convert_config.te_ln_add_extra_state
                                == "tensor_pickle_none"
                            ):
                                import pickle

                                state_serialized = bytearray(pickle.dumps(None))
                                extra_state = torch.frombuffer(
                                    state_serialized, dtype=torch.uint8
                                )
                            elif convert_config.te_ln_add_extra_state == "tensor_0_dim":
                                extra_state = torch.tensor([])
                            else:
                                assert False, (
                                    "te_ln_add_extra_state only avail in [None, 'none', 'tensor_pickle_none', 'tensor_0_dim']"
                                )

                            for model_key in ckpt_value.keys():
                                if not model_key.startswith("model"):
                                    continue
                                model_dict = ckpt_value[model_key]
                                extra_states_to_add = {}
                                keys_to_check = list(model_dict.keys())
                                for key_name in keys_to_check:
                                    if key_name.startswith("decoder.") and (
                                        key_name.endswith(".weight")
                                        or key_name.endswith(".bias")
                                        # moe grouped gemm
                                        or key_name.endswith(".weight0")
                                    ):
                                        module_base_name = key_name.rsplit(".", 1)[0]
                                        if module_base_name.endswith(".router"):
                                            # moe model router.weight don't need _extra_state
                                            continue
                                        extra_state_key = (
                                            f"{module_base_name}._extra_state"
                                        )
                                        if extra_state_key not in model_dict:
                                            extra_states_to_add[extra_state_key] = (
                                                extra_state
                                            )
                                model_dict.update(extra_states_to_add)

                        ckpt_value["iteration"] = iteration
                        ckpt_value["checkpoint_version"] = 3.0
                        saver_handles.append(
                            (
                                pp_rank,
                                i,
                                len(all_ckpts),
                                saver_exec.submit(
                                    CKPTSaver.save_ckpt_one,
                                    save_dir,
                                    ckpt_key,
                                    ckpt_value,
                                ),
                            )
                        )
            for pp_rank, i, size, t in saver_handles:
                exp = t.exception()
                if exp is not None:
                    mp_exec.shutdown(True, cancel_futures=True)
                    saver_exec.shutdown(True, cancel_futures=True)
                    raise exp
                t.result()
    else:
        # run in one thread
        single_thread_init(convert_config)
        for pp_rank in range(Operation.global_pp):
            full_checkpoint = CKPTSaver.reset_ckpt_name(save_dir, pp_rank, vpp_size)
            if len(full_checkpoint) == 0:
                continue
            full_checkpoints = []
            for i, layer_info in enumerate(pp_rank_to_layers[pp_rank]):
                full_checkpoints.append(
                    convert_layer_load(
                        convert_config, layer_info, pp_rank, deepcopy(full_checkpoint)
                    )
                )
            for ckpt_key, ckpt_value in merge_checkpoint_dict(full_checkpoints).items():
                ckpt_value["iteration"] = iteration
                ckpt_value["checkpoint_version"] = 3.0
                CKPTSaver.save_ckpt_one(save_dir, ckpt_key, ckpt_value)


def convert_hf_to_mg(
    hf_ckpt_path: str,
    ckpt_cfg: DictConfig,
):
    """
    Convert a Hugging Face checkpoint to a Megatron-LM checkpoint.

    Args:
        hf_ckpt_path (str): Path to the Hugging Face checkpoint file.
        ckpt_cfg (DictConfig): Configuration for the checkpoint conversion, including paths and model parameters.
    """

    # load hf model to get config info
    convert_config = load_convertor_config(hf_ckpt_path, ckpt_cfg)
    load_path = convert_config.load_path
    save_path = convert_config.save_path
    assert (
        convert_config.te_ln_add_extra_state
        in [None, "none", "tensor_pickle_none", "tensor_0_dim"]
        and convert_config.te_ln_linear_qkv is True
        and convert_config.te_ln_linear_mlp_fc1 is True
    )
    print(f"Checkpoint convert config: {convert_config}")

    print("Start to convert huggingface checkpoint to megatron checkpoint...")
    hf_to_middle_file(convert_config)
    # adjust to script's requirement
    convert_config.load_path = save_path

    middle_file_to_mg(convert_config)
    convert_config.load_path = load_path

    # post process: copy any config json file to mg_ckpt_path
    assert os.path.exists(convert_config.save_path), (
        f"Megatron checkpoint path {convert_config.save_path} does not exist."
    )
    for file in os.listdir(hf_ckpt_path):
        if (
            file.endswith(".json")
            or file.endswith(".py")
            or file.lower().endswith("*.md")
            or file == "LICENSE"
        ):
            os.system(
                f"cp {os.path.join(hf_ckpt_path, file)} {convert_config.save_path}"
            )

    # delete middle file .done
    for file in os.listdir(convert_config.save_path):
        if file.endswith(".safetensors") or file.endswith(".done"):
            os.remove(os.path.join(convert_config.save_path, file))
    if convert_config.iteration == -1:
        release_dir = os.path.join(convert_config.save_path, "release")
        if not os.path.exists(release_dir):
            raise ValueError(
                f"release dir {release_dir} does not exist, save path is {convert_config.save_path}"
            )
        for file in os.listdir(release_dir):
            if file.endswith(".done"):
                os.remove(os.path.join(release_dir, file))
    else:
        iter_dir = os.path.join(
            convert_config.save_path, f"iter_{convert_config.iteration:07d}"
        )
        if not os.path.exists(iter_dir):
            raise ValueError(
                f"iter dir {iter_dir}, f'iter_{convert_config} does not exist."
            )
        for file in os.listdir(iter_dir):
            if file.endswith(".done"):
                os.remove(os.path.join(iter_dir, file))

    print(
        f"Finish converting hf checkpoint to megatron checkpoint, converted checkpoint saved at {convert_config.save_path}"
    )
