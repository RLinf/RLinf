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


import json
import os
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    Literal,
    Optional,
    Sequence,
    Union,
)

import torch
from diffusers.utils import SAFE_WEIGHTS_INDEX_NAME as DIFFUSERS_SAFE_WEIGHTS_INDEX_NAME
from diffusers.utils import (
    SAFETENSORS_WEIGHTS_NAME as DIFFUSERS_SAFETENSORS_WEIGHTS_NAME,
)
from torch import distributed as dist
from torch import nn
from tqdm import tqdm
from transformers.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
)
from transformers.utils.hub import cached_file, get_checkpoint_shard_files
from transformers.utils.import_utils import is_safetensors_available

from ..utils import logging
from ..utils.helper import empty_cache, get_dtype_size

if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.torch import save_file


if TYPE_CHECKING:
    from transformers import (
        GenerationConfig,
        PretrainedConfig,
        PreTrainedModel,
        PreTrainedTokenizer,
        ProcessorMixin,
    )

    ModelAssets = Union[
        GenerationConfig, PretrainedConfig, PreTrainedTokenizer, ProcessorMixin
    ]


logger = logging.get_logger(__name__)


@contextmanager
def init_empty_weights():
    """
    A context manager under which models are initialized with all parameters on the meta device.

    Borrowed from: https://github.com/huggingface/accelerate/blob/v1.0.0rc1/src/accelerate/big_modeling.py#L57
    """
    old_register_parameter = nn.Module.register_parameter

    def register_empty_parameter(module: "nn.Module", name: str, param: "nn.Parameter"):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(
                module._parameters[name].to("meta"), **kwargs
            )

    try:
        nn.Module.register_parameter = register_empty_parameter
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter


@dataclass
class StateDictIterator:
    filepath: str
    prefix: str = ""

    def __iter__(self) -> Generator[tuple[str, "torch.Tensor"], None, None]:
        if self.filepath.endswith(".safetensors"):
            with safe_open(self.filepath, framework="pt", device="cpu") as f:
                for key in f.keys():
                    yield key, f.get_tensor(key)

        else:
            state_dict = torch.load(
                self.filepath, map_location="cpu", weights_only=True, mmap=True
            )
            for key in state_dict.keys():
                yield key, state_dict[key]


def _load_state_dict(
    weights_path: str, expert_vision_path: str | None = None, **kwargs
) -> list["StateDictIterator"]:
    """
    Loads (sharded) state dict in transformers' format.
    """
    cache_kwargs = {"_raise_exceptions_for_missing_entries": False, **kwargs}
    resolved_weight_file = cached_file(weights_path, SAFE_WEIGHTS_NAME, **cache_kwargs)
    if resolved_weight_file:
        return [StateDictIterator(resolved_weight_file)]

    resolved_weight_file = cached_file(
        weights_path, SAFE_WEIGHTS_INDEX_NAME, **cache_kwargs
    )
    if resolved_weight_file:
        if expert_vision_path is not None:
            shard_files, _ = get_checkpoint_shard_files(
                expert_vision_path, resolved_weight_file, **kwargs
            )
        else:
            shard_files, _ = get_checkpoint_shard_files(
                weights_path, resolved_weight_file, **kwargs
            )
        return [StateDictIterator(shard_file) for shard_file in shard_files]

    resolved_weight_file = cached_file(
        weights_path, DIFFUSERS_SAFETENSORS_WEIGHTS_NAME, **cache_kwargs
    )
    if resolved_weight_file:
        return [StateDictIterator(resolved_weight_file)]

    resolved_weight_file = cached_file(
        weights_path, DIFFUSERS_SAFE_WEIGHTS_INDEX_NAME, **cache_kwargs
    )
    if resolved_weight_file:
        shard_files, _ = get_checkpoint_shard_files(
            weights_path, resolved_weight_file, **kwargs
        )
        return [StateDictIterator(shard_file) for shard_file in shard_files]

    resolved_weight_file = cached_file(weights_path, WEIGHTS_NAME, **cache_kwargs)
    if resolved_weight_file:
        return [StateDictIterator(resolved_weight_file)]

    resolved_weight_file = cached_file(weights_path, WEIGHTS_INDEX_NAME, **cache_kwargs)
    if resolved_weight_file:
        shard_files, _ = get_checkpoint_shard_files(
            weights_path, resolved_weight_file, **kwargs
        )
        return [StateDictIterator(shard_file) for shard_file in shard_files]

    raise ValueError(f"Cannot find checkpoint files in {weights_path}.")


def _find_submodule(module: "nn.Module", name: str) -> tuple["nn.Module", str]:
    """
    Finds the leaf module according to the name.
    """
    pieces = name.split(".")
    for piece in pieces[:-1]:
        if not hasattr(module, piece):
            raise ValueError(f"Cannot find {piece} in {module}.")

        module = getattr(module, piece)

    return module, pieces[-1]


def _dispatch_parameter(
    module: "nn.Module",
    name: str,
    tensor: "torch.Tensor",
    dtensor_factory: Optional[
        Callable[["torch.Tensor", Any, Any], "torch.Tensor"]
    ] = None,
) -> None:
    """
    Assigns parameter to an empty model.

    NOTE: FSDP module must use in-place operators.
    """
    module, name = _find_submodule(module, name)
    orig_tensor = module._parameters[name].data
    tensor = tensor.to(orig_tensor)
    if hasattr(orig_tensor, "device_mesh"):  # dtensor
        if orig_tensor.device.type == "cpu":
            raise ValueError("Cannot load dtensor on CPU.")

        device_mesh = getattr(orig_tensor, "device_mesh")
        placements = getattr(orig_tensor, "placements")
        module._parameters[name].data.copy_(
            dtensor_factory(tensor, device_mesh, placements)
        )
    else:  # not dtensor
        module._parameters[name].data.copy_(tensor)


def _dispatch_buffer(
    module: "nn.Module",
    name: str,
    buffer: "torch.Tensor",
) -> None:
    """
    Assigns buffer to an empty model.
    """
    module, name = _find_submodule(module, name)
    orig_tensor = module._buffers[name].data
    module._buffers[name] = buffer.to(orig_tensor)


def _init_parameter(
    module: "nn.Module",
    name: str,
) -> None:
    """
    Initializes parameter in model.
    """
    pieces = name.split(".")
    init_func = None

    for piece in pieces[:-1]:
        if not hasattr(module, piece):
            raise ValueError(f"Cannot find {piece} in {module}.")

        if hasattr(module, "_init_weights"):
            init_func = getattr(module, "_init_weights")

        module = getattr(module, piece)

    if init_func is None:
        print(module)
        raise ValueError(
            f"Cannot retrieve `_init_weights` function in the parents of {module}."
        )

    module.apply(init_func)


def get_model_prefix(parameter_names):
    vlm_prefix = ""
    for param_name in parameter_names:
        parts = param_name.split(".")
        if parts[1] == "paligemma_with_expert" and "expert" not in parts[2]:
            vlm_prefix = ".".join(parts[:3]) + "."
            break
    return vlm_prefix


@torch.no_grad()
def load_model_weights(
    model: Union["nn.Module", "PreTrainedModel"],
    weights_path: str,
    init_device: Literal["cpu", "cuda"] = "cuda",
    dtensor_factory: Optional[
        Callable[["torch.Tensor", Any, Any], "torch.Tensor"]
    ] = None,
    load_vlm_only: bool = False,
    enable_expert_vision: bool = False,
    expert_vision_path: str | None = None,
    post_training: bool = False,
    incremental_training: bool = False,
    depth_incremental_training: bool = False,
    norm_qkv: bool = False,
    adanorm_time: bool = False,
) -> None:
    """
    Loads pre-trained model states in transformers' format.
    """
    buffer_dict = {name: buffer.clone() for name, buffer in model.named_buffers()}
    parameter_names = {name for name, _ in model.named_parameters()}
    vlm_parameter_names = {
        name for name, _ in model.model.qwenvl_with_expert.qwenvl.named_parameters()
    }
    print(f"====vlm contains {len(vlm_parameter_names)} paras=====")
    if expert_vision_path is not None or enable_expert_vision:
        dino_parameter_names = {
            name
            for name, _ in model.model.qwenvl_with_expert.expert_visual.named_parameters()
        }
        print(f"====dino contains {len(dino_parameter_names)} paras=====")
    model.to_empty(device=init_device)
    if post_training:
        logger.info_rank0(
            ">>> Doing Post-Training now, no need to load LLM's embedding weight."
        )
    elif incremental_training:
        logger.info_rank0(">>> Load pretrained weights for incremental training.")
    elif load_vlm_only:
        logger.info_rank0(">>> Doing Pre-Training now.")
    else:
        logger.info_rank0(">>> Fine-tuneing based on PI0 now.")
    # TODO
    state_dict_iterators = _load_state_dict(weights_path, expert_vision_path)
    vlm_perfix = get_model_prefix(parameter_names) if load_vlm_only else ""
    for state_dict_iterator in tqdm(
        state_dict_iterators,
        desc="Loading checkpoint shards",
        disable=int(os.getenv("LOCAL_RANK", "-1")) > 0,
    ):
        for name, tensor in state_dict_iterator:
            if "expert_visual." in name and not post_training:
                name = "model.qwenvl_with_expert." + name
            else:
                name = vlm_perfix + name
            if name in buffer_dict.keys():  # persistent buffers
                buffer_dict[name] = tensor.clone()
            elif name in parameter_names:
                if incremental_training:
                    try:
                        _dispatch_parameter(model, name, tensor, dtensor_factory)
                        parameter_names.remove(name)
                    except Exception:
                        logger.info_rank0(
                            f">>>The {name} weight need to be reinitialized."
                        )
                else:
                    parameter_names.remove(name)
                    _dispatch_parameter(model, name, tensor, dtensor_factory)
            else:
                if post_training:
                    error_msg = f"Unexpected key '{name}' found in state dict during Post-Training. This is not allowed!!!"
                    logger.info_rank0(error_msg)
                    raise KeyError(error_msg)
                if expert_vision_path is not None or enable_expert_vision:
                    assert ".expert_visual." not in name, (
                        "vision encoder need to be inited for action expert!"
                    )
                logger.info_rank0(f"Unexpected key in state dict: {name}.")

        del state_dict_iterator
        empty_cache()

    for name, buffer in buffer_dict.items():
        _dispatch_buffer(model, name, buffer)
    if post_training:
        assert len(parameter_names) == 0, (
            f"Missing {parameter_names} during Post-Training. This is not allowed!!!"
        )
    if len(parameter_names) > 0:
        if (
            load_vlm_only
            and (expert_vision_path is not None or enable_expert_vision)
            and not incremental_training
        ):
            num_missing_vlm_para, num_missing_dino_para = 0, 0
            for name in parameter_names:
                if ".paligemma." in name or ".qwenvl." in name:
                    num_missing_vlm_para += 1
                elif ".expert_visual." in name:
                    num_missing_dino_para += 1
            print(f"====Missing {num_missing_vlm_para} paras in vlm====")
            print(f"====Missing {num_missing_dino_para} paras in DINO====")
            assert (
                all(".paligemma." not in name for name in parameter_names)
                or all(".qwenvl." not in name for name in parameter_names)
            ) and all(".expert_visual." not in name for name in parameter_names), (
                "Parameters in VLM and Expert_Visual are not loaded when PreTraining!!!"
            )
        elif incremental_training and not depth_incremental_training:
            if norm_qkv:
                assert all(
                    "_proj." in name or "_layernorm." in name
                    for name in parameter_names
                ), "Only MLP weight can be reinitialized when IncrementalTraining!!!"
            else:
                assert all(
                    "_proj." in name or "gate" in name for name in parameter_names
                ), "Only MLP weight can be reinitialized when IncrementalTraining!!!"
        elif depth_incremental_training:
            assert all("depth_align_head." in name for name in parameter_names), (
                "Only depth align head weight can be reinitialized when IncrementalTraining with Depth Model!!!"
            )
        elif load_vlm_only:
            num_missing_vlm_para = 0
            for name in parameter_names:
                if ".paligemma." in name or ".qwenvl." in name:
                    num_missing_vlm_para += 1
            print(f"====Missing {num_missing_vlm_para} paras in vlm====")
            assert all(".paligemma." not in name for name in parameter_names) or all(
                ".qwenvl." not in name for name in parameter_names
            ), "Parameters in VLM  are not loaded when PreTraining!!!"
        logger.info_rank0(
            f"Find missing key(s) in state dict: {parameter_names}, initialize them."
        )
        if adanorm_time:
            logger.info_rank0(">>> Parameters in AdaNorm has been ZERO initialized.")
            exclude_keywords = [
                "input_layernorm.gamma_beta_gate",
                "post_attention_layernorm.gamma_beta_gate",
                "norm.gamma_beta_gate",
                "input_layernorm.gamma",
                "post_attention_layernorm.gamma",
                "norm.gamma",
                "input_layernorm.beta",
                "post_attention_layernorm.beta",
                "norm.beta",
                "input_layernorm.gate",
                "post_attention_layernorm.gate",
                "norm.gate",
            ]
        for name in parameter_names:
            if not adanorm_time:
                _init_parameter(model, name)
            else:
                if not any(keyword in name for keyword in exclude_keywords):
                    _init_parameter(model, name)

    # we should tie embeddings after loading weights because to_empty() leads to untied weights,
    # except for fsdp1 (custom init) and fsdp2 (swap tensor) contexts.
    if getattr(model.config, "tie_word_embeddings", True):
        try:
            input_embeddings = model.get_input_embeddings()
            output_embeddings = model.get_output_embeddings()
            output_embeddings._parameters["weight"] = input_embeddings._parameters[
                "weight"
            ]
        except Exception as e:
            logger.info_rank0(f"Failed to tie embeddings: {e}")


def _get_shard_info(
    state_dict: dict[str, "torch.Tensor"],
    save_dtype: Optional[Union[str, "torch.dtype"]],
    shard_size: int,
    safe_serialization: bool,
) -> tuple[bool, int, dict[str, str]]:
    """
    Gets the shard information, should be executed at rank 0.
    """
    current_size, total_size = 0, 0
    current_shard, shard_list = [], []
    for name, tensor in state_dict.items():
        if isinstance(save_dtype, str):
            dtype = getattr(torch, save_dtype)
        elif isinstance(save_dtype, torch.dtype):
            dtype = save_dtype
        else:
            dtype = tensor.dtype
        tensor_size = tensor.numel() * get_dtype_size(
            dtype
        )  # dtensor's numel == tensor's numel
        if current_size != 0 and current_size + tensor_size > shard_size:
            total_size += current_size
            shard_list.append(current_shard)
            current_size = 0
            current_shard = []

        current_size += tensor_size
        current_shard.append(name)

    if current_size != 0:
        total_size += current_size
        shard_list.append(current_shard)

    weights_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
    num_shards = len(shard_list)
    weight_map = OrderedDict()
    is_sharded = None
    if num_shards == 1:
        is_sharded = False
        for name in shard_list[0]:
            weight_map[name] = weights_name
    else:
        is_sharded = True
        for shard_idx, shard in enumerate(shard_list):
            prefix, extension = weights_name.rsplit(".", maxsplit=1)
            file_name = f"{prefix}-{shard_idx + 1:05d}-of-{num_shards:05d}.{extension}"
            for name in shard:
                weight_map[name] = file_name

    return is_sharded, total_size, weight_map


def _save_state_dict(
    state_dict: dict[str, "torch.Tensor"],
    path_to_save: "os.PathLike",
    safe_serialization: bool,
) -> None:
    """
    Save function.
    """
    if os.path.exists(path_to_save):
        os.remove(path_to_save)
    if safe_serialization:
        save_file(state_dict, path_to_save, metadata={"format": "pt"})
    else:
        torch.save(state_dict, path_to_save)


@torch.no_grad()
def save_model_weights(
    output_dir: Union[str, "os.PathLike"],
    state_dict: dict[str, "torch.Tensor"],
    global_rank: Optional[int] = None,
    save_dtype: Optional[Union[str, "torch.dtype"]] = "bfloat16",
    shard_size: int = 5_000_000_000,
    safe_serialization: bool = True,
    model_assets: Optional[Sequence["ModelAssets"]] = None,
) -> None:
    """
    Saves full model weights. The model parameters should be either tensor or dtensor.

    If global_rank is given, it will assume it is executed on all ranks.
    """

    os.makedirs(output_dir, exist_ok=True)
    is_sharded, total_size, weight_map = _get_shard_info(
        state_dict, save_dtype, shard_size, safe_serialization
    )
    full_state_dict = OrderedDict()
    prev_file_name = None
    for name, tensor in state_dict.items():
        if hasattr(tensor.data, "full_tensor"):  # dtensor
            tensor = tensor.data.full_tensor()
        else:
            tensor = tensor.data

        if save_dtype:
            tensor = tensor.to(
                dtype=getattr(torch, save_dtype)
                if isinstance(save_dtype, str)
                else save_dtype
            )

        if prev_file_name is not None and weight_map[name] != prev_file_name:
            if global_rank is None or global_rank == 0:
                _save_state_dict(
                    full_state_dict,
                    os.path.join(output_dir, prev_file_name),
                    safe_serialization,
                )
                full_state_dict = OrderedDict()

            empty_cache()
            if (
                global_rank is not None and dist.is_initialized()
            ):  # avoid process hanging
                torch.cuda.synchronize()
                dist.barrier()

        if global_rank is None or global_rank == 0:
            full_state_dict[name] = tensor.detach().cpu()

        prev_file_name = weight_map[name]
        del tensor

    if global_rank is None or global_rank == 0:
        if len(full_state_dict):
            _save_state_dict(
                full_state_dict,
                os.path.join(output_dir, prev_file_name),
                safe_serialization,
            )

        if is_sharded:
            index = {
                "metadata": {"total_size": total_size},
                "weight_map": weight_map,
            }

            index_file = (
                SAFE_WEIGHTS_INDEX_NAME if safe_serialization else WEIGHTS_INDEX_NAME
            )
            with open(os.path.join(output_dir, index_file), "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)

            logger.info(f"Model weight splits saved in {output_dir}.")
        else:
            logger.info(
                f"Model weights saved at {os.path.join(output_dir, prev_file_name)}."
            )

        if model_assets is not None:
            for model_asset in model_assets:
                if hasattr(model_asset, "save_pretrained"):
                    model_asset.save_pretrained(output_dir)
                else:
                    logger.warning(
                        f"Model asset {model_asset} should implement `save_pretrained`."
                    )


def save_model_assets(
    output_dir: Union[str, "os.PathLike"], model_assets: Sequence["ModelAssets"]
):
    for model_asset in model_assets:
        if hasattr(model_asset, "save_pretrained"):
            model_asset.save_pretrained(output_dir)
        else:
            logger.warning(
                f"Model asset {model_asset} should implement `save_pretrained`."
            )
