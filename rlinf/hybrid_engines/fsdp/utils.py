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

# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import concurrent.futures
import functools
import json
import os
import shutil
from enum import Enum
from typing import Optional, Union

import torch
from accelerate import init_empty_weights
from safetensors.torch import save_file
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp.wrap import (
    _module_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.optim import Optimizer
from transformers.trainer_pt_utils import get_module_class_from_name

from rlinf.hybrid_engines.fsdp import (
    BackwardPrefetch,
    CPUOffloadPolicy,
    DTensor,
    MixedPrecisionPolicy,
    ShardingStrategy,
    fully_shard,
)


class FSDPVersion(str, Enum):
    FSDP = "fsdp"
    FSDP2 = "fsdp2"


def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(
            "cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"]
        )
    else:
        device_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(world_size // fsdp_size, fsdp_size),
            mesh_dim_names=["ddp", "fsdp"],
        )
    return device_mesh


def init_fn(x: torch.nn.Module):
    if not torch.distributed.get_rank() == 0:
        x = x.to_empty(device=torch.cuda.current_device(), recurse=False)
        torch.cuda.empty_cache()
    return x


def get_init_weight_context_manager(use_meta_tensor=True):
    def cpu_init_weights():
        return torch.device("cpu")

    if use_meta_tensor:
        init_context = (
            init_empty_weights
            if torch.distributed.get_rank() != 0
            else cpu_init_weights()
        )
    else:
        init_context = cpu_init_weights
    return init_context


def get_fsdp_wrap_policy(module, config=None, is_lora=False, is_openvla_model=False):
    """
    FSDP wrap policy that handles both standard transformer models and VLA models.

    Args:
        module: The model to wrap
        config: Configuration dictionary for wrap policy
        is_lora: Whether to enable LoRA-specific wrapping

    Returns:
        FSDP auto wrap policy function
    """
    if config is None:
        config = {}

    if config.get("disable", False):
        return None

    # Get transformer layer classes to wrap
    if hasattr(module, "language_model"):
        # For VLA models, get transformer classes from language_model submodule
        default_transformer_cls_names_to_wrap = getattr(
            module.language_model, "_no_split_modules", None
        )
    else:
        # For standard models, get transformer classes directly from module
        default_transformer_cls_names_to_wrap = getattr(
            module, "_no_split_modules", None
        )

    fsdp_transformer_layer_cls_to_wrap = config.get("wrap_policy", {}).get(
        "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
    )

    # Build policies list
    policies = []

    from rlinf.models.embodiment.modules.resnet_utils import ResNet10

    resnet_policy = functools.partial(_module_wrap_policy, module_classes={ResNet10})
    policies.append(resnet_policy)

    # Add vision transformer policies for OpenVLA models
    if is_openvla_model:
        from prismatic.extern.hf.modeling_prismatic import PrismaticProjector
        from timm.models.vision_transformer import VisionTransformer

        # Vision transformer policies
        vit_wrap_policy = functools.partial(
            _module_wrap_policy, module_classes={VisionTransformer}
        )
        policies.append(vit_wrap_policy)

        # Prismatic projector policy for VLA models
        # The prismatic package initializes a DistributedOverwatch by default,
        # which initializes accelerate.PartialState, which in turn
        # initializes a torch.distributed process group in gloo.
        # This results in default group being gloo, which does not support CUDA tensors and allreduce average.

        prismatic_fsdp_wrapping_policy = functools.partial(
            _module_wrap_policy,
            module_classes={PrismaticProjector},
        )
        policies.append(prismatic_fsdp_wrapping_policy)

    if hasattr(module, "value_head"):
        from rlinf.models.embodiment.modules.value_head import ValueHead

        value_head_policy = functools.partial(
            _module_wrap_policy, module_classes={ValueHead}
        )
        policies.append(value_head_policy)

    if hasattr(module, "q_head"):
        from rlinf.models.embodiment.modules.q_head import MultiCrossQHead, MultiQHead

        if isinstance(module.q_head, MultiCrossQHead):
            q_head_policy = functools.partial(
                _module_wrap_policy, module_classes={MultiCrossQHead}
            )
        else:
            q_head_policy = functools.partial(
                _module_wrap_policy, module_classes={MultiQHead}
            )
        policies.append(q_head_policy)

    # Add transformer layer policies
    if fsdp_transformer_layer_cls_to_wrap is not None:
        transformer_cls_to_wrap = set()
        for layer_class in fsdp_transformer_layer_cls_to_wrap:
            transformer_cls = get_module_class_from_name(module, layer_class)
            if transformer_cls is None:
                raise Exception(
                    "Could not find the transformer layer class to wrap in the model."
                )
            else:
                transformer_cls_to_wrap.add(transformer_cls)

        llm_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            # Transformer layer class to wrap
            transformer_layer_cls=transformer_cls_to_wrap,
        )
        policies.append(llm_wrap_policy)

    # Add LoRA lambda policy if enabled
    if is_lora:
        from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

        def lambda_policy_fn(module):
            return bool(
                len(list(module.named_children())) == 0
                and getattr(module, "weight", None) is not None
                and module.weight.requires_grad
                and getattr(module, "_to_lora", True) is True
            )

        lambda_policy = functools.partial(
            lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn
        )
        policies.append(lambda_policy)

    # Return appropriate policy based on number of policies
    if len(policies) == 0:
        return None
    elif len(policies) == 1:
        return policies[0]
    else:
        # Multiple policies - combine with _or_policy
        from torch.distributed.fsdp.wrap import _or_policy

        return functools.partial(_or_policy, policies=policies)


def apply_fsdp2_to_model(
    module,
    config: dict,
    device_mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy,
    offload_policy: CPUOffloadPolicy,
    reshard_after_forward: bool,
):
    """
    FSDP2 version of module sharding application, corresponding to FSDP1's auto_wrap_policy logic

    Args:
        module: The model to be sharded
        config: Configuration dictionary
        device_mesh: The device mesh to use for sharding
        mp_policy: Mixed precision policy
        offload_policy: CPU offload policy
        reshard_after_forward: Whether to reshard after forward pass

    Returns:
        The sharded model
    """
    if config is None:
        config = {}

    if hasattr(module, "language_model"):
        default_transformer_cls_names_to_wrap = getattr(
            module.language_model, "_no_split_modules", None
        )
    else:
        default_transformer_cls_names_to_wrap = getattr(
            module, "_no_split_modules", None
        )

    fsdp_transformer_layer_cls_to_wrap = config.get("wrap_policy", {}).get(
        "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
    )

    if isinstance(fsdp_transformer_layer_cls_to_wrap, str):
        fsdp_transformer_layer_cls_to_wrap = [fsdp_transformer_layer_cls_to_wrap]

    assert (
        len(fsdp_transformer_layer_cls_to_wrap) > 0
        and fsdp_transformer_layer_cls_to_wrap[0] is not None
    )

    modules_to_shard = []

    for name, submodule in module.named_modules():
        if submodule.__class__.__name__ in fsdp_transformer_layer_cls_to_wrap or (
            isinstance(submodule, torch.nn.Embedding)
            and not getattr(module.config, "tie_word_embeddings", False)
        ):
            modules_to_shard.append((name, submodule, "transformer_or_embedding"))

    for name, submodule, module_type in modules_to_shard:
        fully_shard(
            submodule,
            mesh=device_mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            reshard_after_forward=reshard_after_forward,
        )

    return fully_shard(
        module,
        mesh=device_mesh,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        reshard_after_forward=False,
    )


def get_fsdp2_full_state_dict_all_ranks(
    model: torch.nn.Module, offload_to_cpu: bool = True
):
    """
    Get the full state dict for all ranks
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP2

    with FSDP2.summon_full_params(model, writeback=False):
        state_dict = model.state_dict()
        clean_state_dict = {}
        device = (
            torch.device("cpu") if offload_to_cpu else next(model.parameters()).device
        )

        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                clean_value = (
                    value.to(device, non_blocking=True).full_tensor()
                    if hasattr(value, "full_tensor")
                    else value.to(device, non_blocking=True)
                )
                clean_state_dict[key] = clean_value
            else:
                clean_state_dict[key] = value
        return clean_state_dict


def get_lr_scheduler(
    warmup_style: str,
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    if warmup_style == "constant":
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1.0, num_warmup_steps))
            return 1.0

        return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
    elif warmup_style == "cosine":
        from transformers.optimization import get_cosine_schedule_with_warmup

        return get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
        )
    else:
        raise NotImplementedError(f"Scheduler type {warmup_style} is not supported")


def to_local_if_dtensor(tensor: Union[torch.Tensor, DTensor]) -> torch.Tensor:
    """Returns the local shard of the given tensor if it is a DTensor.

    Taken and modified from: https://github.com/NVIDIA/Megatron-LM/blob/605f618f237cda8fa80132bc2ccff933512d5a0d/megatron/core/utils.py#L746
    """
    with torch.no_grad():
        return tensor.to_local() if isinstance(tensor, DTensor) else tensor


@torch.no_grad()
def clip_grad_by_total_norm_(
    parameters: Union[list[Union[torch.Tensor, DTensor]], Union[torch.Tensor, DTensor]],
    max_grad_norm: Union[int, float],
    total_norm: float,
    dtype: torch.dtype = torch.float32,
):
    """Clips gradient of an iterable of parameters by total norm.

    Taken and modified from: https://github.com/NVIDIA/Megatron-LM/blob/a695b2bd2a0ca9ca63385a48c41a1c5a033cdd1e/megatron/core/optimizer/clip_grads.py#L138

    Note that the gradients are modified in place.

    Args:
        parameters (Union[list[Union[torch.Tensor, DTensor]], Union[torch.Tensor, DTensor]]):
            An iterable of Tensors or DTensors, or a single Tensor or DTensor
            that will have gradients normalized.
        max_grad_norm (Union[float, int]): Maximum norm of the gradients.
        total_norm (float): The pre-computed total norm of the gradients to use for scaling.
    """
    if isinstance(parameters, (torch.Tensor, DTensor)):
        parameters = [parameters]

    # Grads.
    grads = [
        to_local_if_dtensor(p.grad.detach()).to(dtype)
        for p in parameters
        if p.grad is not None
    ]

    # Scale.
    clip_coeff = max_grad_norm / (total_norm + 1.0e-6)

    if clip_coeff < 1.0:
        for g in grads:
            g.mul_(clip_coeff)


@torch.no_grad()
def get_grad_norm(
    parameters: Union[list[Union[torch.Tensor, DTensor]], Union[torch.Tensor, DTensor]],
    dp_group: torch.distributed.ProcessGroup,
    norm_type: Union[int, float] = 2,
    dtype: torch.dtype = torch.float32,
) -> float:
    """Calculate the norm of gradients.

    Taken and modified from: https://github.com/NVIDIA/Megatron-LM/blob/a695b2bd2a0ca9ca63385a48c41a1c5a033cdd1e/megatron/core/optimizer/clip_grads.py#L51

    Args:
        parameters (Union[list[Union[torch.Tensor, DTensor]], Union[torch.Tensor, DTensor]]):
            An iterable of Tensors or DTensors, or a single Tensor or DTensor
            that will have gradient norm calculated.
        dp_group (torch.distributed.ProcessGroup): Process group for data parallel communication.
        norm_type (Union[int, float]): Type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        float: Total norm of the gradients (viewed as a single vector)
    """
    if isinstance(parameters, (torch.Tensor, DTensor)):
        parameters = [parameters]

    # Grads.
    grads_for_norm = [
        to_local_if_dtensor(p.grad.detach()).to(dtype)
        for p in parameters
        if p.grad is not None
    ]

    # Norm parameters.
    norm_type = float(norm_type)

    # If there are no gradients to norm (e.g., no trainable params or all grads are None),
    # directly return 0.0 to avoid constructing tensors or calling .cuda() on a float.
    if len(grads_for_norm) == 0:
        return 0.0

    total_norm = 0.0

    # Calculate norm.
    if norm_type == torch.inf:
        total_norm = max(grad.abs().max().item() for grad in grads_for_norm)
        total_norm_cuda = torch.tensor(
            [float(total_norm)], dtype=torch.float, device="cuda"
        )
        # Take max across all data-parallel GPUs if using FSDP and then all model-parallel GPUs.
        if dp_group is not None:
            torch.distributed.all_reduce(
                total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=dp_group
            )
        total_norm = total_norm_cuda[0].item()

    else:
        # Accumulate p-norm over all gradients.
        for grad in grads_for_norm:
            grad_norm = torch.norm(grad, norm_type)
            total_norm += grad_norm**norm_type

        # Ensure total_norm is a tensor on CUDA before all_reduce.
        if not isinstance(total_norm, torch.Tensor):
            total_norm = torch.tensor(
                float(total_norm),
                dtype=torch.float,
                device=grads_for_norm[0].device,
            )
        else:
            total_norm = total_norm.to(device=grads_for_norm[0].device)

        # Sum across all data-parallel GPUs if using FSDP and then all model-parallel GPUs.
        if dp_group is not None:
            torch.distributed.all_reduce(
                total_norm, op=torch.distributed.ReduceOp.SUM, group=dp_group
            )
        total_norm = total_norm.item() ** (1.0 / norm_type)  # type: ignore

    return float(total_norm)


def get_sharding_strategy(strategy_str: str) -> ShardingStrategy:
    """
    Get FSDP sharding strategy from string.

    Args:
        strategy_str (str): The sharding strategy as a string. Can be "full_shard", "shard_grad_op", "hybrid_shard", or "no_shard".

    Returns:
        ShardingStrategy: The corresponding ShardingStrategy enum value.
    """
    SHARDING_STRATEGIES = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
        "no_shard": ShardingStrategy.NO_SHARD,
    }
    assert strategy_str in SHARDING_STRATEGIES, (
        f"Unknown sharding strategy: {strategy_str}"
    )
    return SHARDING_STRATEGIES[strategy_str]


def get_backward_prefetch_strategy(
    prefetch_str: Optional[str],
) -> Optional[BackwardPrefetch]:
    """
    Get the backward prefetch strategy from string.

    Args:
        prefetch_str (Optional[str]): The prefetch strategy as a string. Can be "pre", "post", or None.

    Returns:
        Optional[BackwardPrefetch]: The corresponding BackwardPrefetch enum value or None.
    """
    if prefetch_str is None:
        return None
    BACKWARD_PREFETCH_STRATEGIES = {
        "pre": BackwardPrefetch.BACKWARD_PRE,
        "post": BackwardPrefetch.BACKWARD_POST,
    }
    assert prefetch_str in BACKWARD_PREFETCH_STRATEGIES, (
        f"Unknown backward prefetch strategy: {prefetch_str}"
    )
    return BACKWARD_PREFETCH_STRATEGIES[prefetch_str]


def _tensor_nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def save_state_dict_sharded_safetensors(
    state_dict: dict,
    out_dir: str,
    base_name: str = "model",
    max_shard_size: float | int = 4 * 1024**3,
) -> tuple[int, int]:
    """
    Save the state dict in sharded safetensors format. It will
    first record every tensor that needs to be stored, and create shard plan.
    After this, it will use thread pool to write to safetensors according
    to the sharded plan.

    Args:
        state_dict(dict[str,torch.tensor]): The state dict to save.
        out_dir(str): where to save the sharded safetensors files.
        base_name(str): The base name for the sharded files.
        max_shard_size(int|float): The maximum size of each shard in bytes. Default is 4GB.

    Returns:
        tuple[int,int]: number of shards created and total size in bytes.
    """
    os.makedirs(out_dir, exist_ok=True)

    items = [(k, v) for k, v in state_dict.items() if torch.is_tensor(v)]
    items.sort(key=lambda kv: kv[0])

    # Plan shards
    shards_plan = []
    current_shard_keys = []
    current_shard_bytes = 0

    def flush_plan():
        nonlocal current_shard_keys, current_shard_bytes
        if not current_shard_keys:
            return
        shards_plan.append((current_shard_keys, current_shard_bytes))
        current_shard_keys = []
        current_shard_bytes = 0

    for name, t in items:
        # Calculate size without moving to CPU
        nbytes = _tensor_nbytes(t)

        if nbytes > max_shard_size:
            flush_plan()
            current_shard_keys.append(name)
            current_shard_bytes = nbytes
            flush_plan()
            continue

        if current_shard_bytes + nbytes > max_shard_size and current_shard_keys:
            flush_plan()

        current_shard_keys.append(name)
        current_shard_bytes += nbytes

    flush_plan()

    num_shards = len(shards_plan)
    total_size = sum(b for _, b in shards_plan)
    weight_map = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []

        for idx, (keys, _) in enumerate(shards_plan):
            shard_idx = idx + 1
            shard_dict = {}

            # (CPU transfer happens here)
            for k in keys:
                t = state_dict[k]
                t = t.detach()
                if t.device.type != "cpu":
                    t = t.cpu()
                if not t.is_contiguous():
                    t = t.contiguous()
                shard_dict[k] = t

            fname = f"{base_name}-{shard_idx:05d}-of-{num_shards:05d}.safetensors"
            fpath = os.path.join(out_dir, fname)

            future = executor.submit(
                save_file, shard_dict, fpath, metadata={"format": "pt"}
            )
            futures.append(future)

            for k in keys:
                weight_map[k] = fname

        # Wait for all tasks to complete and check for exceptions
        for future in concurrent.futures.as_completed(futures):
            future.result()

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(
        os.path.join(out_dir, f"{base_name}.safetensors.index.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    return num_shards, total_size


def copy_model_config_and_code(
    model_path: str,
    save_path: str,
    suffixes: tuple[str, ...] = (
        ".py",
        ".json",
        ".md",
    ),
) -> None:
    """
    Recursively copies files with specific suffixes from model_path to save_path.
    """
    if not os.path.exists(model_path):
        return

    os.makedirs(save_path, exist_ok=True)

    for root, _, files in os.walk(model_path):
        for file in files:
            if file.endswith(suffixes):
                src_file = os.path.join(root, file)
                rel_path = os.path.relpath(src_file, model_path)
                dst_file = os.path.join(save_path, rel_path)

                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                shutil.copy2(src_file, dst_file)
