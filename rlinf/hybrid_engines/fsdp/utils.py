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

import functools
from contextlib import nullcontext

import torch
from accelerate import init_empty_weights
from packaging import version
from prismatic.extern.hf.modeling_prismatic import PrismaticProjector
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
from torch.optim import Optimizer
from transformers.trainer_pt_utils import get_module_class_from_name

if version.parse(torch.__version__) >= version.parse("2.6"):
    from torch.distributed.fsdp import (
        FSDPModule,
        fully_shard,
    )
elif version.parse(torch.__version__) >= version.parse("2.4"):
    from torch.distributed._composable.fsdp import (
        FSDPModule,
        fully_shard,
    )
else:
    fully_shard, FSDPModule = None, None


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


def get_fsdp_wrap_policy(module, config=None, is_lora=False):
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

    # Check if this is a VLA model by looking for language_model attribute
    is_vla_model = hasattr(module, "language_model")

    # Get transformer layer classes to wrap
    if is_vla_model:
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

    # Add vision transformer policies for VLA models
    if is_vla_model:
        from timm.models.vision_transformer import VisionTransformer
        from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy

        # Vision transformer policies
        vit_wrap_policy = functools.partial(
            _module_wrap_policy, module_classes={VisionTransformer}
        )
        policies.append(vit_wrap_policy)

        # Prismatic projector policy for VLA models
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

    # Add transformer layer policies
    if fsdp_transformer_layer_cls_to_wrap is not None:
        transformer_cls_to_wrap = set()
        for layer_class in fsdp_transformer_layer_cls_to_wrap:
            print("layer_class is :", layer_class)
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
    module, config=None, fsdp_kwargs=None, is_vla_model=False, is_lora=False
):
    """
    FSDP2 version of module sharding application, corresponding to FSDP1's auto_wrap_policy logic

    Args:
        module: The model to be sharded
        config: Configuration dictionary
        fsdp_kwargs: FSDP2 parameters
        is_vla_model: Whether to enable VLA-specific wrapping
        is_lora: Whether to enable LoRA-specific wrapping

    Returns:
        The sharded model
    """
    if config is None:
        config = {}

    if fsdp_kwargs is None:
        fsdp_kwargs = {}

    if config.get("disable", False):
        return module

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

    if is_vla_model:
        from timm.models.vision_transformer import VisionTransformer

        for name, submodule in module.named_modules():
            if isinstance(submodule, VisionTransformer):
                modules_to_shard.append((name, submodule, "vision_transformer"))

        from prismatic.extern.hf.modeling_prismatic import PrismaticProjector

        for name, submodule in module.named_modules():
            if isinstance(submodule, PrismaticProjector):
                modules_to_shard.append((name, submodule, "prismatic_projector"))

        if hasattr(module, "value_head"):
            from rlinf.models.embodiment.modules.value_head import ValueHead

            for name, submodule in module.named_modules():
                if isinstance(submodule, ValueHead):
                    modules_to_shard.append((name, submodule, "value_head"))

    if is_lora:
        for name, submodule in module.named_modules():
            if (
                len(list(submodule.named_children())) == 0
                and getattr(submodule, "weight", None) is not None
                and submodule.weight.requires_grad
            ):
                modules_to_shard.append((name, submodule, "lora"))

    for name, submodule, module_type in modules_to_shard:
        fully_shard(submodule, **fsdp_kwargs)

    fully_shard(module, **fsdp_kwargs)

    return module


def fsdp2_load_full_state_dict(
    model: torch.nn.Module, full_state: dict, cpu_offload=None
):
    """
    Loads the full state dict (could be only on rank 0) into the sharded model. This is done by broadcasting the
    parameters from rank 0 to all other ranks. This function modifies the model in-place.

    Args:
        model (`torch.nn.Module`): The model to load the state dict into
        full_state (`dict`): The full state dict to load, can only be on rank 0
    """
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        set_model_state_dict,
    )

    # To broadcast, it needs to be instantiated in the GPU.
    if torch.distributed.get_rank() == 0:
        model = model.to(device="cuda", non_blocking=True)
    else:
        model = model.to_empty(device="cuda")

    cpu_offload = cpu_offload is not None
    options = StateDictOptions(
        full_state_dict=True, cpu_offload=cpu_offload, broadcast_from_rank0=True
    )
    set_model_state_dict(model, full_state, options=options)

    # rotary_emb is not in state_dict, so we need to broadcast it manually
    for name, buf in model.named_buffers():
        torch.distributed.broadcast(buf, src=0)


def fsdp_version(model):
    if isinstance(model, FSDP):
        return 1
    elif isinstance(model, FSDPModule):
        return 2
    else:
        return 0


def get_fsdp_state_ctx(model, state_type, state_cfg, optim_cfg):
    if fsdp_version(model) == 1:
        return FSDP.state_dict_type(model, state_type, state_cfg, optim_cfg)
    else:
        return nullcontext()


def get_fsdp_full_state_dict(
    model: torch.nn.Module, offload_to_cpu: bool = True, rank0_only: bool = True
):
    """
    Get the full state dict from an FSDP model.

    Args:
        model (torch.nn.Module): The FSDP model to get state dict from
        offload_to_cpu (bool, optional): Whether to offload the state dict to CPU. Defaults to True.
        rank0_only (bool, optional): Whether to only get state dict on rank 0. Defaults to True.

    Returns:
        dict: The full state dict of the model

    Raises:
        NotImplementedError: If the FSDP version is unknown
    """
    if fsdp_version(model) == 1:
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        state_dict_config = FullStateDictConfig(
            offload_to_cpu=offload_to_cpu, rank0_only=rank0_only
        )
        with get_fsdp_state_ctx(
            model,
            state_type=StateDictType.FULL_STATE_DICT,
            state_cfg=state_dict_config,
            optim_cfg=None,
        ):
            state_dict = model.state_dict()
        return state_dict
    elif fsdp_version(model) == 2:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )

        state_dict_config = StateDictOptions(
            full_state_dict=True,
            cpu_offload=offload_to_cpu,
            broadcast_from_rank0=not rank0_only,
        )
        state_dict = get_model_state_dict(model, options=state_dict_config)
        if not rank0_only:
            if torch.distributed.get_rank() == 0:
                obj_list = [state_dict]
            else:
                obj_list = [None]

            torch.distributed.broadcast_object_list(obj_list, src=0)
            state_dict = obj_list[0]
        return state_dict
    else:
        raise NotImplementedError(f"Unknown FSDP version {fsdp_version}")


def get_lr_scheduler(
    warmup_style: str,
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
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
            min_lr_ratio=min_lr_ratio,
            num_cycles=num_cycles,
        )
    else:
        raise NotImplementedError(f"Scheduler type {warmup_style} is not supported")
