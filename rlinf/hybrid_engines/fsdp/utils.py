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

import torch
from accelerate import init_empty_weights
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
from transformers.trainer_pt_utils import get_module_class_from_name


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

    fsdp_transformer_layer_cls_to_wrap = config.get(
        "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
    )

    # Build policies list
    policies = []

    # Add vision transformer policies for VLA models
    if is_vla_model:
        from prismatic.extern.hf.modeling_prismatic import PrismaticProjector
        from timm.models.vision_transformer import VisionTransformer
        from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy

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
        from prismatic.extern.hf.modeling_prismatic import PrismaticProjector

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
