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

import torch
import torch.optim as optim
from omegaconf import DictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from transformers import AutoModelForCausalLM

from rlinf.config import torch_dtype_from_precision
from rlinf.hybrid_engines.fsdp.utils import (
    get_fsdp_wrap_policy,
    init_fn,
)
from rlinf.utils.utils import clear_memory


class FSDPModelManager:
    """
    FSDP Model Manager for RL training
    """

    def __init__(self, cfg: DictConfig):
        self._cfg = cfg
        self.torch_dtype = torch_dtype_from_precision(
            self._cfg.model.get("precision", None)
        )
        self.gradient_checkpointing_enable = self._cfg.model.get(
            "gradient_checkpointing_enable", False
        )

    def model_provider_func(self) -> torch.nn.Module:
        if self._cfg.model.get("gptq_model", False):
            from auto_gptq import AutoGPTQForCausalLM

            model_wrapper = AutoGPTQForCausalLM.from_quantized(
                self._cfg.model.model_path, device="cuda:0", use_triton=True
            )
            model = model_wrapper.model
        elif self._cfg.model.get("load_in_8bit", False):
            model = AutoModelForCausalLM.from_pretrained(
                self._cfg.model.model_path,
                device_map=self._cfg.model.get("device_map", "auto"),
                load_in_8bit=True,
            )
        else:
            # default load in float16
            model = AutoModelForCausalLM.from_pretrained(
                self._cfg.model.model_path,
                torch_dtype=self.torch_dtype,
                device_map=self._cfg.model.get("device_map", "auto"),
                trust_remote_code=True,
                use_safetensors=self._cfg.model.get("use_safetensors", False),
            )
            if torch.cuda.is_available():
                model = model.cuda()
            if self.torch_dtype == torch.float16:
                model = model.half()

        return model

    def setup_model_and_optimizer(self):
        """Setup model and optimizer."""
        module = self.model_provider_func()

        if (
            hasattr(module, "gradient_checkpointing_enable")
            and self.gradient_checkpointing_enable
        ):
            module.gradient_checkpointing_enable()

        mixed_precision = MixedPrecision(
            param_dtype=self.torch_dtype,
            reduce_dtype=self.torch_dtype,
            buffer_dtype=self.torch_dtype,
        )

        if self._cfg.model.sharding_strategy == "full_shard":
            sharding_strategy = ShardingStrategy.FULL_SHARD
        elif self._cfg.model.sharding_strategy == "shard_grad_op":
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        else:
            sharding_strategy = ShardingStrategy.NO_SHARD
        auto_wrap_policy = get_fsdp_wrap_policy(
            module=module, config=None, is_lora=self._cfg.model.is_lora
        )

        betas = (self._cfg.optim.adam_beta1, self._cfg.optim.adam_beta2)

        self.model = FSDP(
            module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=int(os.environ["LOCAL_RANK"]),
            sharding_strategy=sharding_strategy,  # zero3
            mixed_precision=mixed_precision,
            sync_module_states=True,
        )

        params_actor = []
        params_critic = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "value_proj" in name or "model.value_head" in name:
                    params_critic.append(param)
                else:
                    params_actor.append(param)

        if len(params_critic) > 0:
            self.optimizer = optim.AdamW(
                [
                    {"params": params_actor, "lr": self._cfg.optim.lr, "betas": betas},
                    {
                        "params": params_critic,
                        "lr": self._cfg.optim.value_lr,
                        "betas": betas,
                    },
                ]
            )
        else:
            self.optimizer = optim.AdamW(
                [
                    {
                        "params": params_actor,
                        "lr": self._cfg.optim.value_lr,
                        "betas": betas,
                    },
                ]
            )

    def get_model_state_dict(self):
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
            state_dict = self.model.state_dict()
        return state_dict

    def get_optimizer_state_dict(self):
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
            state_dict = FSDP.optim_state_dict(self.model, self.optimizer)
        return state_dict

    def offload_fsdp_grad(self):
        for _, param in self.model.named_parameters():
            if param.grad is not None:
                param.grad = param.grad.to("cpu", non_blocking=True)
        clear_memory()

    def load_fsdp_grad(self, device_id):
        for _, param in self.model.named_parameters():
            if param.grad is not None:
                param.grad = param.grad.to(device_id, non_blocking=True)
        clear_memory()

    def offload_fsdp_param_and_grad(self, offload_grad=False):
        for _, param in self.model.named_parameters():
            if hasattr(param, "_handle") and param._handle is not None:
                flat_param = param._handle.flat_param
                if (
                    hasattr(flat_param, "_local_shard")
                    and flat_param._local_shard is not None
                ):
                    flat_param._local_shard = flat_param._local_shard.to(
                        "cpu", non_blocking=True
                    )
                if flat_param.data is not None:
                    flat_param.data = flat_param.data.to("cpu", non_blocking=True)
                    flat_param._local_shard = flat_param.data
            elif hasattr(param, "_local_shard") and param._local_shard is not None:
                param._local_shard = param._local_shard.to("cpu", non_blocking=True)

            if param.data is not None:
                param.data = param.data.to("cpu", non_blocking=True)

            if offload_grad and param.grad is not None:
                param.grad = param.grad.to("cpu", non_blocking=True)
        clear_memory()

    def load_fsdp_param_and_grad(self, device_id, load_grad=False):
        for _, param in self.model.named_parameters():
            if hasattr(param, "_handle") and param._handle is not None:
                flat_param = param._handle.flat_param
                if (
                    hasattr(flat_param, "_local_shard")
                    and flat_param._local_shard is not None
                ):
                    flat_param._local_shard = flat_param._local_shard.to(
                        device_id, non_blocking=True
                    )
                if flat_param.data is not None:
                    flat_param.data = flat_param.data.to(device_id, non_blocking=True)
                    flat_param._local_shard = flat_param.data
            elif hasattr(param, "_local_shard") and param._local_shard is not None:
                param._local_shard = param._local_shard.to(device_id, non_blocking=True)

            if param.data is not None:
                param.data = param.data.to(device_id, non_blocking=True)

            if load_grad and param.grad is not None:
                param.grad = param.grad.to(device_id, non_blocking=True)
        clear_memory()

    def offload_fsdp_optimizer(self):
        if not self.optimizer.state:
            return
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                state = self.optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to("cpu", non_blocking=True)
        clear_memory()

    def load_fsdp_optimizer(self, device_id):
        if not self.optimizer.state:
            return
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                state = self.optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device_id, non_blocking=True)
        clear_memory()
