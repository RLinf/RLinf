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
import random

import numpy as np
import torch
import torch.optim as optim
from omegaconf import DictConfig
from packaging import version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    MixedPrecision,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM

from rlinf.config import torch_dtype_from_precision
from rlinf.hybrid_engines.fsdp.utils import (
    apply_fsdp2_to_model,
    clip_grad_by_total_norm_,
    create_device_mesh,
    fsdp_version,
    get_fsdp_full_state_dict,
    get_fsdp_state_ctx,
    get_fsdp_wrap_policy,
    get_grad_norm,
    get_lr_scheduler,
    init_fn,
)
from rlinf.utils.utils import clear_memory

if version.parse(torch.__version__) >= version.parse("2.6"):
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
elif version.parse(torch.__version__) >= version.parse("2.4"):
    from torch.distributed._composable.fsdp import (
        CPUOffloadPolicy,
        MixedPrecisionPolicy,
    )
else:
    MixedPrecisionPolicy, CPUOffloadPolicy = None, None


class FSDPModelManager:
    """
    FSDP Model Manager for RL training
    """

    def __init__(self, cfg: DictConfig, world_size: int, logger=None):
        self._cfg = cfg
        self._logger = logger

        mixed_precision_config = self._cfg.fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            self.param_dtype = torch_dtype_from_precision(
                mixed_precision_config.get("param_dtype", "bf16")
            )
            self.reduce_dtype = torch_dtype_from_precision(
                mixed_precision_config.get("reduce_dtype", "fp32")
            )
            self.buffer_dtype = torch_dtype_from_precision(
                mixed_precision_config.get("buffer_dtype", "fp32")
            )
        else:
            self.param_dtype = torch.bfloat16
            self.reduce_dtype = torch.float32
            self.buffer_dtype = torch.float32

        if self._cfg.model.get("precision"):
            self.param_dtype = torch_dtype_from_precision(self._cfg.model.precision)

        self.device_mesh = create_device_mesh(
            world_size, self._cfg.fsdp_config.get("fsdp_size", -1)
        )
        self.dp_group = (
            self.device_mesh["ddp"].get_group()
            if "ddp" in self.device_mesh.mesh_dim_names
            else None
        )

        self.world_size = torch.distributed.get_world_size()
        self.rank = torch.distributed.get_rank()

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
                torch_dtype=self.param_dtype,
                device_map=self._cfg.model.get("device_map", "auto"),
                trust_remote_code=True,
                use_safetensors=self._cfg.model.get("use_safetensors", False),
            )
            if torch.cuda.is_available():
                model = model.cuda()
            if self.param_dtype == torch.float16:
                model = model.half()

        return model

    def setup_model_and_optimizer(self):
        """Setup model and optimizer."""
        module = self.model_provider_func()

        module.gradient_checkpointing_enable()

        mixed_precision = MixedPrecision(
            param_dtype=self.param_dtype,
            reduce_dtype=self.reduce_dtype,
            buffer_dtype=self.buffer_dtype,
        )

        if self._cfg.fsdp_config.sharding_strategy == "full_shard":
            sharding_strategy = ShardingStrategy.FULL_SHARD
        elif self._cfg.fsdp_config.sharding_strategy == "shard_grad_op":
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        elif self._cfg.fsdp_config.sharding_strategy == "hybrid_shard":
            sharding_strategy = ShardingStrategy.HYBRID_SHARD
        else:
            sharding_strategy = ShardingStrategy.NO_SHARD

        betas = (self._cfg.optim.adam_beta1, self._cfg.optim.adam_beta2)

        fsdp_strategy = self._cfg.fsdp_config.get("strategy", "fsdp")
        if fsdp_strategy == "fsdp":
            auto_wrap_policy = get_fsdp_wrap_policy(
                module=module, config=None, is_lora=self._cfg.model.is_lora
            )
            self.model = FSDP(
                module,
                param_init_fn=init_fn,
                auto_wrap_policy=auto_wrap_policy,
                device_id=int(os.environ["LOCAL_RANK"]),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=self._cfg.fsdp_config.get("forward_prefetch", False),
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, (
                "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            )
            mp_policy = MixedPrecisionPolicy(
                param_dtype=self.param_dtype,
                reduce_dtype=self.reduce_dtype,
                cast_forward_inputs=True,
            )
            offload_policy = (
                CPUOffloadPolicy(pin_memory=False)
                if self._cfg.fsdp_config.get("cpu_offload", False)
                else None
            )

            self.model = apply_fsdp2_to_model(
                module=module,
                config=self._cfg.fsdp_config,
                device_mesh=self.device_mesh,
                mp_policy=mp_policy,
                offload_policy=offload_policy,
                reshard_after_forward=self._cfg.fsdp_config.get(
                    "reshard_after_forward", True
                ),
            )

            self.model = module
        else:
            raise NotImplementedError(f"Not implement {fsdp_strategy}")

        # NOTE: Currently we assume that only the value head contains "value_head" in its name.
        # The value head only serves for value prediction in RL algorithms like PPO.
        param_groups = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if "value_head" not in n and p.requires_grad
                ],
                "lr": self._cfg.optim.lr,
                "betas": betas,
            },
        ]

        if self._cfg.model.get("vh_mode", None) in ["a", "a0", "a6"]:
            param_groups.append(
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if "value_head" in n and p.requires_grad
                    ],
                    "lr": self._cfg.optim.value_lr,
                    "betas": betas,
                }
            )

        self.optimizer = optim.AdamW(param_groups)

        total_steps = self._cfg.optim.get("total_training_steps", 0)
        num_warmup_steps = int(self._cfg.optim.get("lr_warmup_steps", -1))
        warmup_style = self._cfg.optim.get("warmup_style", "constant")
        min_lr_ratio = self._cfg.optim.get("min_lr_ratio", 0.0)
        num_cycles = self._cfg.optim.get("num_cycles", 0.5)
        if num_warmup_steps < 0:
            num_warmup_steps_ratio = self._cfg.optim.get("lr_warmup_steps_ratio", 0.0)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        if self.rank == 0:
            self._logger.info(
                f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}"
            )

        self.lr_scheduler = get_lr_scheduler(
            warmup_style=warmup_style,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
            min_lr_ratio=min_lr_ratio,
            num_cycles=num_cycles,
        )

    def optimizer_step(self):
        assert self._cfg.optim.clip_grad is not None

        if fsdp_version(self.model) == 1:
            grad_norm = self.model.clip_grad_norm_(max_norm=self._cfg.optim.clip_grad)
        else:
            grad_norm = get_grad_norm(
                self.model.parameters(),
                dp_group=self.dp_group,
                dtype=torch.float32,
            )
            if self._cfg.optim.clip_grad is not None:
                clip_grad_by_total_norm_(
                    self.model.parameters(),
                    max_grad_norm=self._cfg.optim.clip_grad,
                    total_norm=grad_norm,
                    dtype=torch.float32,
                )
            grad_norm = torch.tensor([grad_norm])

        # if grad_norm is not finite, skip the update
        success = 1
        if not torch.isfinite(grad_norm):
            self._logger.warning(
                f"[Rank {torch.distributed.get_rank()}] grad_norm is not finite: {grad_norm}"
            )
            self.optimizer.zero_grad()
            success = 0
        else:
            self.optimizer.step()

        # TODO: lr scheduler step after all rollout batches are processed
        lr = self.lr_scheduler.get_last_lr()

        self.lr_scheduler.step()

        return success, grad_norm.item(), lr

    def get_rng_state(self):
        rng_state = {
            "cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state()
        return rng_state

    def load_rng_state(self, rng_state):
        torch.set_rng_state(rng_state["cpu"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["random"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(rng_state["cuda"])

    def get_model_state_dict(self):
        state_dict = get_fsdp_full_state_dict(
            self.model, offload_to_cpu=True, rank0_only=False
        )
        return state_dict

    def load_checkpoint(self, load_path: str):
        """
        Load checkpoint from local path.
        """
        is_cuda_available = torch.cuda.is_available()
        if next(self.model.parameters()).is_cpu and is_cuda_available:
            self.load_fsdp_param_and_grad(torch.cuda.current_device())
            self.load_fsdp_optimizer(torch.cuda.current_device())

        state_dict_cfg = ShardedStateDictConfig(
            offload_to_cpu=True if is_cuda_available else False
        )
        optim_cfg = ShardedOptimStateDictConfig(
            offload_to_cpu=True if is_cuda_available else False
        )

        with get_fsdp_state_ctx(
            self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg
        ):
            model_path = os.path.join(
                load_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt"
            )
            model_state_dict = torch.load(model_path, weights_only=False)
            self.model.load_state_dict(model_state_dict)
            if self.rank == 0:
                self._logger.info(f"Loaded model from {model_path}")

            optim_path = os.path.join(
                load_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt"
            )
            optimizer_state_dict = torch.load(optim_path, weights_only=False)
            self.optimizer.load_state_dict(optimizer_state_dict)
            if self.rank == 0:
                self._logger.info(f"Loaded optimizer from {optim_path}")

            extra_state_path = os.path.join(
                load_path,
                f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt",
            )
            extra_state_dict = torch.load(extra_state_path, weights_only=False)
            if "rng" in extra_state_dict:
                self.load_rng_state(extra_state_dict["rng"])
                if self.rank == 0:
                    self._logger.info(f"Loaded rng from {extra_state_path}")

            lr_scheduler_state_dict = extra_state_dict["lr_scheduler"]
            if lr_scheduler_state_dict is not None and self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)
                if self.rank == 0:
                    self._logger.info(f"Loaded lr_scheduler from {extra_state_path}")

        torch.distributed.barrier()

    def save_checkpoint(self, save_path: str):
        """
        Save checkpoint to local path.
        Every rank will save its own model and optim shard.
        """
        is_cuda_available = torch.cuda.is_available()
        if next(self.model.parameters()).is_cpu and is_cuda_available:
            self.load_fsdp_param_and_grad(torch.cuda.current_device())
            self.load_fsdp_optimizer(torch.cuda.current_device())

        state_dict_cfg = ShardedStateDictConfig(
            offload_to_cpu=True if is_cuda_available else False
        )
        optim_cfg = ShardedOptimStateDictConfig(
            offload_to_cpu=True if is_cuda_available else False
        )
        with get_fsdp_state_ctx(
            self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg
        ):
            model_path = os.path.join(
                save_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt"
            )
            optim_path = os.path.join(
                save_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt"
            )
            extra_path = os.path.join(
                save_path,
                f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt",
            )

            model_state_dict = self.model.state_dict()
            torch.save(model_state_dict, model_path)
            if self.rank == 0:
                self._logger.info(f"Saved model to {os.path.abspath(model_path)}")

            optimizer_state_dict = self.optimizer.state_dict()
            torch.save(optimizer_state_dict, optim_path)
            if self.rank == 0:
                self._logger.info(f"Saved optim to {os.path.abspath(optim_path)}")

            lr_scheduler_state_dict = (
                self.lr_scheduler.state_dict()
                if self.lr_scheduler is not None
                else None
            )
            extra_state_dict = {
                "lr_scheduler": lr_scheduler_state_dict,
                "rng": self.get_rng_state(),
            }
            torch.save(extra_state_dict, extra_path)
            if self.rank == 0:
                self._logger.info(f"Saved extra_state to {os.path.abspath(extra_path)}")

        torch.distributed.barrier()

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
        if fsdp_version(self.model) == 2:
            self.model = self.model.to("cpu")
            clear_memory()
            return

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
        if fsdp_version(self.model) == 2:
            self.model = self.model.to("cuda")
            clear_memory()
            return

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
                    if isinstance(value, (DTensor, torch.Tensor)):
                        state[key] = value.to("cpu", non_blocking=True)
        clear_memory()

    def load_fsdp_optimizer(self, device_id):
        if not self.optimizer.state:
            return
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                state = self.optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, (DTensor, torch.Tensor)):
                        state[key] = value.to(device_id, non_blocking=True)
        clear_memory()
