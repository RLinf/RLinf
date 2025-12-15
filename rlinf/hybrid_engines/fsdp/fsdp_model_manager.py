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
from omegaconf import DictConfig
from torch.distributed.fsdp import (
    BackwardPrefetch,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq

from rlinf.config import torch_dtype_from_precision
from rlinf.data.tokenizers import hf_tokenizer
from rlinf.hybrid_engines.fsdp.utils import (
    get_fsdp_wrap_policy,
    init_fn,
)
from rlinf.utils.logging import get_logger
from rlinf.utils.utils import clear_memory


class FSDPModelManager:
    """
    FSDP Model Manager for RL training
    """

    def __init__(self, cfg: DictConfig):
        self._cfg = cfg
        self.logger = get_logger()
        self.torch_dtype = torch_dtype_from_precision(self._cfg.model.precision)

        self.optimizer_steps = 0
        self.critic_warmup_steps = 0
        if self._cfg.optim.get("critic_warmup_steps", None) and self._cfg.model.get(
            "add_value_head", False
        ):
            self.critic_warmup_steps = self._cfg.optim.critic_warmup_steps
        self.store_requires_grad_param_name = []

        if cfg.get("tokenizer", {}).get("tokenizer_model", None) is not None:
            self.tokenizer = hf_tokenizer(cfg.tokenizer.tokenizer_model)

    def model_provider_func(self) -> torch.nn.Module:
        cfg = self._cfg
        use_gptq = cfg.model.get("gptq_model", False)
        load_in_8bit = cfg.model.get("load_in_8bit", False)

        use_triton = cfg.get("use_triton", True)

        assert torch.cuda.is_available(), "CUDA is not available."
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")

        model_config = AutoConfig.from_pretrained(
            cfg.model.model_path,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )

        if use_gptq:
            from auto_gptq import AutoGPTQForCausalLM  # type: ignore[import-not-found]

            model_wrapper = AutoGPTQForCausalLM.from_quantized(
                cfg.model.model_path,
                device=device,
                use_triton=use_triton,
            )
            model = model_wrapper.model
        elif load_in_8bit:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model.model_path,
                config=model_config,
                load_in_8bit=True,
            )
        else:
            if type(model_config) in AutoModelForVision2Seq._model_mapping.keys():
                auto_model_class = AutoModelForVision2Seq
            else:
                auto_model_class = AutoModelForCausalLM

            model = auto_model_class.from_pretrained(
                cfg.model.model_path,
                torch_dtype=self.torch_dtype,
                config=model_config,
                trust_remote_code=True,
            )

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if cfg.fsdp_config.use_liger_kernel:
            self._optimize_with_liger_kernel(model)

        return model

    def _optimize_with_liger_kernel(self, model: torch.nn.Module) -> None:
        if self._cfg.model.get("gptq_model", False) or self._cfg.model.get(
            "load_in_8bit", False
        ):
            self.logger.info(
                "[FSDP] Skip using liger-kernel optimized modules for GPTQ/8bit models."
            )
            return
        try:
            from liger_kernel.transformers import (
                apply_liger_kernel_to_qwen2,
                apply_liger_kernel_to_qwen2_5_vl,
            )

            MODEL_ARCH_APPLY_FUNC = {
                "qwen2.5": (
                    apply_liger_kernel_to_qwen2,
                    {
                        "rope": True,
                        "rms_norm": True,
                        "swiglu": True,
                        "fused_linear_cross_entropy": True,
                    },
                ),
                "qwen2.5-vl": (
                    apply_liger_kernel_to_qwen2_5_vl,
                    {
                        "rope": True,
                        "rms_norm": True,
                        "swiglu": True,
                        "fused_linear_cross_entropy": True,
                    },
                ),
            }
            model_arch = self._cfg.model.get("model_arch", "").lower()
            if model_arch in MODEL_ARCH_APPLY_FUNC:
                apply_func, apply_kwargs = MODEL_ARCH_APPLY_FUNC[model_arch]
                apply_func(
                    model=model,
                    **apply_kwargs,
                )
                self.logger.info(
                    f"[FSDP] Applied liger-kernel optimizations for model_arch: {model_arch}, used kwargs: {apply_kwargs}"
                )
            else:
                self.logger.info(
                    f"[FSDP] No liger-kernel optimizations applied for model_arch: {model_arch}"
                )
                return
        except Exception as e:
            self.logger.warning(f"[FSDP] Liger kernels not applied: {e}")

    def setup_model_and_optimizer(self, initialize_target=False):
        """Setup model and optimizer."""
        module = self.model_provider_func()
        if initialize_target:
            target_module = self.model_provider_func()

        # Enable gradient checkpointing if configured
        if self._cfg.model.get("gradient_checkpointing", False):
            self.logger.info("[FSDP] Enabling gradient checkpointing")
            module.gradient_checkpointing_enable()
            if initialize_target:
                target_module.gradient_checkpointing_enable()
        else:
            self.logger.info("[FSDP] Gradient checkpointing is disabled")

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

        is_vla_model = (
            True
            if self._cfg.model.get("model_name", None) in ["openvla", "openvla_oft"]
            else False
        )

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=module,
            config=None,
            is_lora=self._cfg.model.is_lora,
            is_vla_model=is_vla_model,
        )
        if initialize_target:
            target_auto_wrap_policy = get_fsdp_wrap_policy(
                module=target_module,
                config=None,
                is_lora=self._cfg.model.is_lora,
                is_vla_model=is_vla_model,
            )

        if self._cfg.fsdp_config.backward_prefetch is None:
            backward_prefetch = None
        elif self._cfg.fsdp_config.backward_prefetch == "pre":
            backward_prefetch = BackwardPrefetch.BACKWARD_PRE
        elif self._cfg.fsdp_config.backward_prefetch == "post":
            backward_prefetch = BackwardPrefetch.BACKWARD_POST
        else:
            raise ValueError(
                f"Invalid fsdp_config.backward_prefetch: {self._cfg.fsdp_config.backward_prefetch}"
            )
        self.model = FSDP(
            module,
            param_init_fn=init_fn,
            auto_wrap_policy=auto_wrap_policy,
            device_id=int(os.environ["LOCAL_RANK"]),
            sharding_strategy=sharding_strategy,  # zero3
            mixed_precision=mixed_precision,
            sync_module_states=True,
            forward_prefetch=self._cfg.fsdp_config.forward_prefetch,
            backward_prefetch=backward_prefetch,
            limit_all_gathers=self._cfg.fsdp_config.limit_all_gathers,
            use_orig_params=self._cfg.fsdp_config.use_orig_params,
        )

        if initialize_target:
            self.target_model = FSDP(
                target_module,
                param_init_fn=init_fn,
                auto_wrap_policy=target_auto_wrap_policy,
                device_id=int(os.environ["LOCAL_RANK"]),
                sharding_strategy=sharding_strategy,  # zero3
                mixed_precision=mixed_precision,
                sync_module_states=True,
                forward_prefetch=self._cfg.fsdp_config.forward_prefetch,
                backward_prefetch=backward_prefetch,
                limit_all_gathers=self._cfg.fsdp_config.limit_all_gathers,
                use_orig_params=self._cfg.fsdp_config.use_orig_params,
            )
            self.target_model.requires_grad_(False)
            self.target_model_initialized = True

        self.build_optimizer(enable_warmup=self.critic_warmup_steps > 0)

    def set_max_steps(self, max_steps):
        self.max_steps = max_steps

    def build_optimizer(self, enable_warmup=False):
        assert hasattr(self, "model")

        betas = (self._cfg.optim.adam_beta1, self._cfg.optim.adam_beta2)
        params_actor = []
        params_critic = []
        if enable_warmup:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.store_requires_grad_param_name.append(name)

                    if "value_head" in name or "model.value_head" in name:
                        params_critic.append(param)
                        continue
                    param.requires_grad = False
        else:
            for name, param in self.model.named_parameters():
                if (
                    len(self.store_requires_grad_param_name) > 0
                    and name in self.store_requires_grad_param_name
                ):
                    param.requires_grad = True

                if param.requires_grad:
                    if "value_head" in name or "model.value_head" in name:
                        params_critic.append(param)
                    else:
                        params_actor.append(param)

        param_groups = []
        if len(params_actor) > 0:
            param_groups.append(
                {"params": params_actor, "lr": self._cfg.optim.lr, "betas": betas}
            )
        if len(params_critic) > 0:
            param_groups.append(
                {
                    "params": params_critic,
                    "lr": self._cfg.optim.value_lr,
                    "betas": betas,
                }
            )

        self.optimizer = torch.optim.AdamW(param_groups)
        lr_scheduler_type = self._cfg.optim.get("lr_scheduler_type", "constant")
        if lr_scheduler_type == "constant":
            self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.optimizer, factor=1
            )
        elif lr_scheduler_type == "cosine":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.max_steps, eta_min=1e-6
            )
        else:
            raise NotImplementedError

    def optimizer_step(self):
        grad_norm = self.model.clip_grad_norm_(max_norm=self._cfg.optim.clip_grad)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.optimizer_steps += 1

        lrs = []  # [actor_lr, Optinal(critic_lr)]
        if self.critic_warmup_steps > 0:
            lrs.extend([0.0, self.optimizer.param_groups[0]["lr"]])
            if self.optimizer_steps >= self.critic_warmup_steps:
                self.build_optimizer(enable_warmup=False)
                self.critic_warmup_steps = -1
        else:
            lrs.append(self.optimizer.param_groups[0]["lr"])
            if len(self.optimizer.param_groups) > 1:
                lrs.append(self.optimizer.param_groups[1]["lr"])

        return grad_norm, lrs

    def get_model_state_dict(self):
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
            state_dict = self.model.state_dict()
        return state_dict
    
    def get_target_model_state_dict(self):
        with FSDP.state_dict_type(self.target_model, StateDictType.FULL_STATE_DICT):
            state_dict = self.target_model.state_dict()
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
