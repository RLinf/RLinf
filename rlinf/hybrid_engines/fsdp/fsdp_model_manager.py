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
from typing import ContextManager, Dict, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig
from packaging import version
from torch.amp.grad_scaler import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq

from rlinf.config import torch_dtype_from_precision
from rlinf.data.tokenizers import hf_tokenizer
from rlinf.hybrid_engines.fsdp import FSDP, FSDPModule
from rlinf.hybrid_engines.fsdp.strategy.base import FSDPStrategyBase
from rlinf.hybrid_engines.fsdp.utils import (
    create_device_mesh,
    get_lr_scheduler,
)
from rlinf.utils.logging import get_logger

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

    def __init__(self, cfg: DictConfig, world_size: int, rank: int) -> None:
        """
        Initialize FSDP Model Manager.

        Assumes:
            - torch.distributed has been initialized outside before calling this constructor.
            - all cfg parameters are validated in `valid_fsdp_config`.

        Params:
            cfg: actor config in yaml file.
            world_size: total number of FSDP actor processes.
        """
        self._cfg = cfg
        self._logger = get_logger()
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

        self._device_mesh = create_device_mesh(
            world_size, self._cfg.fsdp_config.get("fsdp_size", -1)
        )
        self._dp_group = (
            self._device_mesh["ddp"].get_group()
            if "ddp" in self._device_mesh.mesh_dim_names
            else None
        )

        self._strategy = FSDPStrategyBase.create(
            self._cfg, world_size, rank, self._dp_group, self._logger
        )

        self.amp_context = self._create_amp_context()

    def _create_amp_context(self) -> ContextManager:
        """
        Create AMP context manager based on configuration.

        Returns:
            A context manager for automatic mixed precision (AMP) if enabled,
            otherwise a null context manager.
        """
        from contextlib import nullcontext

        if not self._cfg.fsdp_config.amp.enabled:
            self._logger.info("[FSDP] AMP is disabled.")
            return nullcontext()

        precision = torch_dtype_from_precision(self._cfg.fsdp_config.amp.precision)

        self._logger.info(f"[FSDP] AMP is enabled with precision: {precision}.")

        return torch.amp.autocast(device_type="cuda", dtype=precision)

    def model_provider_func(self) -> torch.nn.Module:
        """
        Initialize model used by FSDP actor

        Returns:
            model: the initialized model.
        """
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
        """
        Replace model modules with liger-kernel optimized modules.

        Params:
            model: the model to be optimized.
        """
        if self._cfg.model.get("gptq_model", False) or self._cfg.model.get(
            "load_in_8bit", False
        ):
            self._logger.info(
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
                self._logger.info(
                    f"[FSDP] Applied liger-kernel optimizations for model_arch: {model_arch}, used kwargs: {apply_kwargs}"
                )
            else:
                self._logger.info(
                    f"[FSDP] No liger-kernel optimizations applied for model_arch: {model_arch}"
                )
                return
        except Exception as e:
            self._logger.warning(f"[FSDP] Liger kernels not applied: {e}")

    def setup_model_and_optimizer(self) -> None:
        """Setup model, lr_scheduler, optimizer and grad_scaler."""
        module = self.model_provider_func()

        # Enable gradient checkpointing if configured
        if self._cfg.model.get("gradient_checkpointing", False):
            self._logger.info("[FSDP] Enabling gradient checkpointing")
            module.gradient_checkpointing_enable()
        else:
            self._logger.info("[FSDP] Gradient checkpointing is disabled")

        # build model, optimizer, lr_scheduler, grad_scaler
        self.model = self._strategy.wrap_model(
            model=module, device_mesh=self._device_mesh
        )
        self.optimizer = self.build_optimizer(
            model=self.model, enable_critic_warmup=self.critic_warmup_steps > 0
        )
        self.lr_scheduler = self.build_lr_scheduler(optimizer=self.optimizer)
        self.grad_scaler = self.build_grad_scaler(
            self._cfg.fsdp_config.amp.use_grad_scaler
        )

    def get_rng_state(self) -> Dict:
        """
        Get rng state.

        Returns:
            rng_state: the current rng state.
        """
        return self._strategy.save_rng_state()

    def load_rng_state(self, rng_state: Dict) -> None:
        """
        Load rng state.

        Params:
            rng_state: the rng state to load.
        """
        self._strategy.load_rng_state(rng_state)

    def get_model_state_dict(self) -> Dict:
        """
        Get full model state dict.
        """
        state_dict = self._strategy.get_model_state_dict(self.model)
        return state_dict

    def load_checkpoint(self, load_path: str) -> None:
        """
        Load checkpoint from local path.

        Params:
            load_path: the directory to load checkpoint.
        """
        self._strategy.load_checkpoint(
            self.model, self.optimizer, self.lr_scheduler, load_path
        )

    def save_checkpoint(self, save_path: str, global_steps: int) -> None:
        """
        Save checkpoint to local path.
        Every rank will save its own model and optim shard.

        Params:
            save_path: the directory to save checkpoint.
        """
        self._strategy.save_checkpoint(
            self.model, self.optimizer, self.lr_scheduler, save_path
        )

    def offload_param_and_grad(self, offload_grad: bool = False) -> None:
        """
        Offload FSDP parameters and gradients(options) to CPU.

        Params:
            offload_grad: whether to offload gradients.
        """
        self._strategy.offload_param_and_grad(self.model, offload_grad)

    def load_param_and_grad(self, device_id: int, load_grad: bool = False) -> None:
        """
        Load FSDP parameters and gradients(options) to the specified device.

        Params:
            device_id: the target device id to load parameters and gradients.
            load_grad: whether to load gradients.
        """
        self._strategy.onload_param_and_grad(self.model, device_id, load_grad)

    def offload_optimizer(self) -> None:
        """
        Offload optimizer states to CPU.
        """
        self._strategy.offload_optimizer(self.optimizer)

    def load_optimizer(self, device_id: int) -> None:
        """
        Load optimizer states to the specified device.

        Params:
            device_id: the target device id to load optimizer states.
        """
        self._strategy.onload_optimizer(self.optimizer, device_id)

    def optimizer_step(self) -> tuple[float, list[float]]:
        """
        Perform optimizer step using its optimizer, lr_scheduler and grad_scaler.

        Returns:
            A tuple of (grad_norm, lr_list), lr_list contains learning rates for all param groups.
        """
        self.optimizer_steps += 1
        self.grad_scaler.unscale_(optimizer=self.optimizer)
        grad_norm = self._strategy.clip_grad_norm_(
            model=self.model,
        )

        if not torch.isfinite(torch.as_tensor(grad_norm)):
            self._logger.warning(
                f"[FSDP] Non-finite grad norm {grad_norm} detected. Skipping optimizer step."
            )
        else:
            self.grad_scaler.step(optimizer=self.optimizer)
            self.lr_scheduler.step()

        self.grad_scaler.update()

        if self.critic_warmup_steps > 0:
            lr_list = [0.0 for _ in self.optimizer.param_groups]
            if self.optimizer_steps >= self.critic_warmup_steps:
                self.optimizer = self.build_optimizer(model=self.model)
                self.critic_warmup_steps = 0
        else:
            lr_list = [group["lr"] for group in self.optimizer.param_groups]

        return grad_norm, lr_list

    def build_lr_scheduler(self, optimizer: Optimizer) -> LRScheduler:
        """
        Build the learning rate scheduler based on the configuration.
        Currently only support LambdaLR scheduler with various warmup styles.

        Args:
            optimizer (Optimizer): The optimizer for which to schedule the learning rate.

        Returns:
            LRScheduler: The learning rate scheduler.
        """
        total_steps = self._cfg.optim.get("total_training_steps", 0)
        num_warmup_steps = int(self._cfg.optim.get("lr_warmup_steps", -1))
        warmup_style = self._cfg.optim.get("warmup_style", "constant")
        min_lr_ratio = self._cfg.optim.get("min_lr_ratio", 0.0)
        num_cycles = self._cfg.optim.get("num_cycles", 0.5)
        if num_warmup_steps < 0:
            num_warmup_steps_ratio = self._cfg.optim.get("lr_warmup_steps_ratio", 0.0)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        return get_lr_scheduler(
            warmup_style=warmup_style,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
            min_lr_ratio=min_lr_ratio,
            num_cycles=num_cycles,
        )

    def build_optimizer(
        self,
        model: Union[nn.Module, FSDPModule, FSDP],
        enable_critic_warmup: bool = False,
    ) -> Optimizer:
        """
        Build the optimizer based on the configuration, currently only support Adam optimizer.

        Args:
            model: The model to optimize, can be nn.Module, FSDPModule (used in FSDP2) or FSDP.
            enable_critic_warmup: Whether to enable critic warmup used for value network.

        Returns:
            Optimizer: The constructed optimizer.
        """
        betas = (self._cfg.optim.adam_beta1, self._cfg.optim.adam_beta2)

        params_actor = []
        params_critic = []

        if enable_critic_warmup:
            self._logger.info("[FSDP] Enable critic warmup for value head.")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.store_requires_grad_param_name.append(name)
                    if "value_head" in name or "model.value_head" in name:
                        params_critic.append(param)
                        continue
                    param.requires_grad = False

        else:
            for name, param in model.named_parameters():
                if name in self.store_requires_grad_param_name:
                    param.requires_grad = True
                if param.requires_grad:
                    if "value_head" in name or "model.value_head" in name:
                        params_critic.append(param)
                    else:
                        params_actor.append(param)

        param_groups = []
        if len(params_actor) > 0:
            param_groups.append(
                {
                    "params": params_actor,
                    "lr": self._cfg.optim.lr,
                    "betas": betas,
                }
            )
        if len(params_critic) > 0:
            param_groups.append(
                {
                    "params": params_critic,
                    "lr": self._cfg.optim.value_lr,
                    "betas": betas,
                }
            )
        optimizer = torch.optim.AdamW(
            param_groups,
        )
        return optimizer

    def build_grad_scaler(self, enabled: bool) -> GradScaler:
        """
        Build the gradient scaler based on the configuration.

        Args:
            enabled (bool): Whether to enable gradient scaling.

        Returns:
            GradScaler: The gradient scaler.
        """
        return GradScaler(enabled=enabled)

    def before_micro_batch(
        self, model: Union[FSDP, FSDPModule], is_last_micro_batch: bool
    ) -> ContextManager:
        """
            Setup context manager before processing a micro-batch.
            This is used to control gradient synchronization behavior.
            Depending on the specific FSDP strategy being used, if using
            FSDP, it will return model.no_sync() for non-last micro-batches to
            avoid gradient synchronization, and nullcontext() for the last
            micro-batch to ensure gradients are synchronized and updated.
            If using FSDP2, it will set requires_gradient_sync flag
            on the model accordingly.

        Args:
            model: The FSDP or FSDPModule model.
            is_last_micro_batch: A boolean indicating if this is the last micro-batch.

        Returns:
            A context manager for the micro-batch processing.
        """
        return self._strategy.before_micro_batch(
            model=model, is_last_micro_batch=is_last_micro_batch
        )
