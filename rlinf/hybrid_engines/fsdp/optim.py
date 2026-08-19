# Copyright 2026 The RLinf Authors.
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

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch.optim import Optimizer


def validate_fp32_master_adamw_config(
    *,
    strategy: str,
    sharding_strategy: str,
    is_lora: bool,
) -> None:
    """Validate the FSDP configurations exercised by FP32MasterAdamW."""
    strategy = str(strategy).lower()
    sharding_strategy = str(sharding_strategy).lower()
    if (
        strategy != "fsdp"
        or sharding_strategy not in {"no_shard", "full_shard"}
        or not is_lora
    ):
        raise ValueError(
            "use_fp32_master_params currently supports only FSDP1 LoRA training "
            "with fsdp_config.strategy=fsdp and sharding_strategy set to "
            "no_shard or full_shard."
        )


class FP32MasterAdamW(Optimizer):
    """AdamW with FP32 state and master weights for low-precision parameters.

    The optimizer keeps the model parameters in their original dtype so the
    forward path is unchanged. For FP16/BF16 parameters, updates are applied to
    an FP32 master copy and then cast back to the model parameter.
    """

    _FP32_STATE_KEYS = ("fp32_master_param", "exp_avg", "exp_avg_sq")

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @staticmethod
    def _new_fp32_tensor_like(param: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(
            param, dtype=torch.float32, memory_format=torch.preserve_format
        )

    def _initialize_state(self, param: torch.Tensor) -> dict[str, Any]:
        if not param.is_floating_point():
            raise TypeError(
                "FP32MasterAdamW only supports floating-point parameters, got "
                f"{param.dtype}."
            )

        state = self.state[param]
        if state:
            return state

        state["step"] = torch.tensor(0.0)
        state["exp_avg"] = self._new_fp32_tensor_like(param)
        state["exp_avg_sq"] = self._new_fp32_tensor_like(param)
        if param.dtype != torch.float32:
            state["fp32_master_param"] = param.detach().to(torch.float32).clone()
        return state

    @torch.no_grad()
    def step(
        self, closure: Callable[[], torch.Tensor] | None = None
    ) -> torch.Tensor | None:
        """Apply one AdamW update through FP32 master parameters."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError(
                        "FP32MasterAdamW does not support sparse gradients."
                    )

                state = self._initialize_state(param)
                state["step"].add_(1)
                step = int(state["step"].item())

                update_param = state.get("fp32_master_param", param)
                grad_fp32 = grad.detach().to(torch.float32)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                update_param.mul_(1.0 - lr * weight_decay)
                exp_avg.lerp_(grad_fp32, 1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2_sqrt = math.sqrt(1.0 - beta2**step)
                denom = exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(eps)
                update_param.addcdiv_(exp_avg, denom, value=-(lr / bias_correction1))

                if update_param is not param:
                    param.copy_(update_param)

        return loss

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load optimizer state without casting FP32 tensors to parameter dtype."""
        saved_groups = state_dict["param_groups"]
        if len(saved_groups) != len(self.param_groups) or any(
            len(saved_group["params"]) != len(current_group["params"])
            for saved_group, current_group in zip(
                saved_groups, self.param_groups, strict=False
            )
        ):
            raise ValueError(
                "Cannot load FP32MasterAdamW state with a different parameter topology."
            )

        saved_to_current = {
            param_id: param
            for saved_group, current_group in zip(
                saved_groups, self.param_groups, strict=True
            )
            for param_id, param in zip(
                saved_group["params"], current_group["params"], strict=True
            )
        }
        fp32_state: dict[torch.Tensor, dict[str, torch.Tensor]] = {}
        state_without_fp32 = {}
        for param_id, saved_state in state_dict["state"].items():
            stripped_state = dict(saved_state)
            param = saved_to_current.get(param_id)
            if param is not None:
                extracted = {}
                for key in (*self._FP32_STATE_KEYS, "step"):
                    value = stripped_state.pop(key, None)
                    if isinstance(value, torch.Tensor):
                        target_device = param.device if key != "step" else value.device
                        target_dtype = torch.float32 if key != "step" else value.dtype
                        extracted[key] = (
                            value.detach()
                            .to(device=target_device, dtype=target_dtype)
                            .clone()
                        )
                if extracted:
                    fp32_state[param] = extracted
            state_without_fp32[param_id] = stripped_state

        state_for_super = dict(state_dict)
        state_for_super["state"] = state_without_fp32
        super().load_state_dict(state_for_super)
        for param, saved_state in fp32_state.items():
            self.state[param].update(saved_state)


def build_adamw(
    params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
    *,
    eps: float,
    weight_decay: float,
    use_fp32_master_params: bool = False,
) -> Optimizer:
    """Build the configured AdamW implementation."""
    optimizer_cls = FP32MasterAdamW if use_fp32_master_params else torch.optim.AdamW
    return optimizer_cls(params, eps=eps, weight_decay=weight_decay)
