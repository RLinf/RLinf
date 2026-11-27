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
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import Optimizer


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


def validate_adam_mixed_precision(
    model: torch.nn.Module,
    *,
    use_fp32_master_params: bool = False,
    optimizer_name: str = "AdamW",
) -> None:
    """Check the dtype FSDP prepares for native Adam or AdamW updates."""
    if use_fp32_master_params:
        return
    for path, module in model.named_modules():
        if not isinstance(module, FSDP):
            continue
        handle = module._handle
        if handle is None or not handle._keep_low_precision_grads:
            continue
        param = handle.flat_param
        if not param.requires_grad:
            continue
        # Match FlatParamHandle.prepare_gradient_for_optim (PyTorch 2.4+).
        grad_dtype = (
            handle._fwd_bwd_param_dtype
            if handle.uses_sharded_strategy or handle._offload_params
            else handle._reduce_dtype
        )
        if grad_dtype == param.dtype:
            continue
        scope = path.replace("_fsdp_wrapped_module.", "") or "<root>"
        remedy = "Set keep_low_precision_grads=false in this wrapper's MP policy."
        if optimizer_name == "AdamW":
            remedy += (
                " For FSDP1 NO_SHARD, select "
                "actor.optim.use_fp32_master_params=true to retain these gradients."
            )
        raise ValueError(
            f"FSDP wrapper {scope!r} has {param.dtype} parameters and prepares "
            f"{grad_dtype} gradients. Native {optimizer_name} requires matching dtypes. "
            + remedy
        )


def build_adamw(
    params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
    *,
    eps: float,
    weight_decay: float,
    use_fp32_master_params: bool = False,
    fused: bool = False,
) -> Optimizer:
    """Build the configured AdamW implementation.

    ``fused`` is ignored for ``FP32MasterAdamW``. Native AdamW first tries
    ``fused=`` and falls back if the build or parameter set does not support it.
    """
    if use_fp32_master_params:
        return FP32MasterAdamW(params, eps=eps, weight_decay=weight_decay)

    try:
        return torch.optim.AdamW(
            params, eps=eps, weight_decay=weight_decay, fused=fused
        )
    except (RuntimeError, TypeError):
        return torch.optim.AdamW(params, eps=eps, weight_decay=weight_decay)


class StreamedAdamW:
    """Host-resident FP32 AdamW streamed through GPU staging slots in fixed chunks.

    Mirrors PEAR's ``StreamedAdamW`` (``pear/trainer/engine.py``): optimizer
    moments—and FP32 master copies for low-precision parameters—stay in pinned
    host memory and are pipelined across H2D / compute / D2H streams in
    ``CHUNK_BYTES`` chunks so 27B updates run in FP32 without spiking GPU
    memory or hitting BF16/FP32 dtype mismatches in ``_multi_tensor_adam``.
    """

    CHUNK_BYTES = 2 << 30
    SLOTS = 3

    def __init__(self, optimizer: Optimizer, *, chunk_bytes: int | None = None):
        self.optimizer = optimizer
        if chunk_bytes is not None and chunk_bytes > 0:
            self.CHUNK_BYTES = int(chunk_bytes)
        self._cuda_available = torch.cuda.is_available()
        self.h2d = torch.cuda.Stream() if self._cuda_available else None
        self.d2h = torch.cuda.Stream() if self._cuda_available else None
        for group in optimizer.param_groups:
            if group.get("fused") or group.get("foreach") is False:
                raise ValueError(
                    "StreamedAdamW requires multi-tensor AdamW (fused=False, foreach!=False)."
                )
            for p in group["params"]:
                if p.requires_grad and p not in optimizer.state:
                    state: dict[str, Any] = {
                        "step": torch.tensor(0.0, dtype=torch.float32),
                        "exp_avg": self.zeros(p),
                        "exp_avg_sq": self.zeros(p),
                    }
                    if self.local(p).dtype != torch.float32:
                        state["fp32_master_param"] = self.master_copy(p)
                    optimizer.state[p] = state

    def is_ready(self) -> bool:
        """Whether every active parameter's optimizer state resides on CPU."""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.optimizer.state.get(p)
                if not state:
                    return False
                if (
                    self.local(state["exp_avg"]).device.type != "cpu"
                    or self.local(state["exp_avg_sq"]).device.type != "cpu"
                    or state["step"].device.type != "cpu"
                ):
                    return False
                master = state.get("fp32_master_param")
                if master is not None and self.local(master).device.type != "cpu":
                    return False
        return True

    @torch.no_grad()
    def step(self) -> None:
        from torch.optim.adam import _multi_tensor_adam

        for group in self.optimizer.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            if not self._cuda_available or self.local(params[0]).device.type != "cuda":
                self._step_cpu_fallback(group, params)
                continue
            self._step_cuda_group(group, params, _multi_tensor_adam)

    def _step_cpu_fallback(
        self, group: dict[str, Any], params: list[torch.Tensor]
    ) -> None:
        from torch.optim.adam import _multi_tensor_adam

        beta1, beta2 = group["betas"]
        update_params: list[torch.Tensor] = []
        grads: list[torch.Tensor] = []
        masters: list[tuple[torch.Tensor, torch.Tensor]] = []
        for p in params:
            local_p = self.local(p)
            local_g = self.local(p.grad)
            state = self.optimizer.state[p]
            master = state.get("fp32_master_param")
            if master is not None:
                local_master = self.local(master)
                update_params.append(local_master)
                masters.append((local_p, local_master))
            else:
                update_params.append(local_p)
            grads.append(
                local_g
                if local_g.dtype == torch.float32
                else local_g.detach().to(dtype=torch.float32)
            )
        _multi_tensor_adam(
            update_params,
            grads,
            [self.local(self.optimizer.state[p]["exp_avg"]) for p in params],
            [self.local(self.optimizer.state[p]["exp_avg_sq"]) for p in params],
            [],
            [self.optimizer.state[p]["step"] for p in params],
            None,
            None,
            amsgrad=group.get("amsgrad", False),
            has_complex=False,
            beta1=beta1,
            beta2=beta2,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            eps=group["eps"],
            maximize=group.get("maximize", False),
            capturable=False,
            differentiable=False,
            decoupled_weight_decay=True,
        )
        for local_p, local_master in masters:
            local_p.copy_(local_master)

    def _step_cuda_group(
        self,
        group: dict[str, Any],
        params: list[torch.Tensor],
        multi_tensor_adam: Callable,
    ) -> None:
        chunks = self.chunks(params)
        offsets = [self.offsets(chunk) for chunk in chunks]
        compute = torch.cuda.current_stream()
        capacity = max((total for _, total in offsets), default=0)
        buffers = [
            torch.empty(capacity, device="cuda", dtype=torch.float32)
            for _ in range(min(self.SLOTS, len(chunks)))
        ]
        freed: list[torch.cuda.Event | None] = [None] * len(buffers)
        self.h2d.wait_stream(compute)

        def load(i: int):
            slot = i % len(buffers)
            with torch.cuda.stream(self.h2d):
                if freed[slot] is not None:
                    self.h2d.wait_event(freed[slot])
                staged = []
                for p, (offset, n, has_master) in zip(
                    chunks[i], offsets[i][0], strict=True
                ):
                    state = self.optimizer.state[p]
                    exp_avg = self.local(state["exp_avg"])
                    exp_avg_sq = self.local(state["exp_avg_sq"])
                    m = buffers[slot][offset : offset + n].view_as(exp_avg)
                    v = buffers[slot][offset + n : offset + 2 * n].view_as(exp_avg_sq)
                    m.copy_(exp_avg, non_blocking=True)
                    v.copy_(exp_avg_sq, non_blocking=True)
                    master = None
                    if has_master:
                        fp32_master = self.local(state["fp32_master_param"])
                        master = buffers[slot][offset + 2 * n : offset + 3 * n].view_as(
                            fp32_master
                        )
                        master.copy_(fp32_master, non_blocking=True)
                    staged.append((m, v, master))
                loaded = torch.cuda.Event()
                loaded.record(self.h2d)
            return staged, loaded

        beta1, beta2 = group["betas"]
        pending = [load(i) for i in range(len(buffers) - 1)]
        for i, chunk in enumerate(chunks):
            if i + len(buffers) - 1 < len(chunks):
                pending.append(load(i + len(buffers) - 1))
            staged, loaded = pending.pop(0)
            compute.wait_event(loaded)
            update_params = [
                master if master is not None else self.local(p)
                for p, (_, _, master) in zip(chunk, staged, strict=True)
            ]
            grads = [
                self.local(p.grad)
                if self.local(p.grad).dtype == torch.float32
                else self.local(p.grad).to(dtype=torch.float32)
                for p in chunk
            ]
            multi_tensor_adam(
                update_params,
                grads,
                [m for m, _, _ in staged],
                [v for _, v, _ in staged],
                [],
                [self.optimizer.state[p]["step"] for p in chunk],
                None,
                None,
                amsgrad=group.get("amsgrad", False),
                has_complex=False,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group.get("maximize", False),
                capturable=False,
                differentiable=False,
                decoupled_weight_decay=True,
            )
            for p, (_, _, master) in zip(chunk, staged, strict=True):
                if master is not None:
                    self.local(p).copy_(master)
            updated = torch.cuda.Event()
            updated.record(compute)
            with torch.cuda.stream(self.d2h):
                self.d2h.wait_event(updated)
                for p, (m, v, master) in zip(chunk, staged, strict=True):
                    state = self.optimizer.state[p]
                    self.local(state["exp_avg"]).copy_(m, non_blocking=True)
                    self.local(state["exp_avg_sq"]).copy_(v, non_blocking=True)
                    if master is not None:
                        self.local(state["fp32_master_param"]).copy_(
                            master, non_blocking=True
                        )
                written = torch.cuda.Event()
                written.record(self.d2h)
            freed[i % len(buffers)] = written

        compute.wait_stream(self.d2h)
        self.d2h.synchronize()

    def chunks(self, params: list[torch.Tensor]) -> list[list[torch.Tensor]]:
        chunks: list[list[torch.Tensor]] = []
        current: list[torch.Tensor] = []
        size = 0
        for p in params:
            state = self.optimizer.state[p]
            tensors_count = 3 if "fp32_master_param" in state else 2
            current.append(p)
            size += tensors_count * self.local(state["exp_avg"]).numel() * 4
            if size >= self.CHUNK_BYTES:
                chunks.append(current)
                current, size = [], 0
        if current:
            chunks.append(current)
        return chunks

    def offsets(
        self, chunk: list[torch.Tensor]
    ) -> tuple[list[tuple[int, int, bool]], int]:
        offsets: list[tuple[int, int, bool]] = []
        total = 0
        for p in chunk:
            state = self.optimizer.state[p]
            n = self.local(state["exp_avg"]).numel()
            has_master = "fp32_master_param" in state
            offsets.append((total, n, has_master))
            total += (3 if has_master else 2) * n
        return offsets, total

    @classmethod
    def zeros(cls, param: torch.Tensor) -> torch.Tensor:
        from torch.distributed.tensor import DTensor

        local = cls.local(param)
        pin = torch.cuda.is_available()
        zeros = torch.zeros(local.shape, dtype=torch.float32, pin_memory=pin)
        return (
            DTensor(zeros, param._spec, requires_grad=False)
            if isinstance(param, DTensor)
            else zeros
        )

    @classmethod
    def master_copy(cls, param: torch.Tensor) -> torch.Tensor:
        from torch.distributed.tensor import DTensor

        local = cls.local(param)
        master = local.detach().to(device="cpu", dtype=torch.float32).clone()
        if torch.cuda.is_available():
            master = master.pin_memory()
        return (
            DTensor(master, param._spec, requires_grad=False)
            if isinstance(param, DTensor)
            else master
        )

    @staticmethod
    def local(tensor: torch.Tensor) -> torch.Tensor:
        from torch.distributed.tensor import DTensor

        return tensor.to_local() if isinstance(tensor, DTensor) else tensor
