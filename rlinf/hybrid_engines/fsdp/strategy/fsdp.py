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
from contextlib import nullcontext
from typing import ContextManager, Union

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    StateDictType,
)
from torch.optim import Optimizer

from rlinf.hybrid_engines.fsdp import FSDP, CPUOffload, ShardingStrategy
from rlinf.hybrid_engines.fsdp.compat import apply_buffer_dtype_compatibility
from rlinf.hybrid_engines.fsdp.mixed_precision import (
    build_mixed_precision,
    get_mixed_precision_wrap_policy,
)
from rlinf.hybrid_engines.fsdp.strategy.base import FSDPStrategyBase
from rlinf.hybrid_engines.fsdp.utils import (
    FSDPVersion,
    get_backward_prefetch_strategy,
    get_fsdp_wrap_policy,
    get_grad_norm_for_mixed_precision,
    get_sharding_strategy,
    init_fn,
)
from rlinf.scheduler import Worker
from rlinf.utils.utils import clear_memory


class FSDPStrategy(FSDPStrategyBase):
    _FSDP_CACHE_ATTRS = (
        "_mp_shard",
        "_full_param_padded",
        "_full_prec_full_param_padded",
        "_unsharded_flat_param_for_skipped_views",
    )
    _FSDP_GRAD_ATTRS = ("_saved_grad_shard", "_cpu_grad")

    @staticmethod
    def _iter_fsdp_handles(model: FSDP) -> list:
        handles = []
        seen_handle_ids = set()
        for module in model.modules():
            handle = getattr(module, "_handle", None)
            if handle is not None and id(handle) not in seen_handle_ids:
                seen_handle_ids.add(id(handle))
                handles.append(handle)

            all_handles = getattr(module, "_all_handles", None)
            if all_handles is None:
                continue
            for inner_handle in all_handles:
                if inner_handle is None or id(inner_handle) in seen_handle_ids:
                    continue
                seen_handle_ids.add(id(inner_handle))
                handles.append(inner_handle)
        return handles

    @staticmethod
    def _move_tensor(
        tensor: torch.Tensor | None, device: torch.device | str
    ) -> torch.Tensor | None:
        if tensor is None or tensor.device == torch.device(device):
            return tensor
        return tensor.to(device, non_blocking=True)

    @staticmethod
    def _free_tensor_storage(tensor: torch.Tensor | None) -> None:
        if tensor is None:
            return
        try:
            storage = tensor.untyped_storage()
            if storage.size() > 0:
                storage.resize_(0)
        except Exception:
            pass

    @staticmethod
    def _rebind_sharded_tensor_views(handle) -> None:
        flat_param = handle.flat_param
        if not handle.uses_sharded_strategy:
            handle._use_unsharded_views(as_params=False)
            return

        size_0_empty_tensor = torch.empty(
            0,
            dtype=flat_param.dtype,
            device=flat_param.device,
            requires_grad=False,
        )
        for shard_param_info, (param_name, module, _) in zip(
            flat_param._shard_param_infos,
            flat_param._param_infos,
        ):
            if not shard_param_info.in_shard:
                tensor = size_0_empty_tensor
            else:
                offset = shard_param_info.offset_in_shard
                numel_in_shard = shard_param_info.numel_in_shard
                tensor = flat_param[offset : offset + numel_in_shard]
            handle._setattr_tensor(module, param_name, tensor)

        for (
            param_name,
            module,
            _,
            prim_param_name,
            prim_module,
            _,
        ) in flat_param._shared_param_infos:
            handle._setattr_tensor(
                module, param_name, getattr(prim_module, prim_param_name)
            )

    def _rebind_handle_views(self, handle) -> None:
        if handle._use_orig_params:
            handle._use_sharded_views()
            return
        self._rebind_sharded_tensor_views(handle)

    @staticmethod
    def _has_explicit_wrap_config(fsdp_config) -> bool:
        if fsdp_config.get("disable", False):
            return True
        if fsdp_config.get("wrappers"):
            return True
        wrap_policy_cfg = fsdp_config.get("wrap_policy") or {}
        return any(
            wrap_policy_cfg.get(key) is not None
            for key in (
                "transformer_layer_cls_to_wrap",
                "module_classes_to_wrap",
                "no_split_names",
            )
        )

    def _prepare_no_shard_states(
        self,
        model: nn.Module,
        mixed_precision,
        local_rank: int,
    ) -> tuple[list[nn.Parameter] | None, bool]:
        """Exclude frozen parameters and check whether a single root NO_SHARD unit is valid."""
        all_params = list(model.parameters())
        ignored_states: list[nn.Parameter] | None = None
        ignored_set: set[nn.Parameter] = set()

        if any(param.requires_grad for param in all_params):
            frozen_params = [
                param
                for param in all_params
                if not param.requires_grad
                and not param.is_meta
                and (
                    mixed_precision.param_dtype is None
                    or param.dtype == mixed_precision.param_dtype
                )
            ]
            if frozen_params:
                ignored_states = frozen_params
                ignored_set = set(frozen_params)
                if (
                    not self.cfg.fsdp_config.cpu_offload
                    and Worker.torch_platform is not None
                    and Worker.torch_platform.is_available()
                ):
                    target_device = torch.device(
                        f"{Worker.torch_device_type}:{local_rank}"
                    )
                    for param in ignored_states:
                        param.data = self._move_tensor(param.data, target_device)
                if (
                    torch.distributed.is_initialized()
                    and self.world_size > 1
                    and self._dp_group is not None
                ):
                    src_rank = torch.distributed.get_global_rank(self._dp_group, 0)
                    for param in ignored_states:
                        torch.distributed.broadcast(
                            param.data, src=src_rank, group=self._dp_group
                        )

        managed_params = [param for param in all_params if param not in ignored_set]
        floating_dtypes = {
            param.dtype for param in managed_params if param.is_floating_point()
        }
        grad_flags = {param.requires_grad for param in managed_params}
        ignored_module_classes = tuple(mixed_precision._module_classes_to_ignore)
        has_ignored_modules = bool(ignored_module_classes) and any(
            isinstance(module, ignored_module_classes) for module in model.modules()
        )
        can_use_single_root = (
            bool(managed_params)
            and len(floating_dtypes) <= 1
            and (bool(self.cfg.fsdp_config.use_orig_params) or len(grad_flags) <= 1)
            and not has_ignored_modules
        )
        return ignored_states, can_use_single_root

    def wrap_model(self, model: nn.Module, device_mesh: DeviceMesh) -> FSDP:
        """
        Wrap the model with FSDP using the specified configuration,
        it will apply mixed precision, sharding strategy, and wrapping policy.

        Args:
            - model (nn.Module): The model to be wrapped.
            - device_mesh (DeviceMesh): The device mesh for distributed training.

        Returns:
            - FSDP: The wrapped FSDP model.
        """
        mixed_precision_config = self.cfg.fsdp_config.mixed_precision
        mixed_precision = build_mixed_precision(
            {
                **dict(mixed_precision_config),
                "cast_forward_inputs": mixed_precision_config.get(
                    "cast_forward_inputs", True
                ),
                "cast_root_forward_inputs": mixed_precision_config.get(
                    "cast_root_forward_inputs", True
                ),
            }
        )

        sharding_strategy = get_sharding_strategy(
            self.cfg.fsdp_config.sharding_strategy
        )
        local_rank = int(os.environ["LOCAL_RANK"])

        ignored_states = None
        use_single_root_no_shard = False
        if (
            sharding_strategy == ShardingStrategy.NO_SHARD
            and not self._has_explicit_wrap_config(self.cfg.fsdp_config)
        ):
            ignored_states, use_single_root_no_shard = self._prepare_no_shard_states(
                model, mixed_precision, local_rank
            )

        if use_single_root_no_shard:
            auto_wrap_policy = None
        else:
            auto_wrap_policy = get_fsdp_wrap_policy(
                module=model,
                config=self.cfg.fsdp_config,
                is_lora=self.cfg.model.is_lora,
                model_type=self.cfg.model.model_type,
            )
            auto_wrap_policy = get_mixed_precision_wrap_policy(
                model, self.cfg.fsdp_config, auto_wrap_policy, mixed_precision
            )

        backward_prefetch = get_backward_prefetch_strategy(
            self.cfg.fsdp_config.backward_prefetch
        )

        cpu_offload = CPUOffload(offload_params=self.cfg.fsdp_config.cpu_offload)

        fsdp_model = FSDP(
            module=model,
            param_init_fn=init_fn,
            auto_wrap_policy=auto_wrap_policy,
            device_id=local_rank,
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
            sync_module_states=True,
            device_mesh=device_mesh,
            forward_prefetch=self.cfg.fsdp_config.forward_prefetch,
            backward_prefetch=backward_prefetch,
            limit_all_gathers=self.cfg.fsdp_config.limit_all_gathers,
            use_orig_params=self.cfg.fsdp_config.use_orig_params,
            cpu_offload=cpu_offload,
            ignored_states=ignored_states,
        )
        apply_buffer_dtype_compatibility(fsdp_model)
        return fsdp_model

    @classmethod
    def get_fsdp_version(cls) -> FSDPVersion:
        return FSDPVersion.FSDP

    def get_optimizer_state_dict(self, model: FSDP, optimizer: Optimizer) -> dict:
        """
        Get the full state dict of the optimizer.

        Args:
            - model (FSDP): The FSDP wrapped model.
            - optimizer (Optimizer): The optimizer used for training.

        Returns:
            Dict: The full state dict of the optimizer.
        """
        with FSDP.state_dict_type(
            module=model, state_dict_type=StateDictType.FULL_STATE_DICT
        ):
            optimizer_state_dict = FSDP.optim_state_dict(model, optimizer)
        return optimizer_state_dict

    @torch.no_grad()
    def offload_param_and_grad(self, model: FSDP, offload_grad: bool) -> None:
        """
        Offload model parameters and gradients to CPU.
        Args:
            - model (FSDP): The FSDP wrapped model.
            - offload_grad (bool): Whether to offload gradients or not.
        """
        for handle in self._iter_fsdp_handles(model):
            flat_param = handle.flat_param

            if hasattr(flat_param, "_local_shard"):
                flat_param._local_shard = self._move_tensor(
                    flat_param._local_shard, "cpu"
                )
            flat_param.data = self._move_tensor(flat_param.data, "cpu")
            if hasattr(flat_param, "_local_shard") and flat_param.data is not None:
                flat_param._local_shard = flat_param.data

            if offload_grad:
                flat_param.grad = self._move_tensor(flat_param.grad, "cpu")
                for attr_name in self._FSDP_GRAD_ATTRS:
                    if hasattr(flat_param, attr_name):
                        setattr(
                            flat_param,
                            attr_name,
                            self._move_tensor(getattr(flat_param, attr_name), "cpu"),
                        )

            for attr_name in self._FSDP_CACHE_ATTRS:
                if hasattr(flat_param, attr_name):
                    self._free_tensor_storage(getattr(flat_param, attr_name))

            self._rebind_handle_views(handle)

        for _, param in model.named_parameters():
            param.data = self._move_tensor(param.data, "cpu")
            if offload_grad:
                param.grad = self._move_tensor(param.grad, "cpu")

        for _, buffer in model.named_buffers():
            buffer.data = self._move_tensor(buffer.data, "cpu")

        clear_memory()

    @torch.no_grad()
    def onload_param_and_grad(
        self, model: FSDP, device: torch.device, onload_grad: bool
    ) -> None:
        """
        Load model parameters and gradients to the specified device.

        Args:
            - model (FSDP): The FSDP wrapped model.
            - device (torch.device): The device to load the parameters and gradients to.
            - onload_grad (bool): Whether to load gradients or not.

        """
        for handle in self._iter_fsdp_handles(model):
            flat_param = handle.flat_param

            if hasattr(flat_param, "_local_shard"):
                flat_param._local_shard = self._move_tensor(
                    flat_param._local_shard, device
                )
            flat_param.data = self._move_tensor(flat_param.data, device)
            if hasattr(flat_param, "_local_shard") and flat_param.data is not None:
                flat_param._local_shard = flat_param.data

            if onload_grad:
                flat_param.grad = self._move_tensor(flat_param.grad, device)
                for attr_name in self._FSDP_GRAD_ATTRS:
                    if hasattr(flat_param, attr_name):
                        setattr(
                            flat_param,
                            attr_name,
                            self._move_tensor(getattr(flat_param, attr_name), device),
                        )

            self._rebind_handle_views(handle)

        for _, param in model.named_parameters():
            param.data = self._move_tensor(param.data, device)
            if onload_grad:
                param.grad = self._move_tensor(param.grad, device)

        for _, buffer in model.named_buffers():
            buffer.data = self._move_tensor(buffer.data, device)

        clear_memory()

    @torch.no_grad()
    def offload_optimizer(self, optimizer: Optimizer) -> None:
        """
        Offload optimizer state to CPU.

        Args:
            - optimizer (Optimizer): The optimizer used for training.
        """
        if not optimizer.state:
            return
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                state = optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to("cpu", non_blocking=True)
        clear_memory()

    @torch.no_grad()
    def onload_optimizer(self, optimizer: Optimizer, device: torch.device) -> None:
        """
        Load optimizer state to the specified device.

        Args:
            - optimizer (Optimizer): The optimizer used for training.
            - device (torch.device): The device to load the optimizer state to.
        """
        if not optimizer.state:
            return
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                state = optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device, non_blocking=True)
        clear_memory()

    @torch.no_grad()
    def clip_grad_norm_(
        self,
        model: FSDP,
        norm_type: Union[float, int] = 2.0,
    ) -> float:
        """
        Clip the gradients of the model parameters to a maximum norm specified in the configuration.

        Args:
            - model (FSDP): The FSDP wrapped model.
            - norm_type (Union[float, int]): The type of the used p-norm.

        Returns:
            - float: The total norm of the gradients before clipping.
        """
        device = torch.device(f"{Worker.torch_device_type}:{os.environ['LOCAL_RANK']}")
        max_norm = float(self.cfg.optim.clip_grad)
        norm_type = float(norm_type)
        debug_nan_checks = self.cfg.get("debug_nan_checks", False)
        all_handles = getattr(model, "_all_handles", None)
        if all_handles is None:
            raise RuntimeError("Expected FSDP root module with `_all_handles`.")

        all_no_shard = all(not handle.uses_sharded_strategy for handle in all_handles)
        if all_no_shard:
            return (
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type)
                .cpu()
                .item()
            )
        sharded_params_set, nonsharded_params_set = set(), set()
        sharded_params, nonsharded_params = [], []
        grads = []

        for handle in all_handles:
            if handle.uses_sharded_strategy:
                target_set, target_list = sharded_params_set, sharded_params
            else:
                target_set, target_list = nonsharded_params_set, nonsharded_params

            if handle._use_orig_params:
                for p in handle.flat_param._params:
                    if p not in target_set:
                        target_set.add(p)
                        target_list.append(p)
                        if p.grad is not None:
                            grads.append(p.grad)
            else:
                fp = handle.flat_param
                if fp not in target_set:
                    target_set.add(fp)
                    target_list.append(fp)
                    if fp.grad is not None:
                        grads.append(fp.grad)

        # include non-FSDP-managed params (ignored modules etc.)
        for p in model.parameters():
            not_fsdp_managed = (
                p not in sharded_params_set and p not in nonsharded_params_set
            )
            if not_fsdp_managed:
                nonsharded_params_set.add(p)
                nonsharded_params.append(p)
                if p.grad is not None:
                    grads.append(p.grad)
        local_sharded_norm = get_grad_norm_for_mixed_precision(
            sharded_params,
            norm_type,
            torch.tensor(0.0, device=device, dtype=torch.float32),
            device,
        )
        if debug_nan_checks and not torch.isfinite(local_sharded_norm):
            raise RuntimeError(
                "Non-finite local_sharded_norm from "
                "get_grad_norm_for_mixed_precision(sharded_params)."
            )
        local_nonsharded_norm = (
            get_grad_norm_for_mixed_precision(
                nonsharded_params,
                norm_type,
                torch.tensor(0.0, device=device, dtype=torch.float32),
                device,
            )
            if nonsharded_params
            else None
        )
        if (
            debug_nan_checks
            and local_nonsharded_norm is not None
            and not torch.isfinite(local_nonsharded_norm)
        ):
            raise RuntimeError(
                "Non-finite local_nonsharded_norm from "
                "get_grad_norm_for_mixed_precision(nonsharded_params)."
            )

        if norm_type == torch.inf:
            total_norm = (
                torch.maximum(local_sharded_norm, local_nonsharded_norm)
                if local_nonsharded_norm is not None
                else local_sharded_norm
            )
            torch.distributed.all_reduce(
                total_norm, op=torch.distributed.ReduceOp.MAX, group=self._dp_group
            )
        else:
            total_norm = local_sharded_norm**norm_type
            torch.distributed.all_reduce(
                total_norm, op=torch.distributed.ReduceOp.SUM, group=self._dp_group
            )
            if local_nonsharded_norm is not None:
                total_norm += local_nonsharded_norm**norm_type
            total_norm = total_norm ** (1.0 / norm_type)
        if debug_nan_checks and not torch.isfinite(total_norm):
            nonfinite_grad_count = 0
            for grad in grads:
                nonfinite_grad_count += int((~torch.isfinite(grad)).sum().item())
            if nonfinite_grad_count == 0:
                raise RuntimeError(
                    "Non-finite total_norm in clip_grad_norm_ with finite gradients. "
                    "Suspect reduction/power path inside grad norm computation."
                )
            raise RuntimeError(
                "Non-finite total_norm in clip_grad_norm_ and gradients already "
                f"contain non-finite values (count={nonfinite_grad_count})."
            )

        grad_norm = float(total_norm.item())

        # Only apply clipping when the total norm exceeds the maximum allowed norm.
        # This avoids unnecessary scaling and potential numerical issues for very small norms.
        if grad_norm == 0.0 or grad_norm <= max_norm:
            return grad_norm
        clip_coef = max_norm / total_norm
        clip_coef = torch.clamp(clip_coef, max=1.0)
        for g in grads:
            g.mul_(clip_coef.to(device=g.device, dtype=g.dtype))

        return grad_norm

    def before_micro_batch(
        self, model: FSDP, is_last_micro_batch: bool
    ) -> ContextManager:
        """
        Context manager for handling gradient synchronization during micro-batches for FSDP.
        it will disable gradient synchronization for non-last micro-batches to reduce all-reduce count.

        Args:
            - model (FSDP): The FSDP wrapped model.
            - is_last_micro_batch (bool): Whether the current micro-batch is the last one

        Returns:
            - ContextManager: The context manager for gradient synchronization.
        """
        if self.cfg.fsdp_config.enable_gradient_accumulation:
            return model.no_sync() if not is_last_micro_batch else nullcontext()
        return nullcontext()
