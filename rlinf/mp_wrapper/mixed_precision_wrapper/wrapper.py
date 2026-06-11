"""
MixedPrecisionWrapper: dual-storage mixed precision for FSDP/FSDP2 training.

Architecture:
  - FSDP wraps the raw bf16 model FIRST, wrapper wraps FSDP model SECOND.
  - forward() syncs fp32 master -> bf16 compute, then calls inner module.
  - parameters() / named_parameters() return fp32 master params (optimizer sees fp32).

Normal mode (stream_grads_to_cpu=False):
  - bf16 grad -> fp32 -> accumulate into _flat_grad_gpu on GPU.
  - clip_and_flush(): GPU clip + one D2H to CPU + set master.grad views.

Streaming mode (stream_grads_to_cpu=True):
  - NO _flat_grad_gpu (saves GPU memory).
  - bf16 grads async-copied to CPU via pipeline (one reusable staging buffer).
  - clip_and_flush(): sync last D2H, convert bf16->fp32, clip on CPU.
"""

from __future__ import annotations

import functools
import logging
from typing import Iterator, Mapping

import torch
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)


class MixedPrecisionWrapper(nn.Module):

    def __init__(
        self,
        module: nn.Module,
        compute_dtype: torch.dtype = torch.bfloat16,
        keep_master_weights_on_cpu: bool = True,
        stream_grads_to_cpu: bool = False,
    ):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.keep_master_weights_on_cpu = keep_master_weights_on_cpu
        self.stream_grads_to_cpu = stream_grads_to_cpu

        self._compute_params: dict[str, nn.Parameter] = {}
        self._master_params: dict[str, torch.Tensor] = {}
        self._master_param_path: dict[torch.Tensor, str] = {}
        self._dirty: dict[str, bool] = {}

        self._flat_grad_gpu: torch.Tensor | None = None
        self._flat_grad_cpu: torch.Tensor | None = None
        self._grad_offsets: dict[str, tuple[int, int]] = {}

        # Pipeline D2H state (streaming mode only)
        self._d2h_stream: torch.cuda.Stream | None = None
        self._d2h_staging: torch.Tensor | None = None
        self._d2h_pending: tuple[int, int, torch.Size] | None = None

        self._master_device = torch.device("cpu") if keep_master_weights_on_cpu else torch.device("cuda")

        self._setup_dual_storage()
        self._managed_paths: list[str] = list(self._master_params.keys())

    # ------------------------------------------------------------------
    # Master -> compute sync
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sync_master_to_compute(self, force: bool = False) -> int:
        with torch.cuda.stream(self._d2h_stream):
            for path in self._managed_paths:
                if not force and not self._dirty[path]:
                    continue
                master = self._master_params[path]
                compute = self._compute_params[path]
                src = master.data.to(device=compute.device, dtype=self.compute_dtype, non_blocking=True)
                if hasattr(compute.data, "to_local"):
                    compute.data.to_local().copy_(src)
                else:
                    compute.data.copy_(src)
                self._dirty[path] = False
        self._d2h_stream.synchronize()

    # ------------------------------------------------------------------
    # clip_and_flush
    # ------------------------------------------------------------------

    @torch.no_grad()
    def clip_grad_norm_(self, max_norm: float) -> torch.Tensor:
        # Streaming mode: sync + convert last pending D2H, clip on CPU
        self._convert_pipeline_pending()

        # Clip
        if not self.stream_grads_to_cpu:
            # Normal mode: clip on GPU, one D2H to CPU
            grad_norm = self._clip_grads(self._flat_grad_gpu, max_norm)
            if self.keep_master_weights_on_cpu:
                self._flat_grad_cpu.copy_(self._flat_grad_gpu)
        else:
            grad_norm = self._clip_grads(self._flat_grad_cpu, max_norm)

        # Set master grad view
        grad_buf = self._flat_grad_cpu if self.keep_master_weights_on_cpu else self._flat_grad_gpu
        assert grad_buf is not None
        for path in self._managed_paths:
            offset, numel = self._grad_offsets[path]
            master = self._master_params[path]
            master.grad = grad_buf[offset:offset + numel].view(master.shape)

        return grad_norm

    @torch.no_grad()
    def _clip_grads(self, flat_grad, max_norm: float) -> torch.Tensor:
        total_norm = flat_grad.norm()
        # add dist.allreduce(sum)
        if total_norm.item() > max_norm:
            flat_grad.mul_(max_norm / total_norm.item())
        return total_norm

    def on_optimzer_pre_step(self):
        for path in self._managed_paths:
            self._dirty[path] = True

    @torch.no_grad()
    def on_optimzer_post_step_update(self, master: torch.Tensor, path: str | None=None):
        if path is None:
            path = self._master_param_path[master]
        with torch.cuda.stream(self._d2h_stream):
            compute = self._compute_params[path]
            src = master.data.to(device=compute.device, dtype=self.compute_dtype, non_blocking=True)
            compute_data = compute.data
            if hasattr(compute_data, "to_local"):
                compute_data = compute_data.to_local()
            compute_data.copy_(src)
            self._dirty[path] = False

    @torch.no_grad()
    def on_optimizer_pre_zero_grad(self):
        with torch.cuda.stream(self._d2h_stream):
            for path in self._managed_paths:
                if not self._dirty[path]:
                    continue
                master = self._master_params[path]
                compute = self._compute_params[path]
                src = master.data.to(device=compute.device, dtype=self.compute_dtype, non_blocking=True)
                compute_data = compute.data
                if hasattr(compute_data, "to_local"):
                    compute_data = compute_data.to_local()
                compute_data.copy_(src)
                self._dirty[path] = False

    @torch.no_grad()
    def on_optimizer_post_zero_grad(self):
        if self._flat_grad_cpu is not None:
            # overlap cpu -> gpu copy with cpu tensor zero
            self._flat_grad_cpu.zero_()
        if self._flat_grad_gpu is not None:
            self._flat_grad_gpu.zero_()
        self._d2h_stream.synchronize()

    # ------------------------------------------------------------------
    # Pipeline D2H helpers (streaming mode)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _convert_pipeline_pending(self):
        if self._d2h_pending is None:
            return

        (offset, numel, shape), self._d2h_pending = self._d2h_pending, None
        self._d2h_stream.synchronize()
        view = self._flat_grad_cpu[offset:offset + numel]
        view = view.view(shape)
        view.add_(self._d2h_staging[:numel].view(shape))

    @torch.no_grad()
    def _launch_pipeline_d2h(self, src: torch.Tensor, offset: int, numel: int):
        if self._d2h_staging is None or self._d2h_staging.numel() < src.numel():
            self._d2h_staging = torch.empty(src.numel(), dtype=torch.float32, device="cpu", pin_memory=True)
        event = torch.cuda.current_stream().record_event()
        self._d2h_stream.wait_event(event)
        with torch.cuda.stream(self._d2h_stream):
            dst = self._d2h_staging[:numel].view(src.shape)
            dst.copy_(src.to(dtype=torch.float32), non_blocking=True)
        self._d2h_pending = (offset, numel, src.shape)

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------

    def _setup_dual_storage(self):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP1
        is_fsdp1 = isinstance(self.module, FSDP1)

        for path, param in self.module.named_parameters():
            if not param.requires_grad:
                continue
            self._compute_params[path] = param

            data = param.data
            if hasattr(data, "to_local"):
                data = data.to_local()
            master = data.to(device=self._master_device, dtype=torch.float32).clone()
            if self._master_device.type == "cpu":
                master = master.pin_memory()
            master.requires_grad_(True)
            self._master_params[path] = master
            self._master_param_path[master] = path
            self._dirty[path] = True

            if not is_fsdp1:
                param.register_post_accumulate_grad_hook(
                    functools.partial(self._per_param_grad_hook, path=path)
                )

        if is_fsdp1:
            # self.module.register_forward_pre_hook(self._full_backward_hook)
            self.module.register_full_backward_hook(self._full_backward_hook)

        self._setup_flat_grad_buffers()

        hook_type = "full_backward_hook" if is_fsdp1 else "post_accumulate_grad_hook"
        logger.info(
            f"MixedPrecisionWrapper: {len(self._master_params)} managed params, "
            f"compute_dtype={self.compute_dtype}, master_on_cpu={self.keep_master_weights_on_cpu}, "
            f"stream_grads={self.stream_grads_to_cpu}, hook={hook_type}"
        )

    def _setup_flat_grad_buffers(self):
        managed_paths = list(self._master_params.keys())
        if not managed_paths:
            return

        total_numel = 0
        for path in managed_paths:
            master = self._master_params[path]
            self._grad_offsets[path] = (total_numel, master.numel())
            total_numel += master.numel()

        fp32_gb = total_numel * 4 / (1024 ** 3)

        self._flat_grad_cpu = torch.zeros(total_numel, dtype=torch.float32, device="cpu", pin_memory=True)

        if self.stream_grads_to_cpu:
            logger.info(f"Flat grad CPU buffer: {fp32_gb:.2f} GiB, NO GPU buffer")
        else:
            self._flat_grad_gpu = torch.zeros(total_numel, dtype=torch.float32, device="cuda")
            logger.info(f"Flat grad GPU buffer: {fp32_gb:.2f} GiB, CPU buffer: {fp32_gb:.2f} GiB")

        self._d2h_stream = torch.cuda.Stream()

    # ------------------------------------------------------------------
    # Forward / Backward hooks
    # ------------------------------------------------------------------

    def forward(self, *args, **kwargs):
        self.sync_master_to_compute()
        return self.module(*args, **kwargs)


    def _per_param_grad_hook(self, param: nn.Parameter, *, path: str):
        # DDP / FSDP2: param-level hook
        assert param is self._compute_params[path]
        self._accumulate_grad_to_flat(path)


    def _full_backward_hook(self, module, grad_input, grad_output):
        # FSDP1: module-level hook
        for path in self._managed_paths:
            self._accumulate_grad_to_flat(path)


    def _accumulate_grad_to_flat(self, path: str):
        compute = self._compute_params[path]
        compute_grad = compute.grad
        if compute_grad is None:
            return
        local_grad = compute_grad.to_local() if hasattr(compute_grad, "to_local") else compute_grad
        offset, numel = self._grad_offsets[path]

        if self.stream_grads_to_cpu:
            # Pipeline async D2H of bf16 grad to CPU
            self._convert_pipeline_pending()
            self._launch_pipeline_d2h(local_grad, offset, numel)
        else:
            # Accumulate to GPU buffer
            view = self._flat_grad_gpu[offset:offset + numel].view(local_grad.shape)
            view.add_(local_grad.to(device="cuda", dtype=torch.float32))

        compute.grad, local_grad = None, None

    # ------------------------------------------------------------------
    # state_dict helpers
    # ------------------------------------------------------------------

    def _gather_param_if_sharded(
        self, tensor: torch.Tensor, path: str, world_size: int
    ) -> torch.Tensor:
        compute = self._compute_params[path]
        compute_shape = compute.shape if hasattr(compute, 'shape') else compute.data.shape
        if tensor.shape == compute_shape:
            return tensor
        if tensor.shape[0] != compute_shape[0] and world_size > 1:
            dim0_total = compute_shape[0]
            local_size = torch.tensor(tensor.shape[0], device=tensor.device)
            sizes = [torch.empty((), dtype=torch.long, device=tensor.device) for _ in range(world_size)]
            dist.all_gather(sizes, local_size)
            size_list = [s.item() for s in sizes]
            full = torch.zeros(dim0_total, *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)
            shard_list = list(torch.split(full, size_list, dim=0))
            dist.all_gather(shard_list, tensor)
            return full
        return tensor

    @torch.no_grad()
    def state_dict(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        sd: dict[str, torch.Tensor] = {}
        for path in self._managed_paths:
            master = self._master_params[path]
            full = self._gather_param_if_sharded(master, path, world_size)
            sd[path] = full.detach().cpu()
        return sd

    def load_state_dict(self, state_dict: Mapping[str, torch.Tensor], strict: bool = True):
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        for path, tensor in state_dict.items():
            if path not in self._master_params:
                continue
            master = self._master_params[path]
            if master.shape != tensor.shape and world_size > 1:
                dim0 = tensor.shape[0]
                shard_size = (dim0 + world_size - 1) // world_size
                start = rank * shard_size
                end = min(start + shard_size, dim0)
                tensor = tensor[start:end]
            master.data.copy_(tensor.to(device=master.device, dtype=torch.float32))
        for path in self._managed_paths:
            self._dirty[path] = True

    # ------------------------------------------------------------------
    # nn.Module overrides
    # ------------------------------------------------------------------

    def parameters(self, recurse: bool = True) -> Iterator[torch.Tensor]:
        for path in self._managed_paths:
            yield self._master_params[path]

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[tuple[str, torch.Tensor]]:
        for path in self._managed_paths:
            yield prefix + path, self._master_params[path]
