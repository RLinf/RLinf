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
import json
import gc
from contextlib import nullcontext
from typing import ContextManager, Union

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.optim import Optimizer

from safetensors.torch import load_file

from rlinf.config import torch_dtype_from_precision
from rlinf.hybrid_engines.fsdp import (
    CPUOffloadPolicy,
    FSDPModule,
    MixedPrecisionPolicy,
    OffloadPolicy,
)
from rlinf.hybrid_engines.fsdp.strategy.base import FSDPStrategyBase
from rlinf.hybrid_engines.fsdp.utils import (
    FSDPVersion,
    apply_fsdp2_to_model,
    clip_grad_by_total_norm_,
    get_grad_norm,
)
from rlinf.utils.utils import clear_memory


class FSDP2Strategy(FSDPStrategyBase):
    def wrap_model(self, model: nn.Module, device_mesh: DeviceMesh) -> FSDPModule:
        """
        Wrap the model with FSDP2's fully_shard.

        Args:
            - model (nn.Module): The model to be wrapped.
            - device_mesh (DeviceMesh): The device mesh for FSDP2.

        Returns:
            - FSDPModule: The FSDP2 wrapped model.
        """
        mixed_precision_config = self.cfg.fsdp_config.mixed_precision
        param_dtype = torch_dtype_from_precision(mixed_precision_config.param_dtype)
        reduce_dtype = torch_dtype_from_precision(mixed_precision_config.reduce_dtype)

        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            cast_forward_inputs=True,
        )

        offload_policy = (
            CPUOffloadPolicy(pin_memory=self.cfg.fsdp_config.offload_pin_memory)
            if self.cfg.fsdp_config.cpu_offload
            else OffloadPolicy()
        )

        fsdp2_model: FSDPModule = apply_fsdp2_to_model(
            module=model,
            config=self.cfg.fsdp_config,
            device_mesh=device_mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            reshard_after_forward=self.cfg.fsdp_config.reshard_after_forward,
        )

        return fsdp2_model

    @classmethod
    def get_fsdp_version(cls) -> FSDPVersion:
        return FSDPVersion.FSDP2

    @torch.no_grad()
    def onload_param_and_grad(
        self, model: FSDPModule, device: torch.device, onload_grad: bool
    ) -> None:
        """
        Load model parameters and gradients to the specified device.

        Args:
            - model (FSDPModule): The FSDP2 wrapped model.
            - device (torch.device): The target device.
            - onload_grad (bool): Whether to load gradients or not.
        """
        model.to(device=device)
        if onload_grad:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = param.grad.to(device)
        clear_memory()

    @torch.no_grad()
    def offload_param_and_grad(self, model: FSDPModule, offload_grad: bool) -> None:
        """
        Offload model parameters and gradients to CPU.

        Args:
            - model (FSDPModule): The FSDP2 wrapped model.
            - offload_grad (bool): Whether to offload gradients or not.
        """
        model.to(device="cpu")

        if offload_grad:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = param.grad.cpu()
        clear_memory()

    @torch.no_grad()
    def offload_optimizer(self, optimizer: Optimizer) -> None:
        """
        Offload optimizer states to CPU.

        Args:
            - optimizer (Optimizer): The optimizer.
        """
        for st in optimizer.state.values():
            if not isinstance(st, dict):
                continue
            for k, v in list(st.items()):
                if torch.is_tensor(v):
                    if v.device.type != "cpu":
                        st[k] = v.detach().to("cpu", non_blocking=True)
                        del v
        clear_memory()

    @torch.no_grad()
    def onload_optimizer(self, optimizer: Optimizer, device: torch.device) -> None:
        """
        Load optimizer states to the specified device.

        Args:
            - optimizer (Optimizer): The optimizer.
            - device (torch.device): The target device.
        """
        for st in optimizer.state.values():
            if not isinstance(st, dict):
                continue
            for k, v in list(st.items()):
                if torch.is_tensor(v):
                    if v.device != device:
                        st[k] = v.detach().to(device, non_blocking=True)
                        del v
        clear_memory()

    def clip_grad_norm_(
        self,
        model: FSDPModule,
        norm_type: Union[float, int] = 2.0,
    ) -> float:
        """
        Clip the gradients of the model parameters by total norm.

        Args:
            - model (FSDPModule): The FSDP2 wrapped model.
            - norm_type (float): The type of the used p-norm.

        Returns:
            - float: The total norm of the gradients before clipping.
        """
        grad_norm = get_grad_norm(
            model.parameters(),
            dp_group=self._dp_group,
            norm_type=norm_type,
        )
        clip_grad_by_total_norm_(
            model.parameters(),
            max_grad_norm=self.cfg.optim.clip_grad,
            total_norm=grad_norm,
        )
        return grad_norm

    def before_micro_batch(
        self, model: FSDPModule, is_last_micro_batch: bool
    ) -> ContextManager:
        """
        Context manager to control gradient synchronization for FSDP2.
        FSDP2 does not provide model.no_sync, but provides set_requires_gradient_sync.

        Args:
            - model (FSDPModule): The FSDP2 wrapped model.
            - is_last_micro_batch (bool): Whether this is the last micro batch.

        Returns:
            - ContextManager: nullcontext, just for interface consistency.
        """
        if not self.cfg.fsdp_config.enable_gradient_accumulation:
            return nullcontext()
        if is_last_micro_batch:
            model.set_requires_gradient_sync(True)
        else:
            model.set_requires_gradient_sync(False)
        return nullcontext()
    
    def load_hf_checkpoint_to_fsdp2_model(
        self,
        model,
        model_path: str,
        device_mesh,
        dtype: torch.dtype,
    ):
        """
        Memory-safer HuggingFace safetensors -> FSDP2/DTensor checkpoint loader.

        Key idea:
        - rank0 reads HF checkpoint tensor on CPU
        - rank0 slices tensor into per-rank shards on CPU
        - scatter only local shard to each rank
        - construct DTensor from local shard
        - load_state_dict(assign=True) to materialize meta parameters

        Requirements:
        - model has already been wrapped by FSDP2 fully_shard()
        - all ranks call this function
        - optimizer must be created AFTER this function
        """

        assert dist.is_initialized(), (
            "Distributed process group must be initialized before FSDP2 checkpoint loading."
        )

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if hasattr(torch, "npu") and torch.npu.is_available():
            device = torch.device(f"npu:{local_rank}")
            torch.npu.set_device(device)
        elif torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
        else:
            raise RuntimeError("No accelerator device found.")

        def _empty_cache():
            gc.collect()
            if hasattr(torch, "npu") and torch.npu.is_available():
                torch.npu.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

        def _get_mesh_size(mesh, mesh_dim: int = 0) -> int:
            try:
                return mesh.size(mesh_dim)
            except TypeError:
                return mesh.size()
            except Exception:
                return world_size

        def _get_mesh_coordinate(mesh):
            try:
                coord = mesh.get_coordinate()
                if coord is not None:
                    return tuple(coord)
            except Exception:
                pass
            return (rank,)

        def _find_shard_placement(dt: DTensor):
            """
            Return (mesh_dim, shard_dim, placement) for first Shard placement.
            FSDP2 usually uses one placement: [Shard(0)].
            """
            for mesh_dim, placement in enumerate(dt.placements):
                if placement.__class__.__name__ == "Shard":
                    shard_dim = getattr(placement, "dim", None)
                    if shard_dim is None:
                        # Some versions store it as _dim.
                        shard_dim = getattr(placement, "_dim", None)
                    if shard_dim is None:
                        raise RuntimeError(f"Cannot get shard dim from placement: {placement}")
                    return mesh_dim, shard_dim, placement
            return None, None, None

        def _compute_unpadded_local_shape(full_shape, shard_dim: int, mesh_size: int, shard_idx: int):
            """
            torch.tensor_split-style uneven split shape.
            """
            full_dim = full_shape[shard_dim]
            base = full_dim // mesh_size
            rem = full_dim % mesh_size
            local_dim = base + (1 if shard_idx < rem else 0)

            local_shape = list(full_shape)
            local_shape[shard_dim] = local_dim
            return tuple(local_shape)

        def _slice_cpu_shard(full_cpu_tensor, shard_dim: int, mesh_size: int, shard_idx: int):
            """
            Slice CPU tensor into the shard for shard_idx using torch.tensor_split semantics.
            """
            chunks = torch.tensor_split(full_cpu_tensor, mesh_size, dim=shard_dim)
            return chunks[shard_idx].contiguous()

        def _make_dtensor_from_local(local_tensor, target_dtensor: DTensor, full_shape):
            """
            Construct DTensor with same mesh/placements as target DTensor.
            Different PyTorch versions have slightly different DTensor.from_local signatures,
            so use a robust fallback chain.
            """
            try:
                return DTensor.from_local(
                    local_tensor,
                    device_mesh=target_dtensor.device_mesh,
                    placements=target_dtensor.placements,
                    run_check=False,
                    shape=torch.Size(full_shape),
                    stride=target_dtensor.stride(),
                )
            except TypeError:
                try:
                    return DTensor.from_local(
                        local_tensor,
                        device_mesh=target_dtensor.device_mesh,
                        placements=target_dtensor.placements,
                        run_check=False,
                    )
                except TypeError:
                    return DTensor.from_local(
                        local_tensor,
                        target_dtensor.device_mesh,
                        target_dtensor.placements,
                    )

        def _is_meta_tensor(x):
            if isinstance(x, DTensor):
                try:
                    return x.to_local().is_meta
                except Exception:
                    return False
            return getattr(x, "is_meta", False)

        def _assert_no_meta_tensors(tag: str):
            bad = []

            for name, p in model.named_parameters():
                if _is_meta_tensor(p):
                    bad.append(name)
                    if len(bad) >= 20:
                        break

            if len(bad) < 20:
                for name, b in model.named_buffers():
                    if _is_meta_tensor(b):
                        bad.append(name)
                        if len(bad) >= 20:
                            break

            if bad:
                raise RuntimeError(
                    f"[{tag}] Still have meta tensors after checkpoint loading. "
                    f"Examples: {bad}"
                )
                
        def _materialize_remaining_meta_buffers(model, device):
            """
            Materialize meta buffers that are not saved in HF checkpoint.

            Common case:
            - rotary_emb.inv_freq is a non-persistent buffer, so it does not appear in state_dict.
            - Since the model was initialized on meta device, this buffer remains meta after load_state_dict.
            """

            def _get_root_config(model):
                # FSDP2-wrapped model usually still exposes .config, but be defensive.
                if hasattr(model, "config"):
                    return model.config
                if hasattr(model, "module") and hasattr(model.module, "config"):
                    return model.module.config
                return None

            root_config = _get_root_config(model)

            def _build_rope_inv_freq(module, config):
                """
                Build standard RoPE inv_freq.

                This covers Qwen/LLaMA-style rotary embeddings:
                    inv_freq = 1 / (rope_theta ** (arange(0, dim, 2) / dim))
                """
                # Prefer module's own fields if available.
                base = getattr(module, "base", None)
                if base is None and config is not None:
                    base = getattr(config, "rope_theta", None)
                if base is None:
                    base = 10000.0

                dim = getattr(module, "dim", None)

                if dim is None and config is not None:
                    head_dim = getattr(config, "head_dim", None)
                    if head_dim is None:
                        hidden_size = getattr(config, "hidden_size", None)
                        num_attention_heads = getattr(config, "num_attention_heads", None)
                        if hidden_size is not None and num_attention_heads is not None:
                            head_dim = hidden_size // num_attention_heads

                    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)

                    if head_dim is not None:
                        dim = int(head_dim * partial_rotary_factor)

                if dim is None:
                    raise RuntimeError(
                        "Cannot infer rotary embedding dim for meta buffer inv_freq. "
                        "Please inspect the rotary embedding module fields."
                    )

                inv_freq = 1.0 / (
                    float(base)
                    ** (
                        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
                        / float(dim)
                    )
                )
                return inv_freq

            materialized = []

            for module_name, module in model.named_modules():
                # Use recurse=False so we can replace buffers on the owning module.
                for buffer_name, buffer in list(module.named_buffers(recurse=False)):
                    if not getattr(buffer, "is_meta", False):
                        continue

                    full_name = f"{module_name}.{buffer_name}" if module_name else buffer_name

                    # Case 1: recent transformers rotary embedding may expose rope_init_fn.
                    if buffer_name == "inv_freq" and "rotary" in module_name.lower():
                        try:
                            rope_init_fn = getattr(module, "rope_init_fn", None)
                            module_config = getattr(module, "config", root_config)

                            if rope_init_fn is not None and module_config is not None:
                                inv_freq, attention_scaling = rope_init_fn(
                                    module_config,
                                    device=device,
                                )
                                inv_freq = inv_freq.to(device=device, dtype=torch.float32)

                                # Some implementations also keep attention_scaling as an attr.
                                if hasattr(module, "attention_scaling"):
                                    module.attention_scaling = attention_scaling
                            else:
                                inv_freq = _build_rope_inv_freq(module, root_config)

                        except Exception:
                            # Fallback to standard RoPE formula.
                            inv_freq = _build_rope_inv_freq(module, root_config)

                        persistent = buffer_name not in getattr(
                            module, "_non_persistent_buffers_set", set()
                        )

                        module.register_buffer(
                            buffer_name,
                            inv_freq,
                            persistent=persistent,
                        )

                        materialized.append(full_name)
                        continue

                    raise RuntimeError(
                        f"Found unsupported meta buffer after checkpoint loading: {full_name}. "
                        f"shape={tuple(buffer.shape)}, dtype={buffer.dtype}. "
                        "Please add a materialization rule for this buffer."
                    )

            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            if rank == 0 and materialized:
                print(
                    f"[FSDP2 load] materialized non-checkpoint meta buffers: {materialized}",
                    flush=True,
                )

        target_state = model.state_dict()

        if rank == 0:
            print(f"[FSDP2 load] target_state keys: {len(target_state)}", flush=True)

        index_file = os.path.join(model_path, "model.safetensors.index.json")

        if os.path.exists(index_file):
            with open(index_file, "r") as f:
                index = json.load(f)

            weight_map = index["weight_map"]
            shard_files = sorted(set(weight_map.values()))
            param_to_file = weight_map
        else:
            shard_files = ["model.safetensors"]
            param_to_file = None

        load_state = {}

        for shard_idx, shard_file in enumerate(shard_files):
            shard_path = os.path.join(model_path, shard_file)

            if rank == 0:
                print(
                    f"[FSDP2 load] loading HF shard {shard_idx + 1}/{len(shard_files)}: {shard_file}",
                    flush=True,
                )
                shard = load_file(shard_path, device="cpu")
            else:
                shard = None

            if param_to_file is not None:
                names_in_this_shard = [
                    name for name, fn in param_to_file.items() if fn == shard_file
                ]
            else:
                names_in_this_shard = list(target_state.keys())

            for name in names_in_this_shard:
                if name not in target_state:
                    if rank == 0:
                        print(
                            f"[FSDP2 load][warn] HF key not in target_state, skip: {name}",
                            flush=True,
                        )
                    continue

                target_value = target_state[name]
                full_shape = tuple(target_value.shape)

                if isinstance(target_value, DTensor):
                    mesh_dim, shard_dim, _ = _find_shard_placement(target_value)

                    if mesh_dim is not None:
                        mesh_size = _get_mesh_size(target_value.device_mesh, mesh_dim)
                        coord = _get_mesh_coordinate(target_value.device_mesh)
                        shard_rank = coord[mesh_dim]

                        local_shape = _compute_unpadded_local_shape(
                            full_shape=full_shape,
                            shard_dim=shard_dim,
                            mesh_size=mesh_size,
                            shard_idx=shard_rank,
                        )

                        recv_tensor = torch.empty(
                            local_shape,
                            dtype=dtype,
                            device=device,
                        )

                        if rank == 0:
                            assert shard is not None
                            if name not in shard:
                                raise KeyError(
                                    f"Key {name} is listed in {shard_file}, "
                                    f"but not found in loaded shard."
                                )

                            full_cpu_tensor = shard[name].to(dtype=dtype).contiguous()

                            scatter_list = []
                            for dst_idx in range(mesh_size):
                                cpu_piece = _slice_cpu_shard(
                                    full_cpu_tensor,
                                    shard_dim=shard_dim,
                                    mesh_size=mesh_size,
                                    shard_idx=dst_idx,
                                )
                                scatter_list.append(cpu_piece.to(device=device, non_blocking=False))

                            del full_cpu_tensor
                        else:
                            scatter_list = None

                        dist.scatter(
                            recv_tensor,
                            scatter_list=scatter_list,
                            src=0,
                        )

                        if rank == 0 and scatter_list is not None:
                            del scatter_list

                        load_state[name] = _make_dtensor_from_local(
                            recv_tensor,
                            target_dtensor=target_value,
                            full_shape=full_shape,
                        )

                        del recv_tensor

                    else:
                        # Replicated DTensor. Usually buffers/small tensors.
                        if rank == 0:
                            assert shard is not None
                            full_tensor = shard[name].to(
                                dtype=dtype,
                                device=device,
                                non_blocking=False,
                            ).contiguous()
                        else:
                            full_tensor = torch.empty(
                                full_shape,
                                dtype=dtype,
                                device=device,
                            )

                        dist.broadcast(full_tensor, src=0)

                        load_state[name] = _make_dtensor_from_local(
                            full_tensor,
                            target_dtensor=target_value,
                            full_shape=full_shape,
                        )

                        del full_tensor

                else:
                    # Non-DTensor parameter/buffer.
                    # Usually small buffers. Broadcast full value.
                    if rank == 0:
                        assert shard is not None
                        full_tensor = shard[name].to(
                            dtype=dtype,
                            device=device,
                            non_blocking=False,
                        ).contiguous()
                    else:
                        full_tensor = torch.empty(
                            full_shape,
                            dtype=dtype,
                            device=device,
                        )

                    dist.broadcast(full_tensor, src=0)
                    load_state[name] = full_tensor
                    del full_tensor

            if shard is not None:
                del shard

            _empty_cache()
            dist.barrier()

        if rank == 0:
            print("[FSDP2 load] calling model.load_state_dict(assign=True)", flush=True)

        incompatible = model.load_state_dict(
            load_state,
            strict=True,
            assign=True,
        )

        missing = incompatible.missing_keys
        unexpected = incompatible.unexpected_keys

        if rank == 0:
            print(f"[FSDP2 load] missing keys: {len(missing)}", flush=True)
            print(f"[FSDP2 load] unexpected keys: {len(unexpected)}", flush=True)
            if missing:
                print(f"[FSDP2 load] first missing: {missing[:20]}", flush=True)
            if unexpected:
                print(f"[FSDP2 load] first unexpected: {unexpected[:20]}", flush=True)

        if missing or unexpected:
            raise RuntimeError(
                f"FSDP2 checkpoint loading failed: "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
            
        _materialize_remaining_meta_buffers(model, device)

        del load_state
        _empty_cache()

        _assert_no_meta_tensors("FSDP2 load")

        dist.barrier()

        if rank == 0:
            print(
                "[FSDP2 load] checkpoint loaded successfully; no meta tensors remain.",
                flush=True,
            )
