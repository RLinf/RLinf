# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


# Adapted from https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/parallel_dims.py

import math
import os
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Callable, Literal, Optional

import torch
from torch import distributed as dist

from ..utils import logging
from ..utils.import_utils import is_torch_npu_available, is_torch_version_greater_than

if is_torch_version_greater_than("2.4"):
    from torch.distributed.device_mesh import DeviceMesh, init_device_mesh


if TYPE_CHECKING:
    from torch.distributed import ProcessGroup
    from torch.distributed.device_mesh import DeviceMesh


logger = logging.get_logger(__name__)

_PARALLEL_STATE: "ParallelState" = None


def requires_mesh(fn: Callable) -> Callable:
    @wraps(fn)
    def _inner(self: "ParallelState", *args, **kwargs):
        if self.device_mesh is None:
            raise ValueError("Device mesh is not initialized.")

        return fn(self, *args, **kwargs)

    return _inner


def init_ep_mesh_matrix(
    ep_size: int, ep_fsdp_size: int, ep_outside: bool = False
) -> "DeviceMesh":
    """
    Initialize the device mesh matrix for the EP.
    Args:
        ep_size (int): The size of the EP.
        ep_fsdp_size (int): The size of the EP-FSDP.
        ep_outside (bool): Whether the EP is outside in ep-fsdp group.
    """
    if ep_outside:
        with torch.device("cpu"):
            mesh = torch.arange(
                math.prod((ep_size, ep_fsdp_size)), dtype=torch.int
            ).view(ep_size, ep_fsdp_size)
    else:
        with torch.device("cpu"):
            mesh = (
                torch.arange(math.prod((ep_size, ep_fsdp_size)), dtype=torch.int)
                .view(ep_fsdp_size, ep_size)
                .transpose(0, 1)
            )
    return mesh


@dataclass(frozen=True)
class ParallelState:
    dp_size: int = 1
    dp_replicate_size: int = 1
    dp_shard_size: int = 1
    tp_size: int = 1
    ep_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    ulysses_size: int = 1
    dp_mode: Literal["ddp", "fsdp1", "fsdp2"] = "fsdp1"
    device_type: str = "npu" if is_torch_npu_available() else "cuda"
    include_sp_in_fsdp: bool = True
    device_mesh: Optional["DeviceMesh"] = None
    ep_fsdp_device_mesh: Optional["DeviceMesh"] = None

    def __post_init__(self):
        if not self.include_sp_in_fsdp:
            raise NotImplementedError(
                "Decoupled sequence parallel has not been implemented."
            )

        if self.cp_size > 1:
            raise NotImplementedError("Ring attention is not supported yet.")

        if (
            self.pp_size
            * self.dp_size
            * self.cp_size
            * self.ulysses_size
            * self.tp_size
            != self.world_size
        ):
            raise ValueError(
                "The product of parallel sizes should be equal to the world size."
            )

        if self.dp_replicate_size * self.dp_shard_size != self.dp_size:
            raise ValueError(
                f"The product of dp_replicate_size: {self.dp_replicate_size} and dp_shard_size: {self.dp_shard_size} should be equal to dp_size: {self.dp_size}."
            )

        if self.sp_enabled:
            from ..distributed.sequence_parallel import (
                init_sequence_parallel,
                set_context_parallel_group,
                set_data_parallel_group,
                set_ulysses_sequence_parallel_group,
                set_unified_sequence_parallel_group,
            )

            if self.device_mesh is not None:
                set_data_parallel_group(self.device_mesh.get_group("dp"))
                if self.ulysses_size > 1:
                    set_ulysses_sequence_parallel_group(
                        self.device_mesh.get_group("ulysses")
                    )
                if self.cp_size > 1:
                    set_context_parallel_group(self.device_mesh.get_group("cp"))
                # set unified sequence parallel group
                set_unified_sequence_parallel_group(self.device_mesh.get_group("sp"))
            else:
                init_sequence_parallel(
                    ulysses_size=self.ulysses_size,
                    sep_dp=True,
                    ulysses_group_key="default",
                    cp_size=self.cp_size,
                )

    @property
    def is_initialized(self) -> bool:
        return dist.is_initialized()

    @property
    def local_rank(self) -> int:
        return int(os.getenv("LOCAL_RANK", "-1"))

    @property
    def global_rank(self) -> int:
        if self.is_initialized:
            return dist.get_rank()
        return -1

    @property
    def world_size(self) -> int:
        if self.is_initialized:
            return dist.get_world_size()
        return 1

    # ------------------------------ DP ------------------------------ #
    @property
    def dp_group(self) -> Optional["ProcessGroup"]:
        if self.device_mesh is not None:
            return self.device_mesh.get_group("dp")

        if self.sp_enabled:
            from ..distributed.sequence_parallel import get_data_parallel_group

            return get_data_parallel_group()

        return self.fsdp_group

    @property
    def dp_rank(self) -> int:
        if self.device_mesh is not None:
            return self.device_mesh.get_local_rank("dp")

        if self.sp_enabled:
            from ..distributed.sequence_parallel import get_data_parallel_rank

            return get_data_parallel_rank()

        return self.fsdp_rank

    @property
    @requires_mesh
    def dp_mesh(self) -> "DeviceMesh":
        if self.device_mesh is not None:
            return self.device_mesh["dp"]

        raise self.fsdp_mesh

    @property
    def dp_enabled(self) -> bool:
        return self.dp_size > 1

    # ------------------------------ DP replicate ------------------------------ #
    @property
    def dp_replicate_group(self) -> Optional["ProcessGroup"]:
        if self.device_mesh is not None:
            return self.device_mesh.get_group("dp_replicate")

    @property
    def dp_replicate_rank(self) -> int:
        if self.device_mesh is not None:
            return self.device_mesh.get_local_rank("dp_replicate")

    @property
    @requires_mesh
    def dp_replicate_mesh(self) -> "DeviceMesh":
        if self.device_mesh is not None:
            return self.device_mesh["dp_replicate"]

    @property
    def dp_replicate_enabled(self) -> bool:
        return self.dp_replicate_size > 1

    # ------------------------------ DP shard ------------------------------ #
    @property
    def dp_shard_group(self) -> Optional["ProcessGroup"]:
        if self.device_mesh is not None:
            return self.device_mesh.get_group("dp_shard")

    @property
    def dp_shard_rank(self) -> int:
        if self.device_mesh is not None:
            return self.device_mesh.get_local_rank("dp_shard")

    @property
    @requires_mesh
    def dp_shard_mesh(self) -> "DeviceMesh":
        if self.device_mesh is not None:
            return self.device_mesh["dp_shard"]

    @property
    def dp_shard_enabled(self) -> bool:
        return self.dp_shard_size >= 1

    # ----------------------------- FSDP ----------------------------- #
    @property
    def fsdp_group(self) -> Optional["ProcessGroup"]:
        if self.device_mesh is not None:
            return self.device_mesh.get_group("dp_sp")

    @property
    def fsdp_rank(self) -> int:
        if self.device_mesh is not None:
            return self.device_mesh.get_local_rank("dp_sp")

        return self.global_rank

    @property
    def dp_shard_sp_enabled(self) -> bool:
        return self.dp_shard_enabled and self.sp_enabled

    @property
    @requires_mesh
    def fsdp_mesh(self) -> "DeviceMesh":
        if self.dp_replicate_enabled:
            # HSDP
            if self.dp_shard_sp_enabled:
                return self.device_mesh["dp_replicate", "dp_shard_sp"]
            elif self.dp_shard_enabled:
                return self.device_mesh["dp_replicate", "dp_shard"]
            else:
                # DDP
                return self.device_mesh["dp_replicate"]
        # FSDP
        elif self.dp_shard_sp_enabled:
            return self.device_mesh["dp_shard_sp"]
        elif self.dp_shard_enabled:
            return self.device_mesh["dp_shard"]
        else:
            return self.device_mesh["dp"]

    @property
    def fsdp_enabled(self) -> bool:
        return self.fsdp_size > 1

    @property
    def fsdp_size(self) -> int:
        return self.world_size // (self.pp_size * self.tp_size)

    # ------------------------------ TP ------------------------------ #
    @property
    @requires_mesh
    def tp_rank(self) -> int:
        return self.device_mesh.get_local_rank("tp")

    @property
    @requires_mesh
    def tp_mesh(self) -> "DeviceMesh":
        return self.device_mesh["tp"]

    @property
    def tp_enabled(self) -> bool:
        return self.tp_size > 1

    # ------------------------------ PP ------------------------------ #
    @property
    @requires_mesh
    def pp_rank(self) -> int:
        return self.device_mesh.get_local_rank("pp")

    @property
    @requires_mesh
    def pp_mesh(self) -> "DeviceMesh":
        return self.device_mesh["pp"]

    @property
    def pp_enabled(self) -> bool:
        return self.pp_size > 1

    @property
    @requires_mesh
    def is_first_pp_stage(self) -> bool:
        return self.pp_rank == 0

    @property
    @requires_mesh
    def is_last_pp_stage(self) -> bool:
        return self.pp_rank == (self.pp_size - 1)

    # ------------------------------ EP ------------------------------ #
    @property
    @requires_mesh
    def ep_mesh(self) -> "DeviceMesh":
        return self.ep_fsdp_device_mesh["ep"]

    @property
    @requires_mesh
    def ep_fsdp_mesh(self) -> "DeviceMesh":
        return self.ep_fsdp_device_mesh["ep", "ep_fsdp"]

    @property
    @requires_mesh
    def ep_group(self) -> "ProcessGroup":
        return self.ep_mesh.get_group()

    @property
    def ep_enabled(self) -> bool:
        return self.ep_size > 1

    @property
    def ep_rank(self) -> int:
        return self.ep_fsdp_device_mesh.get_local_rank("ep")

    # ------------------------------ SP ------------------------------ #
    @property
    def sp_group(self) -> Optional["ProcessGroup"]:
        if self.device_mesh is not None:
            return self.device_mesh.get_group("sp")

        if self.sp_enabled:
            from .sequence_parallel import get_unified_sequence_parallel_group

            return get_unified_sequence_parallel_group()

        return None

    @property
    def sp_rank(self) -> int:
        if self.device_mesh is not None:
            return self.device_mesh.get_local_rank("sp")

        if self.sp_enabled:
            from .sequence_parallel import get_unified_sequence_parallel_rank

            return get_unified_sequence_parallel_rank()

        return -1

    @property
    def sp_enabled(self) -> bool:
        return self.cp_size > 1 or self.ulysses_size > 1

    @property
    def sp_size(self) -> int:
        return self.ulysses_size * self.cp_size

    @property
    def ulysses_group(self) -> Optional["ProcessGroup"]:
        if self.device_mesh is not None:
            return self.device_mesh.get_group("ulysses")

        if self.sp_enabled:
            from .sequence_parallel import get_ulysses_sequence_parallel_group

            return get_ulysses_sequence_parallel_group()

        return None

    @property
    def ulysses_rank(self) -> int:
        if self.device_mesh is not None:
            return self.device_mesh.get_local_rank("ulysses")

        if self.sp_enabled:
            from .sequence_parallel import get_ulysses_sequence_parallel_rank

            return get_ulysses_sequence_parallel_rank()

        return -1

    @property
    def ulysses_enabled(self) -> bool:
        return self.ulysses_size > 1

    @property
    def cp_group(self) -> Optional["ProcessGroup"]:
        if self.device_mesh is not None:
            return self.device_mesh.get_group("cp")

        if self.sp_enabled:
            from .sequence_parallel import get_context_parallel_group

            return get_context_parallel_group()

        return None

    @property
    def cp_rank(self) -> int:
        if self.device_mesh is not None:
            return self.device_mesh.get_local_rank("cp")

        if self.sp_enabled:
            from .sequence_parallel import get_context_parallel_rank

            return get_context_parallel_rank()

        return -1

    @property
    def cp_enabled(self) -> bool:
        return self.cp_size > 1


def init_parallel_state(
    dp_size: int = 1,
    dp_replicate_size: int = 1,
    dp_shard_size: int = 1,
    tp_size: int = 1,
    ep_size: int = 1,
    pp_size: int = 1,
    cp_size: int = 1,
    ulysses_size: int = 1,
    dp_mode: Literal["ddp", "fsdp1", "fsdp2"] = "fsdp1",
    device_type: str | None = None,
    include_sp_in_fsdp: bool = True,
    ep_outside: bool = False,
) -> None:
    """
    Initializes global parallel state.
    """
    global _PARALLEL_STATE
    if _PARALLEL_STATE is not None:
        logger.warning("Parallel state has already been initialized.")
        return

    if device_type is None:
        device_type = "npu" if is_torch_npu_available() else "cuda"

    # Set dp_shard_size to dp_size if dp_shard_size and dp_replicate_size are not set when dp enabled
    if dp_size > 1 and dp_shard_size == 1 and dp_replicate_size == 1:
        dp_shard_size = dp_size

    logger.info_rank0(
        f"Initializing parallel state... dp_size {dp_size}, dp_replicate_size {dp_replicate_size}, dp_shard_size {dp_shard_size},tp_size {tp_size}, pp_size {pp_size}, cp_size {cp_size}, ulysses_size {ulysses_size}"
    )

    device_mesh, ep_fsdp_device_mesh = None, None
    if is_torch_version_greater_than("2.4"):
        mesh_shape = []
        mesh_dim_names = []
        for d, name in zip(
            [pp_size, dp_replicate_size, dp_shard_size, ulysses_size, cp_size, tp_size],
            ["pp", "dp_replicate", "dp_shard", "ulysses", "cp", "tp"],
        ):
            if d > 1 or name in ["dp_shard"]:
                mesh_shape.append(d)
                mesh_dim_names.append(name)

        device_mesh = init_device_mesh(
            device_type=device_type,
            mesh_shape=tuple(mesh_shape),
            mesh_dim_names=tuple(mesh_dim_names),
        )

        # Mesh for data loading (no communication on this mesh)
        dp_mesh_dim_names = []
        # Mesh for param sharding
        dp_shard_sp_mesh_dim_names = []
        # Mesh for loss all-reduce
        dp_sp_mesh_dim_names = []
        # Mesh for sequence parallel
        sp_mesh_dim_names = []

        if dp_replicate_size > 1:
            dp_mesh_dim_names.append("dp_replicate")
            dp_sp_mesh_dim_names.append("dp_replicate")
        if dp_shard_size >= 1:
            dp_mesh_dim_names.append("dp_shard")
            dp_shard_sp_mesh_dim_names.append("dp_shard")
            dp_sp_mesh_dim_names.append("dp_shard")
        if ulysses_size > 1:
            dp_shard_sp_mesh_dim_names.append("ulysses")
            sp_mesh_dim_names.append("ulysses")
            dp_sp_mesh_dim_names.append("ulysses")
        if cp_size > 1:
            dp_shard_sp_mesh_dim_names.append("cp")
            sp_mesh_dim_names.append("cp")
            dp_sp_mesh_dim_names.append("cp")

        if dp_mesh_dim_names != []:
            device_mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp")

        if dp_shard_sp_mesh_dim_names != []:
            device_mesh[tuple(dp_shard_sp_mesh_dim_names)]._flatten(
                mesh_dim_name="dp_shard_sp"
            )

        if dp_sp_mesh_dim_names != []:
            device_mesh[tuple(dp_sp_mesh_dim_names)]._flatten(mesh_dim_name="dp_sp")

        if sp_mesh_dim_names != []:
            device_mesh[tuple(sp_mesh_dim_names)]._flatten(mesh_dim_name="sp")

        if ep_size > 1:
            world_size = dist.get_world_size()
            assert world_size % ep_size == 0, "ep_size must be a factor of world_size"
            ep_fsdp_size = world_size // ep_size

            mesh = init_ep_mesh_matrix(
                ep_size=ep_size, ep_fsdp_size=ep_fsdp_size, ep_outside=ep_outside
            )
            ep_fsdp_device_mesh = DeviceMesh(
                device_type=device_type,
                mesh=mesh,
                mesh_dim_names=("ep", "ep_fsdp"),
            )

        logger.info_rank0(f"Device mesh: {device_mesh}")
        logger.info_rank0(f"EP FSDP device mesh: {ep_fsdp_device_mesh}")

    _PARALLEL_STATE = ParallelState(
        dp_size=dp_size,
        dp_replicate_size=dp_replicate_size,
        dp_shard_size=dp_shard_size,
        tp_size=tp_size,
        ep_size=ep_size,
        pp_size=pp_size,
        cp_size=cp_size,
        ulysses_size=ulysses_size,
        dp_mode=dp_mode,
        device_type=device_type,
        include_sp_in_fsdp=include_sp_in_fsdp,
        device_mesh=device_mesh,
        ep_fsdp_device_mesh=ep_fsdp_device_mesh,
    )


def get_parallel_state() -> "ParallelState":
    """
    Returns global parallel state.
    """
    if _PARALLEL_STATE is None:
        logger.warning_once(
            "Parallel state has not been initialized. returning default Single-process state."
        )
        return ParallelState()

    return _PARALLEL_STATE
