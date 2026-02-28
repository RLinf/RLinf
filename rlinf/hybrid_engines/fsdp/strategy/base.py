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
from abc import ABC, abstractmethod
from collections.abc import Iterable
from logging import Logger
from typing import TYPE_CHECKING, ContextManager, Optional, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from rlinf.hybrid_engines.fsdp import FSDP, FSDPModule
from rlinf.hybrid_engines.fsdp.strategy.checkpoint import Checkpoint
from rlinf.hybrid_engines.fsdp.utils import (
    FSDPVersion,
)
from rlinf.utils.utils import clear_memory

if TYPE_CHECKING:
    from rlinf.workers.actor.fsdp_actor_worker import FSDPActor
    from rlinf.workers.inference.fsdp_inference_worker import FSDPInference


class FSDPStrategyBase(ABC):
    def __init__(
        self,
        cfg: DictConfig,
        world_size: int,
        dp_group: Optional[torch.distributed.ProcessGroup] = None,
        logger: Optional[Logger] = None,
    ):
        self.cfg = cfg
        self.logger = logger
        self.world_size = world_size
        self._dp_group = dp_group

    @classmethod
    def create(
        cls,
        cfg: DictConfig,
        world_size: int,
        dp_group: Optional[torch.distributed.ProcessGroup] = None,
        logger=None,
    ) -> "FSDPStrategyBase":
        """
        Factory method: create and return a concrete FSDP strategy instance based on cfg.

        Selection rules (case-insensitive):
        - fsdp         -> FSDPStrategy (classic torch.distributed.fsdp)
        - fsdp2        -> FSDP2Strategy (fully_shard API)

        Args:
            cfg: DictConfig that must contain fsdp_config.strategy
            world_size: actor distributed world size
            dp_group: optional data parallel process group
            logger: optional logger, if none, a default logger will be created

        Returns:
            An instance of a subclass of FSDPStrategyBase.
        """
        assert hasattr(cfg, "fsdp_config"), (
            "fsdp_config is required for creating corresponding FSDP strategy"
        )
        strategy = str(cfg.fsdp_config.get("strategy", "fsdp2")).lower()
        match strategy:
            case FSDPVersion.FSDP:
                from .fsdp import FSDPStrategy

                return FSDPStrategy(
                    cfg=cfg,
                    world_size=world_size,
                    dp_group=dp_group,
                    logger=logger,
                )
            case FSDPVersion.FSDP2:
                from .fsdp2 import FSDP2Strategy

                return FSDP2Strategy(
                    cfg=cfg,
                    world_size=world_size,
                    dp_group=dp_group,
                    logger=logger,
                )
            case _:
                raise ValueError(
                    f"Unknown FSDP strategy '{strategy}'. Expected one of: 'fsdp','fsdp2'"
                )

    @abstractmethod
    def clip_grad_norm_(
        self,
        model: Union[FSDP, FSDPModule],
        norm_type: Union[float, int] = 2.0,
    ) -> float:
        """
        Clip the gradients of the model parameters to a maximum norm.

        Args:
            model (Union[FSDP, FSDPModule]): The model whose gradients are to be clipped.
            norm_type (Union[float,int]): The type of the used p-norm.

        Returns:
            float: The total norm of the parameters before clipping.
        """
        raise NotImplementedError(
            "clip_grad_norm_ method must be implemented by subclasses."
        )

    @abstractmethod
    def wrap_model(
        self, model: nn.Module, device_mesh: DeviceMesh
    ) -> Union[FSDP, FSDPModule]:
        """
        Wrap the model with FSDP or FSDPModule based on the strategy.

        Args:
            model (nn.Module): The model to be wrapped.

        Returns:
            Union[FSDP, FSDPModule]: The wrapped model.
        """
        raise NotImplementedError(
            "_wrap_model method must be implemented by subclasses."
        )

    @classmethod
    @abstractmethod
    def get_fsdp_version(cls) -> FSDPVersion:
        """
        Get the FSDP version associated with the strategy.
        """
        raise NotImplementedError(
            "get_fsdp_version method must be implemented by subclasses."
        )

    @classmethod
    def save_checkpoint(
        cls,
        model: Union[FSDP, FSDPModule],
        optimizers: Union[Optimizer, Iterable[Optimizer]],
        lr_schedulers: Union[LRScheduler, Iterable[LRScheduler]],
        save_path: str,
        save_full_model_weights: bool = True,
        checkpoint_format: str = "dcp",
    ) -> None:
        """
        Save the training state checkpoint.

        Assumes:
        cls must have get_fsdp_version method,
        and torch.distributed has been initialized. Most importantly,
        optimizers should have all state(by calling `fake_optimizer_step`),
        this is done in `init_worker`.

        Args:
            model (Union[FSDP, FSDPModule]): The model to be saved.
            optimizers (Union[Optimizer, Iterable[Optimizer]]): The optimizers to be saved.
            lr_schedulers (LRScheduler): The learning rate scheduler to be saved.
            save_path (str): The path to save the checkpoint.
            save_full_model_weights (bool): Whether to save full model weights.
            checkpoint_format (str): "dcp" or "local_shard".
        """
        clear_memory()
        torch.distributed.barrier()
        opts = StateDictOptions(full_state_dict=False, cpu_offload=True)
        try:
            training_state = Checkpoint(
                model,
                optimizers,
                lr_schedulers,
                opts,
                fsdp_version=cls.get_fsdp_version(),
                checkpoint_format=checkpoint_format,
            )
            if checkpoint_format == "local_shard":
                local_shard_save_path = os.path.join(
                    save_path, "local_shard_checkpoint"
                )
                rank = torch.distributed.get_rank()
                os.makedirs(local_shard_save_path, exist_ok=True)
                torch.save(
                    training_state.state_dict(),
                    os.path.join(local_shard_save_path, f"checkpoint_rank_{rank}.pt"),
                )
            else:
                from torch.distributed import checkpoint as dcp

                dcp_save_path = os.path.join(save_path, "dcp_checkpoint")
                dcp.save(
                    {"fsdp_checkpoint": training_state},
                    checkpoint_id=dcp_save_path,
                )

        except BaseException as e:
            import traceback

            if hasattr(cls, "logger") and cls.logger is not None:
                cls.logger.error(f"Failed to save checkpoint to {save_path}: {e}")
            traceback.print_exc()
            raise e
        torch.distributed.barrier()

        if save_full_model_weights:
            opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
            sd_save_path = os.path.join(save_path, "model_state_dict")
            model_state_dict = get_model_state_dict(model=model, options=opts)
            if torch.distributed.get_rank() == 0:
                os.makedirs(sd_save_path, exist_ok=True)
                torch.save(
                    model_state_dict, os.path.join(sd_save_path, "full_weights.pt")
                )

            torch.distributed.barrier()

    @classmethod
    def load_checkpoint(
        cls,
        model: Union[FSDP, FSDPModule],
        optimizers: Union[Optimizer, Iterable[Optimizer]],
        lr_schedulers: Union[LRScheduler, Iterable[LRScheduler]],
        load_path: str,
        checkpoint_format: str = "dcp",
    ) -> None:
        """
        Load the training state checkpoint.

        Assumes:
        cls must have logger attribute and get_fsdp_version method,
        and torch.distributed has been initialized. Most importantly,
        optimizer should have all state(by calling `fake_optimizer_step`),
        this is done in `init_worker`.

        Args:
            model (Union[FSDP, FSDPModule]): The model to load the checkpoint into.
            optimizers (Union[Optimizer, Iterable[Optimizer]]): The optimizer to load the checkpoint into.
            lr_schedulers (Union[LRScheduler, Iterable[LRScheduler]]): The learning rate scheduler to load the checkpoint into.
            load_path (str): The path to load the checkpoint from.
            checkpoint_format (str): "dcp" or "local_shard".
        """
        opts = StateDictOptions(full_state_dict=False, cpu_offload=True)
        training_state = Checkpoint(
            model=model,
            optimizers=optimizers,
            lr_schedulers=lr_schedulers,
            opts=opts,
            fsdp_version=cls.get_fsdp_version(),
            checkpoint_format=checkpoint_format,
        )
        try:
            if checkpoint_format == "local_shard":
                rank = torch.distributed.get_rank()
                local_ckpt_file = os.path.join(
                    load_path, "local_shard_checkpoint", f"checkpoint_rank_{rank}.pt"
                )
                if not os.path.isfile(local_ckpt_file):
                    raise FileNotFoundError(
                        f"Expected local shard checkpoint '{local_ckpt_file}' not found"
                    )

                if hasattr(cls, "logger") and cls.logger is not None:
                    cls.logger.info(
                        f"[Checkpoint] loading local shard checkpoint from {local_ckpt_file}"
                    )

                checkpoint = torch.load(local_ckpt_file, weights_only=False)

                # version check
                ckpt_fsdp_version = FSDPVersion(checkpoint["fsdp_version"])
                if ckpt_fsdp_version != cls.get_fsdp_version():
                    raise ValueError(
                        f"FSDP version mismatch: {ckpt_fsdp_version} != {cls.get_fsdp_version()}"
                    )

                training_state.load_state_dict(checkpoint)
            else:
                import glob

                from torch.distributed import checkpoint as dcp

                dcp_dir = os.path.join(load_path, "dcp_checkpoint")
                dcp_load_path = dcp_dir if os.path.isdir(dcp_dir) else load_path

                distcp_files = glob.glob(os.path.join(dcp_load_path, "*.distcp"))
                if len(distcp_files) == 0:
                    raise FileNotFoundError(
                        f"Could not find a valid DCP checkpoint under '{dcp_load_path}'. "
                    )

                if hasattr(cls, "logger") and cls.logger is not None:
                    cls.logger.info(
                        f"[Checkpoint] loading DCP checkpoint from {dcp_load_path}"
                    )

                dcp.load(
                    {"fsdp_checkpoint": training_state},
                    checkpoint_id=dcp_load_path,
                )
        except BaseException as e:
            import traceback

            if hasattr(cls, "logger") and cls.logger is not None:
                cls.logger.error(f"Failed to load checkpoint from {load_path}: {e}")
            traceback.print_exc()
            raise e

        torch.distributed.barrier()

    def get_model_state_dict(
        self, model: FSDPModule, cpu_offload: bool, full_state_dict: bool
    ) -> dict:
        """
        Get the full model state dict of FSDP2 from all ranks.

        Args:
            - model (FSDPModule): The FSDP2 wrapped model.
            - cpu_offload (bool): Whether returned state_dict's value will be offloaded to CPU. If true, will
                be copied to CPU memory, or just keep a reference to the original GPU tensor.
            - full_state_dict (bool): Whether to get the full state dict.

        Returns:
            - dict: The state dict of the FSDP/FSDP2 wrapped model according to the specified options.
        """
        opts = StateDictOptions(
            cpu_offload=cpu_offload, full_state_dict=full_state_dict
        )
        state_dict = get_model_state_dict(model=model, options=opts)
        return state_dict

    def load_model_with_state_dict(
        self,
        model: FSDPModule,
        model_state_dict,
        cpu_offload: bool,
        full_state_dict: bool,
    ):
        opts = StateDictOptions(
            cpu_offload=cpu_offload, full_state_dict=full_state_dict
        )
        set_model_state_dict(
            model=model,
            model_state_dict=model_state_dict,
            options=opts,
        )

    @abstractmethod
    def offload_optimizer(self, optimizer: Optimizer) -> None: ...

    @abstractmethod
    def onload_optimizer(self, optimizer: Optimizer, device: torch.device) -> None: ...

    @abstractmethod
    def offload_param_and_grad(
        self, model: Union[FSDP, FSDPModule], offload_grad: bool
    ) -> None: ...

    @abstractmethod
    def onload_param_and_grad(
        self, model: Union[FSDP, FSDPModule], device: torch.device, onload_grad: bool
    ) -> None: ...

    @abstractmethod
    def before_micro_batch(
        self, model: Union[FSDP, FSDPModule], is_last_micro_batch: bool
    ) -> ContextManager: ...

    def setup_inference_sync_actor_ranks(self, inference: "FSDPInference") -> None:
        """
        See `setup_actor_sync_inference_ranks` for details. It will send the sharding metadata
        (including offsets and sizes for each param) to all actor workers, waiting to receive
        their responses(whether inference workers need params from them and offsets,size if so).

        Args:
            - inference (FSDPInference): The FSDP inference worker.
        """
        # param name -> (global_start, needed_size)
        local_meta: list[str, tuple[int, int]] = {}
        inference_model_state_dict = inference.get_model_state_dict(
            cpu_offload=False, full_state_dict=False
        )
        inference_world_size = inference._world_size
        inference_rank = inference._rank
        for name, param in inference_model_state_dict.items():
            if isinstance(param, DTensor):
                full_tensor_size = param.numel()

                shard_size = (
                    full_tensor_size + inference_world_size - 1
                ) // inference_world_size
                global_start = inference_rank * shard_size
                # last rank may have smaller shard
                global_end = min(global_start + shard_size, full_tensor_size)
                needed_size = global_end - global_start
                if needed_size > 0:
                    local_meta[name] = (global_start, needed_size)
            elif torch.is_tensor(param):
                full_tensor_size = param.numel()
                local_meta[name] = (0, full_tensor_size)

        actor_group = inference._actor_group_name
        for actor_rank in range(inference._actor_world_size):
            inference.send(
                dst_rank=actor_rank, dst_group_name=actor_group, object=local_meta
            )

        jobs = [
            inference.recv(
                src_rank=actor_rank, src_group_name=actor_group, async_op=True
            )
            for actor_rank in range(inference._actor_world_size)
        ]
        results = [job.wait() for job in jobs]

        for actor_rank, resp in enumerate(results):
            if resp:
                inference._actor_dst_map[actor_rank] = resp

    def setup_actor_sync_inference_ranks(self, actor: "FSDPActor") -> None:
        """
        Setup the mapping from actor ranks to inference ranks for synchronizing
        their sharded params. It will receive the sharding metadata from all inference workers,
        compute which params need to be sent back to each inference worker, and then send it's metadata
        (including offsets and sizes for each param) to the corresponding inference workers.
        Actually, unlike FSDP, FSDP2's using of DTensor does do better than FSDP's FlatParams,
        it sharded params can be directly computed, but for further ergonomic and development consideration
        (like support multi-dim sharding in future), we still use this way to exchange sharding metadata.

        Args:
            - actor (FSDPActor): The FSDP actor worker.
        """
        inference_group = actor._inference_group_name
        jobs = [
            actor.recv(
                src_rank=inference_rank, src_group_name=inference_group, async_op=True
            )
            for inference_rank in range(actor._inference_world_size)
        ]

        inference_requests: list[dict] = [job.wait() for job in jobs]

        actor_world_size = actor._world_size
        actor_rank = actor._rank

        local_meta = {}
        actor_model_state_dict = actor.get_model_state_dict(
            cpu_offload=False, full_state_dict=False
        )
        for name, param in actor_model_state_dict.items():
            if isinstance(param, DTensor):
                full_tensor_size = param.numel()
                shard_size = (
                    full_tensor_size + actor_world_size - 1
                ) // actor_world_size
                global_start = actor_rank * shard_size
                global_end = min(global_start + shard_size, full_tensor_size)
                size = global_end - global_start
                if size > 0:
                    local_meta[name] = (global_start, size)
            elif torch.is_tensor(param):
                full_tensor_size = param.numel()
                local_meta[name] = (0, full_tensor_size)

        for inference_rank, inference_param_metadata in enumerate(inference_requests):
            resp = {}
            has_intersection = False
            for name, (inf_offset, inf_size) in inference_param_metadata.items():
                if name in local_meta:
                    act_offset, act_size = local_meta[name]
                    # check overlap
                    start1, end1 = inf_offset, inf_offset + inf_size
                    start2, end2 = act_offset, act_offset + act_size

                    global_start = max(start1, start2)
                    global_end = min(end1, end2)
                    # it means there is intersection between this actor shard and sender inference shard,
                    # so we just calculate the intersection part and send back to inference worker
                    if global_start < global_end:
                        needed_sizes = global_end - global_start
                        act_shard_offset = global_start - act_offset
                        inf_shard_offset = global_start - inf_offset
                        resp[name] = (act_shard_offset, inf_shard_offset, needed_sizes)
                        has_intersection = True

            if has_intersection:
                actor._inference_dst_map[inference_rank] = list(resp.keys())

            actor.send(
                dst_rank=inference_rank,
                dst_group_name=inference_group,
                object=resp,
            )
        actor.logger.info(
            f"Actor rank {actor._rank} will send params to inference ranks: {actor._inference_dst_map.keys()}"
        )
