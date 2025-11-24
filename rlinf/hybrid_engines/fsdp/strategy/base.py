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

from abc import ABC, abstractmethod
from logging import Logger
from typing import ContextManager, Optional, Union

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from omegaconf import DictConfig
from torch.distributed.checkpoint.state_dict import StateDictOptions
from torch.distributed.device_mesh import DeviceMesh
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from rlinf.hybrid_engines.fsdp import FSDP, FSDPModule
from rlinf.hybrid_engines.fsdp.strategy.checkpoint import Checkpoint
from rlinf.hybrid_engines.fsdp.utils import FSDPVersion


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
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        save_path: str,
    ) -> None:
        """
        Save the training state checkpoint.

        Assumes:
        cls must have get_fsdp_version method,
        and torch.distributed has been initialized. Most importantly,
        optimizer should have all state(by calling `fake_optimizer_step`),
        this is done in `init_worker`.

        Args:
            model (Union[FSDP, FSDPModule]): The model to be saved.
            optimizer (Optimizer): The optimizer to be saved.
            lr_scheduler (LRScheduler): The learning rate scheduler to be saved.
            save_path (str): The path to save the checkpoint.
        """
        torch.distributed.barrier()
        opts = StateDictOptions(full_state_dict=False, cpu_offload=True)
        try:
            training_state = Checkpoint(
                model,
                optimizer,
                lr_scheduler,
                opts,
                fsdp_version=cls.get_fsdp_version(),
            )
            dcp.save({"fsdp_checkpoint": training_state}, checkpoint_id=save_path)
        except BaseException as e:
            import traceback

            if hasattr(cls, "logger") and cls.logger is not None:
                cls.logger.error(f"Failed to save checkpoint to {save_path}: {e}")
            traceback.print_exc()
            raise e

        torch.distributed.barrier()

    @classmethod
    def load_checkpoint(
        cls,
        model: Union[FSDP, FSDPModule],
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        load_path: str,
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
            optimizer (Optimizer): The optimizer to load the checkpoint into.
            lr_scheduler (LRScheduler): The learning rate scheduler to load the checkpoint into.
            load_path (str): The path to load the checkpoint from.
        """
        torch.distributed.barrier()
        opts = StateDictOptions(full_state_dict=False, cpu_offload=True)
        try:
            training_state = Checkpoint(
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                opts=opts,
                fsdp_version=cls.get_fsdp_version(),
            )
            dcp.load({"fsdp_checkpoint": training_state}, checkpoint_id=load_path)
        except BaseException as e:
            import traceback

            if hasattr(cls, "logger") and cls.logger is not None:
                cls.logger.error(f"Failed to load checkpoint from {load_path}: {e}")
            traceback.print_exc()
            raise e
        torch.distributed.barrier()

    @abstractmethod
    def get_model_state_dict(self, model: Union[FSDP, FSDPModule]) -> dict:
        raise NotImplementedError(
            "state_dict method must be implemented by subclasses."
        )

    @abstractmethod
    def offload_optimizer(self, optimizer: Optimizer) -> None:
        raise NotImplementedError(
            "offload_optimizer method must be implemented by subclasses."
        )

    @abstractmethod
    def onload_optimizer(self, optimizer: Optimizer, device: torch.device) -> None:
        raise NotImplementedError(
            "onload_optimizer method must be implemented by subclasses."
        )

    @abstractmethod
    def offload_param_and_grad(
        self, model: Union[FSDP, FSDPModule], offload_grad: bool
    ) -> None:
        raise NotImplementedError(
            "offload_param method must be implemented by subclasses."
        )

    @abstractmethod
    def onload_param_and_grad(
        self, model: Union[FSDP, FSDPModule], device: torch.device, onload_grad: bool
    ) -> None:
        raise NotImplementedError(
            "onload_param method must be implemented by subclasses."
        )

    @abstractmethod
    def before_micro_batch(
        self, model: Union[FSDP, FSDPModule], is_last_micro_batch: bool
    ) -> ContextManager:
        raise NotImplementedError(
            "before_micro_batch method must be implemented by subclasses."
        )
