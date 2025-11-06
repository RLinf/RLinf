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

import random
from abc import ABC, abstractmethod
from typing import ContextManager, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.distributed.device_mesh import DeviceMesh
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from rlinf.hybrid_engines.fsdp import FSDP, FSDPModule


class FSDPStrategyBase(ABC):
    def __init__(
        self,
        cfg: DictConfig,
        world_size: int,
        rank: int,
        dp_group: Optional[torch.distributed.ProcessGroup] = None,
        logger=None,
    ):
        self.cfg = cfg
        self._logger = logger
        self.world_size = world_size
        self.rank = rank
        self._dp_group = dp_group

    @property
    def logger(self):
        if self._logger is None:
            import logging

            self._logger = logging.getLogger(__name__)
        return self._logger

    @classmethod
    def create(
        cls,
        cfg: DictConfig,
        world_size: int,
        rank: int,
        dp_group: Optional[torch.distributed.ProcessGroup] = None,
        logger=None,
    ) -> "FSDPStrategyBase":
        """
        Factory method: create and return a concrete FSDP strategy instance based on cfg.

        Selection rules (case-insensitive):
        - fsdp / fsdp1 -> FSDP1Strategy (classic torch.distributed.fsdp)
        - fsdp2        -> FSDP2Strategy (fully_shard API)

        Args:
            cfg: DictConfig that must contain fsdp_config.strategy
            world_size: actor distributed world size
            rank: current process's distributed rank
            dp_group: optional data parallel process group
            logger: optional logger, if none, a default logger will be created

        Returns:
            An instance of a subclass of FSDPStrategyBase.
        """
        assert hasattr(cfg, "fsdp_config"), (
            "fsdp_config is required for creating corresponding FSDP strategy"
        )
        strategy = str(cfg.fsdp_config.get("strategy", "fsdp2")).lower()

        if strategy in ("fsdp", "fsdp1"):
            from .fsdp1 import FSDP1Strategy

            return FSDP1Strategy(
                cfg=cfg,
                world_size=world_size,
                rank=rank,
                dp_group=dp_group,
                logger=logger,
            )
        elif strategy == "fsdp2":
            from .fsdp2 import FSDP2Strategy

            return FSDP2Strategy(
                cfg=cfg,
                world_size=world_size,
                rank=rank,
                dp_group=dp_group,
                logger=logger,
            )
        else:
            raise ValueError(
                f"Unknown FSDP strategy '{strategy}'. Expected one of: 'fsdp', 'fsdp1', 'fsdp2'."
            )

    def load_rng_state(self, rng_state: dict) -> None:
        """
        Load the RNG state from the provided state dictionary.

        Args:
            rng_state (Dict): The RNG state dictionary containing states for 'cpu', 'numpy', 'random', and optionally 'cuda'.
        """
        required_keys = ["cpu", "numpy", "random"]
        assert set(required_keys).issubset(rng_state.keys()), (
            f"rng_state must contain the keys: {required_keys}"
        )

        torch.set_rng_state(rng_state["cpu"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["random"])
        if torch.cuda.is_available() and "cuda" in rng_state:
            torch.cuda.set_rng_state(rng_state["cuda"])

    def save_rng_state(self) -> dict:
        """
        Save the current RNG state into a dictionary.

            Returns:
                Dict: The RNG state dictionary containing states for 'cpu', 'numpy', 'random', and optionally 'cuda'.
        """
        rng_state = {
            "cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state()
        return rng_state

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

    @abstractmethod
    def save_checkpoint(
        self,
        model: Union[FSDP, FSDPModule],
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        save_path: str,
    ) -> None:
        raise NotImplementedError(
            "save_checkpoint method must be implemented by subclasses."
        )

    @abstractmethod
    def load_checkpoint(
        self,
        model: Union[FSDP, FSDPModule],
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        load_path: str,
    ) -> None:
        raise NotImplementedError(
            "load_checkpoint method must be implemented by subclasses."
        )

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
