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

from collections.abc import Iterable
from typing import Union

import torch
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from rlinf.hybrid_engines.fsdp import FSDP, FSDPModule
from rlinf.hybrid_engines.fsdp.utils import FSDPVersion, to_local_if_dtensor
from rlinf.utils.logging import get_logger
from rlinf.utils.utils import get_rng_state, set_rng_state


class Checkpoint(Stateful):
    """Training state; DCP state dictionaries must be built on every rank."""

    def __init__(
        self,
        model: Union[FSDP, FSDPModule],
        optimizers: Union[Optimizer, Iterable[Optimizer]],
        lr_schedulers: Union[LRScheduler, Iterable[LRScheduler]],
        opts: StateDictOptions,
        fsdp_version: FSDPVersion,
        checkpoint_format: str = "dcp",
    ):
        self.model = model
        self.optimizers = optimizers
        self.lr_schedulers = (
            (lr_schedulers,)
            if isinstance(lr_schedulers, LRScheduler)
            else tuple(lr_schedulers)
        )
        self.opts = opts
        self.fsdp_version = fsdp_version
        self.checkpoint_format = checkpoint_format
        self.legacy_rng_state = False

    def _get_local_optim_state_dicts(self):
        if isinstance(self.optimizers, Optimizer):
            return self.optimizers.state_dict()
        return [opt.state_dict() for opt in self.optimizers]

    def _load_local_optim_state_dicts(self, optim_state_dicts):
        if isinstance(self.optimizers, Optimizer):
            self.optimizers.load_state_dict(optim_state_dicts)
        else:
            for opt, opt_sd in zip(self.optimizers, optim_state_dicts):
                opt.load_state_dict(opt_sd)

    def state_dict(self):
        if self.checkpoint_format == "local_shard":
            model_sd = self.model.state_dict()
            model_sd = {
                key: to_local_if_dtensor(value).cpu()
                if isinstance(value, torch.Tensor)
                else value
                for key, value in model_sd.items()
            }
            optim_sd = self._get_local_optim_state_dicts()

            lr_sched_sd = [lr.state_dict() for lr in self.lr_schedulers]

            out = {
                "model": model_sd,
                "optimizers": optim_sd,
                "lr_schedulers": lr_sched_sd,
                "fsdp_version": self.fsdp_version.value,
                "rng": get_rng_state(),
            }
        else:
            model_sd, optim_sd = get_state_dict(
                model=self.model,
                optimizers=self.optimizers,
                options=self.opts,
            )

            lr_sched_sd = [lr.state_dict() for lr in self.lr_schedulers]

            rng_state = get_rng_state()
            if not self.legacy_rng_state:
                all_rng_states = [rng_state]
                if torch.distributed.is_initialized():
                    all_rng_states = [None] * torch.distributed.get_world_size()
                    torch.distributed.all_gather_object(all_rng_states, rng_state)
                # DCP deduplicates replicated values. Give every rank the same
                # complete set. Tuples are serialized as one DCP value, keeping
                # the saved world size intact when loading with a new topology.
                rng_state = tuple(all_rng_states)

            out = {
                "model": model_sd,
                "optimizers": optim_sd,
                "lr_schedulers": lr_sched_sd,
                "fsdp_version": self.fsdp_version.value,
                "rng": rng_state,
            }
        return out

    def load_state_dict(self, state):
        rng_state = state.get("rng")
        if isinstance(rng_state, tuple):
            distributed = torch.distributed.is_initialized()
            world_size = torch.distributed.get_world_size() if distributed else 1
            rank = torch.distributed.get_rank() if distributed else 0
            if len(rng_state) != world_size and rank == 0:
                get_logger().warning(
                    f"RNG world size mismatch: checkpoint has {len(rng_state)} ranks, "
                    f"current job has {world_size}. Existing ranks restore their saved "
                    "RNG states; new ranks keep their initialized RNG states. "
                    "Training is not exactly reproducible across world size changes."
                )
            rng_state = rng_state[rank] if rank < len(rng_state) else get_rng_state()

        assert "fsdp_version" in state, "Checkpoint is missing FSDP version info."
        ckpt_fsdp_version = FSDPVersion(state["fsdp_version"])
        if ckpt_fsdp_version != self.fsdp_version:
            raise ValueError(
                f"FSDP version mismatch: {ckpt_fsdp_version} != {self.fsdp_version}"
            )

        if self.checkpoint_format == "local_shard":
            self.model.load_state_dict(state["model"])

            self._load_local_optim_state_dicts(state["optimizers"])

        else:
            set_state_dict(
                model=self.model,
                optimizers=self.optimizers,
                model_state_dict=state["model"],
                optim_state_dict=state.get("optimizers", state.get("optim")),
                options=self.opts,
            )

        # lr schedulers
        if "lr_schedulers" in state:
            for lr, lr_sd in zip(self.lr_schedulers, state["lr_schedulers"]):
                lr.load_state_dict(lr_sd)

        if rng_state is not None:
            set_rng_state(rng_state)
