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

from typing import Union

from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from rlinf.hybrid_engines.fsdp import FSDP, FSDPModule
from rlinf.hybrid_engines.fsdp.utils import FSDPVersion
from rlinf.utils.utils import get_rng_state, set_rng_state


class Checkpoint(Stateful):
    def __init__(
        self,
        model: Union[FSDP, FSDPModule],
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        opts: StateDictOptions,
        fsdp_version: FSDPVersion,
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.opts = opts
        self.fsdp_version = fsdp_version

    def state_dict(self):
        model_sd, optim_sd = get_state_dict(
            model=self.model, optimizers=self.optimizer, options=self.opts
        )
        out = {
            "model": model_sd,
            "optim": optim_sd,
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "fsdp_version": self.fsdp_version.value,
        }
        out["rng"] = get_rng_state()
        return out

    def load_state_dict(self, state):
        assert "fsdp_version" in state, "Checkpoint is missing FSDP version info."
        ckpt_fsdp_version = FSDPVersion(state["fsdp_version"])
        if ckpt_fsdp_version != self.fsdp_version:
            raise ValueError(
                f"FSDP version mismatch: checkpoint version {ckpt_fsdp_version} != current version {self.fsdp_version}"
            )
        set_state_dict(
            model=self.model,
            optimizers=self.optimizer,
            model_state_dict=state["model"],
            optim_state_dict=state["optim"],
            options=self.opts,
        )
        if self.lr_scheduler is not None and "lr_scheduler" in state:
            self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        if "rng" in state:
            set_rng_state(state["rng"])
