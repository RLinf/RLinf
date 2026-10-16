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

import json
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
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

from rlinf.hybrid_engines.fsdp import FSDP, DTensor, FSDPModule
from rlinf.hybrid_engines.fsdp.utils import FSDPVersion, to_local_if_dtensor
from rlinf.utils.logging import get_logger
from rlinf.utils.utils import get_rng_state, set_rng_state

_CHECKPOINT_FORMATS = frozenset(("dcp", "local_shard"))


def validate_checkpoint_format(checkpoint_format: str) -> str:
    """Validate and normalize the FSDP checkpoint serialization format."""
    normalized = str(checkpoint_format).strip().lower()
    if normalized not in _CHECKPOINT_FORMATS:
        raise ValueError(
            "Unsupported FSDP checkpoint format "
            f"{checkpoint_format!r}; expected one of "
            f"{sorted(_CHECKPOINT_FORMATS)}."
        )
    return normalized


def should_save_training_state(
    step: int,
    interval: int | None = None,
    *,
    force: bool = False,
) -> bool:
    """Select resumable saves; a null interval preserves every save."""
    if interval is not None and (
        isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0
    ):
        raise ValueError(
            "training_state_save_interval must be null or a positive integer."
        )
    return interval is None or force or step % interval == 0


def validate_checkpoint_for_resume(load_path: str) -> None:
    """Reject unfinished or inference-only checkpoints before loading state."""
    metadata_path = Path(load_path) / "checkpoint_metadata.json"
    if not metadata_path.is_file():
        return
    metadata = json.loads(metadata_path.read_text())
    if not metadata.get("complete", False):
        raise ValueError(
            f"Checkpoint {load_path} is incomplete; cannot resume training."
        )
    if not metadata.get("save_training_state", False):
        raise ValueError(
            f"Checkpoint {load_path} contains inference weights only; "
            "cannot resume training without model, optimizer and RNG state."
        )


def validate_checkpoint_communication_backend(
    backend: str | None, checkpoint_format: str = "dcp"
) -> str | None:
    """Allow an opt-in CPU communication group for DCP checkpoints."""
    if backend is None:
        return None
    if not isinstance(backend, str) or backend.strip().lower() != "gloo":
        raise ValueError("checkpoint_communication_backend must be null or 'gloo'.")
    if validate_checkpoint_format(checkpoint_format) != "dcp":
        raise ValueError("checkpoint_communication_backend='gloo' requires DCP.")
    return "gloo"


@contextmanager
def checkpoint_communication_group(
    backend: str | None, checkpoint_format: str = "dcp"
) -> Iterator[torch.distributed.ProcessGroup | None]:
    """Own a checkpoint-only group; the existing group is never replaced."""
    backend = validate_checkpoint_communication_backend(backend, checkpoint_format)
    if backend is None:
        yield None
        return
    from rlinf.scheduler import Cluster

    group = torch.distributed.new_group(
        backend=backend, timeout=Cluster.get_collective_timeout()
    )
    try:
        yield group
    finally:
        torch.distributed.destroy_process_group(group)


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
        process_group: torch.distributed.ProcessGroup | None = None,
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
        self.checkpoint_format = validate_checkpoint_format(checkpoint_format)
        self.legacy_rng_state = False
        self.process_group = process_group

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

    def _has_dtensor(self) -> bool:
        """Return whether any parameter or buffer is a DTensor.

        FSDP1 ``state_dict()`` gathers the full parameter set onto rank 0.
        The live parameters and buffers already answer this question, so a
        model with only local tensors can skip that gather.
        """
        if any(isinstance(tensor, DTensor) for tensor in self.model.parameters()):
            return True
        return any(isinstance(tensor, DTensor) for tensor in self.model.buffers())

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
                    all_rng_states = [None] * torch.distributed.get_world_size(
                        self.process_group
                    )
                    torch.distributed.all_gather_object(
                        all_rng_states, rng_state, group=self.process_group
                    )
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
            world_size = (
                torch.distributed.get_world_size(self.process_group)
                if distributed
                else 1
            )
            rank = torch.distributed.get_rank(self.process_group) if distributed else 0
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
            model_sd = state["model"]
            if self._has_dtensor():
                model_sd = model_sd.copy()
                for key, target in self.model.state_dict().items():
                    if isinstance(target, DTensor) and key in model_sd:
                        local_tensor = model_sd[key]
                        if local_tensor.shape != target.to_local().shape:
                            raise ValueError(
                                f"Local shard shape mismatch for {key}: "
                                f"{local_tensor.shape} != {target.to_local().shape}. "
                                "local_shard checkpoints require the same model and "
                                "distributed topology used when saving."
                            )
                        # The file stores local tensors, while load_state_dict
                        # expects the global DTensor shape, including for uneven
                        # shards.
                        model_sd[key] = DTensor.from_local(
                            local_tensor.to(target.device),
                            device_mesh=target.device_mesh,
                            placements=target.placements,
                            shape=target.shape,
                            stride=target.stride(),
                        )
            self.model.load_state_dict(model_sd)

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
