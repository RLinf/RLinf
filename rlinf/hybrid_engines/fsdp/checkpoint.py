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
from pathlib import Path
from typing import TypedDict

import torch
from torch.distributed import checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from rlinf.hybrid_engines.fsdp import FSDP, DTensor, FSDPModule
from rlinf.hybrid_engines.fsdp.utils import FSDPVersion, to_local_if_dtensor
from rlinf.scheduler import Worker
from rlinf.utils.logging import get_logger
from rlinf.utils.utils import clear_memory, get_rng_state, set_rng_state


class OptimizerProgress(TypedDict):
    """Update count and configured warmup length for the saved optimizer."""

    optimizer_steps: int
    critic_warmup_steps: int


class _TrainingState(Stateful):
    """Training state; DCP state dictionaries must be built on every rank."""

    def __init__(
        self,
        model: FSDP | FSDPModule,
        optimizers: Optimizer | Iterable[Optimizer],
        lr_schedulers: LRScheduler | Iterable[LRScheduler],
        opts: StateDictOptions,
        fsdp_version: FSDPVersion,
        checkpoint_format: str = "dcp",
        optimizer_progress: OptimizerProgress | None = None,
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
        self.optimizer_progress = optimizer_progress
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
        if self.optimizer_progress is not None:
            out["optimizer_progress"] = self.optimizer_progress
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


class Checkpoint:
    """Save and load training state for one checkpoint path on all ranks."""

    def __init__(
        self,
        path: str,
        fsdp_version: FSDPVersion,
        *,
        checkpoint_format: str = "dcp",
    ) -> None:
        self._path = Path(path)
        self._format = checkpoint_format
        self._fsdp_version = fsdp_version
        # Reuse the same source when inspecting progress before loading.
        self._source: dict | tuple[dcp.FileSystemReader, dict] | None = None

    def _open(self) -> dict | tuple[dcp.FileSystemReader, dict]:
        if self._source is not None:
            return self._source

        path = self._path
        rank = torch.distributed.get_rank()
        if self._format == "local_shard":
            shard = path / "local_shard_checkpoint" / f"checkpoint_rank_{rank}.pt"
            if rank == 0:
                get_logger().info(f"Loading local shard checkpoint from {shard.parent}")
            self._source = torch.load(shard, weights_only=False)
        else:
            dcp_dir = path / "dcp_checkpoint"
            if dcp_dir.is_dir():
                path = dcp_dir
            if rank == 0:
                get_logger().info(f"Loading DCP checkpoint from {path}")
            reader = dcp.FileSystemReader(path)
            self._source = (reader, reader.read_metadata().state_dict_metadata)
        return self._source

    def read_optimizer_progress(self) -> OptimizerProgress | None:
        """Read the saved optimizer progress before constructing parameter groups."""
        source = self._open()
        if isinstance(source, dict):
            return source.get("optimizer_progress")
        reader, metadata = source
        if "fsdp_checkpoint.optimizer_progress.optimizer_steps" not in metadata:
            return None

        progress = OptimizerProgress(optimizer_steps=0, critic_warmup_steps=0)
        dcp.load(
            {"fsdp_checkpoint": {"optimizer_progress": progress}},
            storage_reader=reader,
        )
        return progress

    def load(
        self,
        model: FSDP | FSDPModule,
        optimizers: Optimizer | Iterable[Optimizer],
        lr_schedulers: LRScheduler | Iterable[LRScheduler],
    ) -> None:
        """Restore into the supplied model, optimizers, and schedulers on every rank."""
        source = self._open()
        self._source = None
        training_state = _TrainingState(
            model=model,
            optimizers=optimizers,
            lr_schedulers=lr_schedulers,
            opts=StateDictOptions(full_state_dict=False, cpu_offload=True),
            fsdp_version=self._fsdp_version,
            checkpoint_format=self._format,
        )
        if isinstance(source, dict):
            training_state.load_state_dict(source)
        else:
            reader, metadata = source
            # Old DCP checkpoints flattened a single RNG dictionary.
            training_state.legacy_rng_state = "fsdp_checkpoint.rng" not in metadata
            dcp.load(
                {"fsdp_checkpoint": training_state},
                storage_reader=reader,
            )

        torch.distributed.barrier()

    def save(
        self,
        model: FSDP | FSDPModule,
        optimizers: Optimizer | Iterable[Optimizer],
        lr_schedulers: LRScheduler | Iterable[LRScheduler],
        *,
        save_full_model_weights: bool = True,
        optimizer_progress: OptimizerProgress | None = None,
    ) -> None:
        """Save training state collectively and optionally export full model weights.

        All ranks must call this function with initialized optimizer state.
        """
        self._source = None
        clear_memory()
        torch.distributed.barrier()
        rank = torch.distributed.get_rank()
        training_state = _TrainingState(
            model,
            optimizers,
            lr_schedulers,
            StateDictOptions(full_state_dict=False, cpu_offload=True),
            fsdp_version=self._fsdp_version,
            checkpoint_format=self._format,
            optimizer_progress=optimizer_progress,
        )
        if self._format == "local_shard":
            shard_dir = self._path / "local_shard_checkpoint"
            shard_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                training_state.state_dict(), shard_dir / f"checkpoint_rank_{rank}.pt"
            )
        else:
            dcp.save(
                {"fsdp_checkpoint": training_state},
                checkpoint_id=self._path / "dcp_checkpoint",
            )
        torch.distributed.barrier()

        if save_full_model_weights:
            opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
            model_state = get_model_state_dict(model=model, options=opts)
            if rank == 0:
                weights_dir = self._path / "model_state_dict"
                weights_dir.mkdir(parents=True, exist_ok=True)
                weights_path = weights_dir / "full_weights.pt"
                if Worker.torch_device_type == "npu":
                    model_state = {
                        key: value.cpu()
                        if isinstance(value, torch.Tensor) and value.is_npu
                        else value
                        for key, value in model_state.items()
                    }
                    torch.save(
                        model_state, weights_path, _use_new_zipfile_serialization=True
                    )
                else:
                    torch.save(model_state, weights_path)
            torch.distributed.barrier()
