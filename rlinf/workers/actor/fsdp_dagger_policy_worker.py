# Copyright 2026 The RLinf Authors.
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

import asyncio
import logging
import os

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.config import SupportedModel, get_supported_model
from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Channel, Worker
from rlinf.utils import drq
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict, compute_split_num
from rlinf.utils.nested_dict_process import put_tensor_device, split_dict_to_chunk
from rlinf.utils.utils import clear_memory
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.data.rolling_lerobot_dataset import (
    build_rolling_lerobot_dataset,
    build_dataloader_from_dataset,
)

class EmbodiedDAGGERFSDPPolicy(EmbodiedFSDPActor):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.replay_buffer = None
        self.update_step = 0
        self.enable_drq = bool(getattr(self.cfg.actor, "enable_drq", False))
        self.dataset = None
        self.data_source = cfg.actor.get("data_source", "buffer")

    def _build_sft_data_loader(self):
        self.dataset = build_rolling_lerobot_dataset(
            root_dir=self.cfg.actor.sft_data_path,
            chunk_size=self.cfg.actor.model.num_action_chunks,
            min_datasets=self.cfg.actor.get("min_datasets", 1),
            wait_interval_s=self.cfg.actor.get("wait_interval_s", 10.0),
        )
        self.data_loader = build_dataloader_from_dataset(
            dataset=self.dataset,
            batch_size=self.cfg.actor.micro_batch_size * self._world_size,
            world_size=self._world_size,
            rank=self._rank,
        )
        self.data_iter = iter(self.data_loader)
        self._data_iter_offset = 0
        self._data_epoch = 0

    def init_worker(self):
        super().setup_model_and_optimizer()
        self.setup_dagger_components()
        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()
        self._setup_rollout_weight_dst_ranks()
        if self.cfg.actor.get("compile_model", False):
            self.model = torch.compile(self.model, mode="default")

    def setup_dagger_components(self):
        """Initialize DAgger-specific replay buffer state."""
        seed = self.cfg.actor.get("seed", 1234)
        auto_save_path = self.cfg.algorithm.replay_buffer.get("auto_save_path", None)
        if auto_save_path is None:
            auto_save_path = os.path.join(
                self.cfg.runner.logger.log_path, f"replay_buffer/rank_{self._rank}"
            )
        else:
            auto_save_path = os.path.join(auto_save_path, f"rank_{self._rank}")
        if self.data_source == "buffer":
            self.replay_buffer = TrajectoryReplayBuffer(
                seed=seed,
                enable_cache=self.cfg.algorithm.replay_buffer.enable_cache,
                cache_size=self.cfg.algorithm.replay_buffer.cache_size,
                sample_window_size=self.cfg.algorithm.replay_buffer.sample_window_size,
                auto_save=self.cfg.algorithm.replay_buffer.get("auto_save", False),
                auto_save_path=auto_save_path,
                trajectory_format=self.cfg.algorithm.replay_buffer.get(
                    "trajectory_format", "pt"
                ),
            )

    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        clear_memory(sync=False)

        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        recv_list = []
        for _ in range(split_num):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            recv_list.append(trajectory)

        if self.data_source == "buffer":
            return self.recv_buffer_rollout_trajectories(recv_list)
        elif self.data_source == "lerobot":
            return self.recv_lerobot_rollout_trajectories()
        else:
            raise ValueError(f"Invalid data source: {self.data_source}")

    def recv_buffer_rollout_trajectories(self, recv_list: list[Trajectory]) -> None:
        intervene_traj_list = []
        for traj in recv_list:
            assert isinstance(traj, Trajectory)
            intervene_trajs = traj.extract_intervene_traj(mode="all")
            if intervene_trajs is not None:
                intervene_traj_list.extend(intervene_trajs)
        if intervene_traj_list:
            self.replay_buffer.add_trajectories(intervene_traj_list)
    
    def recv_lerobot_rollout_trajectories(self) -> None:
        if self.data_source == "lerobot":
            if self.dataset is None:
                self._build_sft_data_loader()
        self.dataset.refresh()
        self.data_loader = build_dataloader_from_dataset(
            dataset=self.dataset,
            batch_size=self.cfg.actor.micro_batch_size * self._world_size,
            world_size=self._world_size,
            rank=self._rank,
        )

    def _prepare_sft_batch(self, batch):
        """Prepare model-specific DAgger training inputs."""
        if self.data_source == "buffer":
            return self.model.prepare_dagger_sft_batch(batch)
        elif self.data_source == "lerobot":
            return self.model.prepare_lerobot_sft_batch(batch)
        else:
            raise ValueError(f"Invalid data source: {self.data_source}")

    def _reduce_sft_loss(self, loss):
        """Reduce model-specific SFT loss to a scalar."""
        if not isinstance(loss, torch.Tensor):
            loss = torch.as_tensor(loss, device=self.device)

        if (
            get_supported_model(self.cfg.actor.model.model_type)
            == SupportedModel.OPENPI
        ):
            action_chunk = self.model.config.action_chunk
            action_dim = self.model.config.action_env_dim
            loss = loss[:, :action_chunk, :action_dim]

        return loss.mean()

    @Worker.timer("forward_actor")
    def forward_actor(self, batch):
        """Run one supervised forward pass for DAgger."""
        data = self._prepare_sft_batch(batch)
        actor_loss = self.model(forward_type=ForwardType.SFT, data=data)
        return self._reduce_sft_loss(actor_loss)

    @Worker.timer("update_one_epoch")
    def update_one_epoch(self):
        if self.data_source == "buffer":
            return self.update_buffer_one_epoch()
        elif self.data_source == "lerobot":
            return self.update_lerobot_one_epoch()
        else:
            raise ValueError(f"Invalid data source: {self.data_source}")

    def update_buffer_one_epoch(self):
        """Run one replay-buffer update epoch for DAgger."""
        global_batch_size_per_rank = (
            self.cfg.actor.global_batch_size // self._world_size
        )
        with self.worker_timer("sample"):
            global_batch = self.replay_buffer.sample(
                num_chunks=global_batch_size_per_rank
            )

        train_micro_batch_list = split_dict_to_chunk(
            global_batch,
            global_batch_size_per_rank // self.cfg.actor.micro_batch_size,
        )
        for idx, batch in enumerate(train_micro_batch_list):
            batch = put_tensor_device(batch, device=self.device)
            if self.enable_drq:
                drq.apply_drq(batch["curr_obs"], pad=4)
                drq.apply_drq(batch["next_obs"], pad=4)
            train_micro_batch_list[idx] = batch

        self.optimizer.zero_grad()
        gbs_actor_loss = []
        for mb_idx, batch in enumerate(train_micro_batch_list):
            backward_ctx = self.before_micro_batch(
                self.model,
                is_last_micro_batch=(mb_idx + 1) == self.gradient_accumulation,
            )
            with self.amp_context:
                actor_loss = self.forward_actor(batch["forward_inputs"])
            actor_loss = actor_loss / self.gradient_accumulation
            with backward_ctx:
                self.grad_scaler.scale(actor_loss).backward()
            gbs_actor_loss.append(actor_loss.item() * self.gradient_accumulation)

        actor_grad_norm = self.model.clip_grad_norm_(
            max_norm=self.cfg.actor.optim.clip_grad
        )
        self.optimizer.step()
        self.lr_scheduler.step()

        return {
            "dagger/actor_loss": np.mean(gbs_actor_loss),
            "actor/lr": self.optimizer.param_groups[0]["lr"],
            "actor/grad_norm": actor_grad_norm,
        }
        
    def update_lerobot_one_epoch(self):
        """Run one Lerobot update epoch for DAgger."""
        gbs_actor_loss = []
        for idx in range(self.gradient_accumulation):
            # set the gradient accumulation backward_ctx
            backward_ctx = self.before_micro_batch(
                self.model,
                is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
            )

            try:
                batch = next(self.data_iter)
                self._data_iter_offset += 1
            except StopIteration:
                self._data_epoch += 1
                logging.info(
                    f"[INFO] data_iter exhausted, reset iterator self._data_epoch {self._data_epoch}"
                )
                if hasattr(self.data_loader, "sampler") and hasattr(
                    self.data_loader.sampler, "set_epoch"
                ):
                    self.data_loader.sampler.set_epoch(self._data_epoch)
                self.data_iter = iter(self.data_loader)
                batch = next(self.data_iter)
                self._data_iter_offset = 1

            with self.amp_context:
                actor_loss = self.forward_actor(batch)
            
            actor_loss = actor_loss / self.gradient_accumulation
            with backward_ctx:
                self.grad_scaler.scale(actor_loss).backward()
            gbs_actor_loss.append(actor_loss.item() * self.gradient_accumulation)

        actor_grad_norm = self.model.clip_grad_norm_(
            max_norm=self.cfg.actor.optim.clip_grad
        )
        self.optimizer.step()
        self.lr_scheduler.step()

        return {
            "dagger/actor_loss": np.mean(gbs_actor_loss),
            "actor/lr": self.optimizer.param_groups[0]["lr"],
            "actor/grad_norm": actor_grad_norm,
        }

    
    def process_train_metrics(self, metrics):
        """Aggregate DAgger training and replay-buffer metrics."""
        if self.data_source == "buffer":
            replay_buffer_stats = self.replay_buffer.get_stats()
            replay_buffer_stats = {
                f"replay_buffer/{key}": value for key, value in replay_buffer_stats.items()
            }
            append_to_dict(metrics, replay_buffer_stats)

        mean_metric_dict = {}
        for key, value in metrics.items():
            if isinstance(value, list) and value:
                cpu_values = [
                    v.detach().cpu().item() if isinstance(v, torch.Tensor) else v
                    for v in value
                ]
                mean_metric_dict[key] = np.mean(cpu_values)
            else:
                mean_metric_dict[key] = (
                    value.detach().cpu().item()
                    if isinstance(value, torch.Tensor)
                    else value
                )

        return all_reduce_dict(mean_metric_dict, op=torch.distributed.ReduceOp.AVG)

    @Worker.timer("run_training")
    def run_training(self):
        """Run DAgger updates with replay-buffer samples."""
        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)

        if self.data_source == "buffer":
            min_buffer_size = self.cfg.algorithm.replay_buffer.get("min_buffer_size", 100)
            if not self.replay_buffer.is_ready(min_buffer_size):
                self.log_on_first_rank(
                    f"Replay buffer size {len(self.replay_buffer)} < {min_buffer_size}, skipping training"
                )
                return {}

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )
        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        self.model.train()
        metrics = {}
        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            metrics_data = self.update_one_epoch()
            append_to_dict(metrics, metrics_data)
            self.update_step += 1

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return self.process_train_metrics(metrics)

    def compute_advantages_and_returns(self):
        """Skip advantage computation for supervised DAgger updates."""
        return {}

    def save_checkpoint(self, save_base_path, step):
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
            self.is_weight_offloaded = False
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)
            self.is_optimizer_offloaded = False

        self._strategy.save_checkpoint(
            model=self.model,
            optimizers=[self.optimizer],
            lr_schedulers=[self.lr_scheduler],
            save_path=save_base_path,
            checkpoint_format="local_shard"
            if self.cfg.actor.fsdp_config.use_orig_params
            else "dcp",
        )

        if self.data_source == "buffer":
            buffer_save_path = os.path.join(
                save_base_path, f"dagger_components/replay_buffer/rank_{self._rank}"
            )
            self.replay_buffer.save_checkpoint(buffer_save_path)

    def load_checkpoint(self, load_base_path):
        self._strategy.load_checkpoint(
            model=self.model,
            optimizers=[self.optimizer],
            lr_schedulers=[self.lr_scheduler],
            load_path=load_base_path,
            checkpoint_format="local_shard"
            if self.cfg.actor.fsdp_config.use_orig_params
            else "dcp",
        )

        if self.data_source == "buffer":
            buffer_load_path = os.path.join(
                load_base_path, f"dagger_components/replay_buffer/rank_{self._rank}"
            )
            self.replay_buffer.load_checkpoint(buffer_load_path)
