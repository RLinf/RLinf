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
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


class EmbodiedDAGGERFSDPPolicy(EmbodiedFSDPActor):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.replay_buffer = None
        self.update_step = 0
        self.enable_drq = bool(getattr(self.cfg.actor, "enable_drq", False))

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
        seed = self.cfg.actor.get("seed", 1234)
        auto_save_path = self.cfg.algorithm.replay_buffer.get("auto_save_path", None)
        if auto_save_path is None:
            auto_save_path = os.path.join(
                self.cfg.runner.logger.log_path, f"replay_buffer/rank_{self._rank}"
            )
        else:
            auto_save_path = os.path.join(auto_save_path, f"rank_{self._rank}")
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
        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        recv_list = []
        for _ in range(split_num):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            recv_list.append(trajectory)

        intervene_traj_list = []
        for traj in recv_list:
            assert isinstance(traj, Trajectory)
            intervene_trajs = traj.extract_intervene_traj(mode="all")
            if intervene_trajs is not None:
                intervene_traj_list.extend(intervene_trajs)
        if intervene_traj_list:
            self.replay_buffer.add_trajectories(intervene_traj_list)

    def _prepare_mlp_sft_batch(self, batch):
        target_actions = (
            batch["model_action"] if "model_action" in batch else batch["action"]
        )
        return {"states": batch["states"], "action": target_actions}

    def _prepare_openpi_sft_batch(self, batch):
        import jax
        import openpi.models.model as _model

        obs_dict = {}
        obs_prefix_keys = [k for k in batch.keys() if k.startswith("observation/")]
        for key in obs_prefix_keys:
            obs_dict[key] = batch[key]
        if "tokenized_prompt" in batch:
            obs_dict["tokenized_prompt"] = batch["tokenized_prompt"]
        if "tokenized_prompt_mask" in batch:
            obs_dict["tokenized_prompt_mask"] = batch["tokenized_prompt_mask"]

        bsz = batch["action"].shape[0]
        if "model_action" in batch:
            actions = (
                batch["model_action"]
                .reshape(
                    bsz, self.model.config.action_horizon, self.model.config.action_dim
                )
                .clone()
            )
            processed_obs = self.model.input_transform(obs_dict, transpose=False)
            processed_obs = self.model.precision_processor(processed_obs)
            observation = _model.Observation.from_dict(processed_obs)
        else:
            obs_dict["actions"] = batch["action"].reshape(
                bsz, self.model.config.action_chunk, -1
            )
            if obs_dict["actions"].shape[2] < self.model.config.action_dim:
                padding_action_dim = torch.zeros(
                    bsz,
                    obs_dict["actions"].shape[1],
                    self.model.config.action_dim - obs_dict["actions"].shape[2],
                    device=obs_dict["actions"].device,
                )
                obs_dict["actions"] = torch.cat(
                    [obs_dict["actions"], padding_action_dim], dim=2
                )
            if obs_dict["actions"].shape[1] < self.model.config.action_horizon:
                padding_action_chunk = torch.zeros(
                    bsz,
                    self.model.config.action_horizon - obs_dict["actions"].shape[1],
                    self.model.config.action_dim,
                    device=obs_dict["actions"].device,
                )
                obs_dict["actions"] = torch.cat(
                    [obs_dict["actions"], padding_action_chunk], dim=1
                )
            obs_dict["prompt"] = ["empty" for _ in range(bsz)]
            processed_obs = self.model.input_transform(obs_dict, transpose=False)
            if "tokenized_prompt" in batch:
                processed_obs["tokenized_prompt"] = batch["tokenized_prompt"]
            if "tokenized_prompt_mask" in batch:
                processed_obs["tokenized_prompt_mask"] = batch["tokenized_prompt_mask"]
            processed_obs = self.model.precision_processor(processed_obs)
            observation = _model.Observation.from_dict(processed_obs)
            actions = processed_obs["actions"].clone()
            processed_obs.pop("actions")

        observation = jax.tree.map(
            lambda x: torch.as_tensor(x, device=self.device).contiguous().clone(),
            observation,
        )
        return {
            "observation": observation,
            "actions": actions.to(torch.float32).to(self.device),
        }

    def _prepare_sft_batch(self, batch):
        model_type = get_supported_model(self.cfg.actor.model.model_type)
        if model_type == SupportedModel.MLP_POLICY:
            return self._prepare_mlp_sft_batch(batch)
        if model_type == SupportedModel.OPENPI:
            return self._prepare_openpi_sft_batch(batch)
        raise NotImplementedError(
            f"Model type {self.cfg.actor.model.model_type} is not supported for DAgger."
        )

    def _reduce_sft_loss(self, loss):
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
        data = self._prepare_sft_batch(batch)
        actor_loss = self.model(forward_type=ForwardType.SFT, data=data)
        return self._reduce_sft_loss(actor_loss)

    @Worker.timer("update_one_epoch")
    def update_one_epoch(self):
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

    def process_train_metrics(self, metrics):
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
        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)

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

        buffer_load_path = os.path.join(
            load_base_path, f"dagger_components/replay_buffer/rank_{self._rank}"
        )
        self.replay_buffer.load_checkpoint(buffer_load_path)
