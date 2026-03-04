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

import jax
import numpy as np
import openpi.models.model as _model
import torch
from omegaconf import DictConfig

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Channel, Worker
from rlinf.utils import drq
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_split_num,
)
from rlinf.utils.nested_dict_process import (
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


class EmbodiedDAGGERFSDPPolicy(EmbodiedFSDPActor):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # SAC-specific initialization
        self.replay_buffer = None
        # self.target_model = None
        # self.entropy_temp = None
        # self.demo_buffer = None
        # self.alpha_optimizer = None
        self.update_step = 0
        self.enable_drq = bool(getattr(self.cfg.actor, "enable_drq", False))

    def init_worker(self):
        super().setup_model_and_optimizer()
        self.setup_dagger_components()
        # self.soft_update_target_model(tau=1.0)
        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()
        # TODO: init dataloader
        self._setup_rollout_weight_dst_ranks()
        if self.cfg.actor.get("compile_model", False):
            self.model = torch.compile(
                self.model, mode="default"
            )  # max-autotune-no-cudagraphs
            # self.target_model = torch.compile(self.target_model, mode="default")

    # def setup_model_and_optimizer(self, initialize_target=False) -> None:
    #     """Setup model, lr_scheduler, optimizer and grad_scaler."""
    #     """Add initializing target model logic."""
    #     module = self.model_provider_func()
    #     if initialize_target:
    #         target_module = self.model_provider_func()

    #     # Enable gradient checkpointing if configured
    #     if self.cfg.actor.model.get("gradient_checkpointing", False):
    #         self.logger.info("[FSDP] Enabling gradient checkpointing")
    #         module.gradient_checkpointing_enable()
    #         if initialize_target:
    #             target_module.gradient_checkpointing_enable()
    #     else:
    #         self.logger.info("[FSDP] Gradient checkpointing is disabled")

    #     # build model, optimizer, lr_scheduler, grad_scaler
    #     self.model = self._strategy.wrap_model(
    #         model=module, device_mesh=self._device_mesh
    #     )
    #     if initialize_target:
    #         self.target_model = self._strategy.wrap_model(
    #             model=target_module, device_mesh=self._device_mesh
    #         )
    #         self.target_model.requires_grad_(False)
    #         self.target_model_initialized = True

    #     param_filters = {"critic": ["encoders", "encoder", "q_head", "state_proj"]}
    #     filtered_optim_config = {"critic": self.cfg.actor.critic_optim}
    #     optimizers = self.build_optimizers(
    #         model=self.model,
    #         main_optim_config=self.cfg.actor.optim,
    #         param_filters=param_filters,
    #         filtered_optim_config=filtered_optim_config,
    #     )
    #     self.optimizer = optimizers[0]
    #     self.qf_optimizer = optimizers[1]

    #     # SAC alpha
    #     # Initialize temperature parameter for automatic entropy tuning
    #     alpha_type = self.cfg.algorithm.entropy_tuning.get(
    #         "alpha_type", "softplus"
    #     )  # supported type: ["softplus","exp","fixed_alpha"]
    #     self.entropy_temp = EntropyTemperature(
    #         initial_alpha=self.cfg.algorithm.entropy_tuning.get("initial_alpha", 0.01),
    #         alpha_type=alpha_type,
    #         device=self.device,
    #         dtype=self.torch_dtype,
    #     )
    #     if alpha_type != "fixed_alpha":
    #         self.target_entropy = self.cfg.algorithm.entropy_tuning.get(
    #             "target_entropy",
    #             -self.cfg.actor.model.action_dim,
    #         )

    #         self.alpha_optimizer = torch.optim.Adam(
    #             self.entropy_temp.parameters(),
    #             lr=self.cfg.algorithm.entropy_tuning.optim.lr,
    #         )

    #     self.build_lr_schedulers()

    #     self.grad_scaler = self.build_grad_scaler(
    #         self.cfg.actor.fsdp_config.amp.use_grad_scaler
    #     )

    def build_lr_schedulers(self):
        self.lr_scheduler = self.build_lr_scheduler(
            self.optimizer, self.cfg.actor.optim
        )
        self.qf_lr_scheduler = self.build_lr_scheduler(
            self.qf_optimizer, self.cfg.actor.critic_optim
        )
        if self.alpha_optimizer is not None:
            self.alpha_lr_scheduler = self.build_lr_scheduler(
                self.optimizer, self.cfg.algorithm.entropy_tuning.optim
            )

    def setup_dagger_components(self):
        """Initialize DAGGER-specific components"""
        # Initialize replay buffer
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

        # if self.cfg.algorithm.get("demo_buffer", None) is not None:
        #     auto_save_path = self.cfg.algorithm.demo_buffer.get("auto_save_path", None)
        #     if auto_save_path is None:
        #         auto_save_path = os.path.join(
        #             self.cfg.runner.logger.log_path, f"demo_buffer/rank_{self._rank}"
        #         )
        #     else:
        #         auto_save_path = os.path.join(auto_save_path, f"rank_{self._rank}")
        #     self.demo_buffer = TrajectoryReplayBuffer(
        #         seed=seed,
        #         enable_cache=self.cfg.algorithm.demo_buffer.enable_cache,
        #         cache_size=self.cfg.algorithm.demo_buffer.cache_size,
        #         sample_window_size=self.cfg.algorithm.demo_buffer.sample_window_size,
        #         auto_save=self.cfg.algorithm.demo_buffer.get("auto_save", False),
        #         auto_save_path=auto_save_path,
        #         trajectory_format="pt",
        #     )
        #     if self.cfg.algorithm.demo_buffer.get("load_path", None) is not None:
        #         self.demo_buffer.load_checkpoint(
        #             self.cfg.algorithm.demo_buffer.load_path,
        #             is_distributed=True,
        #             local_rank=self._rank,
        #             world_size=self._world_size,
        #         )

        # self.critic_actor_ratio = self.cfg.algorithm.get("critic_actor_ratio", 1)
        # self.critic_subsample_size = self.cfg.algorithm.get("critic_subsample_size", -1)
        # self.critic_sample_generator = torch.Generator(self.device)
        # self.critic_sample_generator.manual_seed(seed)

        # self.target_update_type = self.cfg.algorithm.get("target_update_type", "all")
        # assert self.target_update_type in ["all", "q_head_only"], (
        #     f"{self.target_update_type=} is not suppported!"
        # )

    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        """
        Receive rollout trajectories from rollout workers.

        Args:
            input_channel: The input channel to read from.
        """
        send_num = self._component_placement.get_world_size("rollout") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        recv_list = []

        for _ in range(split_num):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            recv_list.append(trajectory)

        intervene_traj_list = []
        for traj in recv_list:
            assert isinstance(traj, Trajectory)
            intervene_traj = traj.extract_intervene_traj(mode="all")
            if intervene_traj is not None:
                intervene_traj_list.extend(intervene_traj)
        if len(intervene_traj_list) > 0:
            self.replay_buffer.add_trajectories(intervene_traj_list)

    @Worker.timer("forward_actor")
    def forward_actor(self, batch):
        # use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"
        # if "actor_agg_q" in self.cfg.algorithm:
        #     agg_q = self.cfg.algorithm["actor_agg_q"]
        # else:
        #     agg_q = self.cfg.algorithm.get("agg_q", "min")

        # kwargs = {}
        # if self.cfg.actor.model.model_type in ["openvla", "openvla_oft"]:
        #     kwargs["temperature"] = self.cfg.algorithm.sampling_params.temperature_train
        obs_dict = {}
        obs_prefix_keys = [k for k in batch.keys() if k.startswith("observation/")]
        for key in obs_prefix_keys:
            obs_dict[key] = batch[key]
        # Also extract tokenized prompt fields if present
        if "tokenized_prompt" in batch:
            obs_dict["tokenized_prompt"] = batch["tokenized_prompt"]
        if "tokenized_prompt_mask" in batch:
            obs_dict["tokenized_prompt_mask"] = batch["tokenized_prompt_mask"]

        bsz = batch["action"].shape[0]

        if "model_action" in batch:
            # Expert-model path: model_action is already in model/normalized space
            # [B, action_horizon * action_dim].  Use it directly as the imitation target
            # without re-applying input_transform on the action (which would double-normalize).
            # We still run input_transform on the obs to get correctly normalized observations;
            # a zero placeholder is used for the "actions" slot so the transform pipeline
            # does not see env-space data.
            actions = (
                batch["model_action"]
                .reshape(
                    bsz, self.model.config.action_horizon, self.model.config.action_dim
                )
                .clone()
            )
            processed_obs = self.model.input_transform(obs_dict, transpose=False)
            processed_obs = self.model.precision_processor(
                processed_obs
            )  # obs precision processor
            observation = _model.Observation.from_dict(processed_obs)
        else:
            # Human-intervene path: action is in env space [B, action_chunk * action_env_dim].
            # Reshape, pad to model dims, then apply input_transform to normalize into model space.
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
            processed_obs["tokenized_prompt"] = batch["tokenized_prompt"]
            processed_obs["tokenized_prompt_mask"] = batch["tokenized_prompt_mask"]
            processed_obs = self.model.precision_processor(
                processed_obs
            )  # obs precision processor
            observation = _model.Observation.from_dict(processed_obs)
            actions = processed_obs["actions"].clone()
            processed_obs.pop("actions")

        observation = jax.tree.map(
            lambda x: torch.as_tensor(x, device=self.device).contiguous().clone(),
            observation,
        )
        actions = actions.to(torch.float32)
        actions = actions.to(self.device)

        actor_loss = self.model(
            forward_type=ForwardType.SFT,
            data={"observation": observation, "actions": actions},
        )

        return actor_loss.mean()

    @Worker.timer("update_one_epoch")
    def update_one_epoch(self):
        global_batch_size_per_rank = (
            self.cfg.actor.global_batch_size // self._world_size
        )

        with self.worker_timer("sample"):
            # Sample batch from replay buffer
            # TODO: sample from dataset & buffer
            global_batch = self.replay_buffer.sample(
                num_chunks=global_batch_size_per_rank
            )

        train_micro_batch_list = split_dict_to_chunk(
            global_batch,
            global_batch_size_per_rank // self.cfg.actor.micro_batch_size,
        )

        # move train_micro_batch_list to device and apply DRQ for critic/actor/alpha passes
        for i, batch in enumerate(train_micro_batch_list):
            batch = put_tensor_device(batch, device=self.device)
            if self.enable_drq:
                drq.apply_drq(batch["curr_obs"], pad=4)
                drq.apply_drq(batch["next_obs"], pad=4)
            train_micro_batch_list[i] = batch

        # self.qf_optimizer.zero_grad()
        # gbs_critic_loss = []
        # all_critic_metrics = {}
        # for batch in train_micro_batch_list:
        #     critic_loss, critic_metrics = self.forward_critic(batch)
        #     critic_loss = critic_loss / self.gradient_accumulation
        #     critic_loss.backward()
        #     gbs_critic_loss.append(critic_loss.item() * self.gradient_accumulation)
        #     append_to_dict(all_critic_metrics, critic_metrics)
        # all_critic_metrics = {
        #     f"critic/{key}": np.mean(value) for key, value in all_critic_metrics.items()
        # }
        # qf_grad_norm = self.model.clip_grad_norm_(
        #     max_norm=self.cfg.actor.critic_optim.clip_grad
        # )

        # self.qf_optimizer.step()
        # self.qf_lr_scheduler.step()

        # metrics_data = {
        #     "sac/critic_loss": np.mean(gbs_critic_loss),
        #     "critic/lr": self.qf_optimizer.param_groups[0]["lr"],
        #     "critic/grad_norm": qf_grad_norm,
        #     **all_critic_metrics,
        # }

        self.optimizer.zero_grad()
        gbs_actor_loss = []
        # all_actor_metrics = {}
        for batch in train_micro_batch_list:
            actor_loss = self.forward_actor(batch["forward_inputs"])
            actor_loss = actor_loss / self.gradient_accumulation
            actor_loss.backward()
            gbs_actor_loss.append(actor_loss.item() * self.gradient_accumulation)
        # all_actor_metrics = {
        #     f"actor/{key}": np.mean(value)
        #     for key, value in all_actor_metrics.items()
        # }
        actor_grad_norm = self.model.clip_grad_norm_(
            max_norm=self.cfg.actor.optim.clip_grad
        )
        self.optimizer.step()
        self.lr_scheduler.step()

        # Update temperature parameter if using automatic entropy tuning
        # gbs_alpha_loss = [0]
        # alpha_grad_norm = 0
        # if self.alpha_optimizer is not None:
        #     self.alpha_optimizer.zero_grad()
        #     gbs_alpha_loss = []
        #     for batch in train_micro_batch_list:
        #         alpha_loss = self.forward_alpha(batch) / self.gradient_accumulation
        #         alpha_loss.backward()
        #         gbs_alpha_loss.append(
        #             alpha_loss.item() * self.gradient_accumulation
        #         )
        #     torch.distributed.all_reduce(
        #         self.entropy_temp.base_alpha.grad, op=torch.distributed.ReduceOp.AVG
        #     )
        #     alpha_grad_norm = torch.nn.utils.clip_grad_norm_(
        #         self.entropy_temp.base_alpha,
        #         self.cfg.algorithm.entropy_tuning.optim.clip_grad,
        #     )
        #     self.alpha_optimizer.step()
        #     self.alpha_lr_scheduler.step()

        # Collect metrics
        metrics_data = {
            "dagger/actor_loss": np.mean(gbs_actor_loss),
            "actor/lr": self.optimizer.param_groups[0]["lr"],
            "actor/grad_norm": actor_grad_norm,
        }
        # Soft update target network
        # if (
        #     self.target_model_initialized
        #     and self.update_step % self.cfg.algorithm.get("target_update_freq", 1) == 0
        # ):
        #     self.soft_update_target_model()

        return metrics_data

    def process_train_metrics(self, metrics):
        replay_buffer_stats = self.replay_buffer.get_stats()
        replay_buffer_stats = {
            f"replay_buffer/{key}": value for key, value in replay_buffer_stats.items()
        }
        append_to_dict(metrics, replay_buffer_stats)

        mean_metric_dict = {}
        for key, value in metrics.items():
            if isinstance(value, list) and len(value) > 0:
                # Convert tensor values to CPU and detach before computing mean
                cpu_values = []
                for v in value:
                    if isinstance(v, torch.Tensor):
                        cpu_values.append(v.detach().cpu().item())
                    else:
                        cpu_values.append(v)
                mean_metric_dict[key] = np.mean(cpu_values)
            else:
                # Handle single values
                if isinstance(value, torch.Tensor):
                    mean_metric_dict[key] = value.detach().cpu().item()
                else:
                    mean_metric_dict[key] = value

        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )
        return mean_metric_dict

    @Worker.timer("run_training")
    def run_training(self):
        """DAGGER training using replay buffer"""
        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)

        # Check if replay buffer has enough samples
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

        mean_metric_dict = self.process_train_metrics(metrics)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return mean_metric_dict

    def compute_advantages_and_returns(self):
        """
        SAC doesn't compute advantages/returns like PPO.
        This method is kept for compatibility but returns empty metrics.
        """
        return {}

    def save_checkpoint(self, save_base_path, step):
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
            self.is_weight_offloaded = False
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)
            self.is_optimizer_offloaded = False

        # Save model
        self._strategy.save_checkpoint(
            model=self.model,
            optimizers=[self.optimizer, self.qf_optimizer],
            lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
            save_path=save_base_path,
            checkpoint_format="local_shard"
            if self.cfg.actor.fsdp_config.use_orig_params
            else "dcp",
        )

        # save replay buffer
        buffer_save_path = os.path.join(
            save_base_path, f"sac_components/replay_buffer/rank_{self._rank}"
        )
        self.replay_buffer.save_checkpoint(buffer_save_path)

    def load_checkpoint(self, load_base_path):
        # load model
        self._strategy.load_checkpoint(
            model=self.model,
            optimizers=[self.optimizer, self.qf_optimizer],
            lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
            load_path=load_base_path,
            checkpoint_format="local_shard"
            if self.cfg.actor.fsdp_config.use_orig_params
            else "dcp",
        )

        # load alpha
        if self.alpha_optimizer is not None:
            alpha_load_path = os.path.join(load_base_path, "sac_components/alpha")
            self._strategy.load_checkpoint(
                model=self.entropy_temp,
                optimizers=self.alpha_optimizer,
                lr_schedulers=self.alpha_lr_scheduler,
                load_path=alpha_load_path,
            )

        # load target model
        target_model_load_path = os.path.join(
            load_base_path, "sac_components/target_model"
        )
        target_model_state_dict = torch.load(
            os.path.join(target_model_load_path, f"checkpoint_rank_{self._rank}.pt")
        )
        self._strategy.load_model_with_state_dict(
            self.target_model,
            target_model_state_dict,
            cpu_offload=False,
            full_state_dict=True,
        )

        # load replay buffer
        buffer_load_path = os.path.join(
            load_base_path, f"sac_components/replay_buffer/rank_{self._rank}"
        )
        self.replay_buffer.load_checkpoint(buffer_load_path)
