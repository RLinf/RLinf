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

import gc
import os

import itertools
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
# import rlinf.algorithms.advantages_sac  # noqa: F401
from rlinf.data.replay_buffer import SACReplayBuffer, concat_batch
from rlinf.data.io_struct.utils import (
    put_tensor_device, 
    split_dict_to_chunk
)
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_rollout_metrics,
)

class EmbodiedSACFSDPActor(EmbodiedFSDPActor):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        
        # SAC-specific initialization
        self.replay_buffer = None
        self.target_model = None
        self.base_alpha = None
        self.demo_buffer = None
        self.alpha_optimizer = None
        self.update_step = 0

    def init_worker(self):
        self.setup_model_and_optimizer(initialize_target=True)
        self.setup_sac_components()
        self.soft_update_target_model(tau=1.0)
        if self.cfg.actor.get("enable_offload", False):
            self.offload_fsdp_param_and_grad()
            self.offload_fsdp_optimizer()
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

    def build_optimizer(self, enable_warmup=False):
        betas = (self._cfg.optim.adam_beta1, self._cfg.optim.adam_beta2)
        params_actor = []
        params_critic = []
        if enable_warmup:
            raise NotImplementedError
        else:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if "q_head" in name:
                        params_critic.append(param)
                    else:
                        params_actor.append(param)
        assert len(params_critic) > 0
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": params_actor, 
                    "lr": self._cfg.optim.lr, 
                    "betas": betas
                },
                
            ]
        )
        self.qf_optimizer = torch.optim.Adam(
            [
                {
                    "params": params_critic,
                    "lr": self._cfg.optim.value_lr,
                    "betas": betas,
                },
            ]
        )
        # Initialize temperature parameter for automatic entropy tuning
        if self.cfg.algorithm.get("auto_entropy_tuning", False):
            target_entropy = self.cfg.algorithm.get(
                "target_entropy", 
                -self.cfg.actor.model.action_dim  # Heuristic: -|A|
            )
            self.target_entropy = target_entropy

            self.alpha_type = "exp" # "exp"
            self.alpha_type = "softplus"
            if self.alpha_type == "exp":
                self.base_alpha = torch.nn.Parameter(
                    np.log(self.cfg.algorithm.get("initial_alpha", 1))
                    * torch.ones(1, device=self.device), requires_grad=True
                )
            elif self.alpha_type == "softplus":
                self.base_alpha = torch.nn.Parameter(
                    np.log(np.exp(self.cfg.algorithm.get("initial_alpha", 0.01))-1)
                    * torch.ones(1, device=self.device), requires_grad=True
                )
            else:
                raise NotImplementedError
            self.alpha_optimizer = torch.optim.Adam(
                [self.base_alpha], 
                lr=self.cfg.algorithm.get("alpha_lr", 3e-4)
            )

        lr_scheduler_type = self._cfg.optim.get("lr_scheduler_type", "constant")
        if lr_scheduler_type == "constant":
            self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.optimizer, factor=1
            )
            self.qf_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.qf_optimizer, factor=1
            )
            if self.cfg.algorithm.get("auto_entropy_tuning", False):
                self.alpha_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                    self.alpha_optimizer, factor=1
                )
        elif lr_scheduler_type == "cosine":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.max_steps, eta_min=1e-6
            )
            self.qf_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.qf_optimizer, T_max=self.max_steps, eta_min=1e-6
            )
            if self.cfg.algorithm.get("auto_entropy_tuning", False):
                self.alpha_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.alpha_optimizer, T_max=self.max_steps, eta_min=1e-6
                )
        else:
            raise NotImplementedError
    
    def compute_alpha(self):
        if self.cfg.algorithm.get("auto_entropy_tuning", False):
            if self.alpha_type == "exp":
                alpha = self.base_alpha.exp()
            elif self.alpha_type == "softplus":
                alpha = torch.nn.functional.softplus(self.base_alpha)
            else:
                raise NotImplementedError
        else:
            alpha = torch.Tensor(self.cfg.algorithm.get("alpha", 0.2)).to(dtype=self.torch_dtype, device=self.device)
        return alpha

    @property
    def alpha(self):
        return self.compute_alpha().item()

    def setup_sac_components(self):
        """Initialize SAC-specific components"""
        # Initialize replay buffer
        buffer_capacity = self.cfg.algorithm.get("replay_buffer_capacity", 100000)
        seed = self.cfg.actor.get("seed", 1234)
        self.replay_buffer = SACReplayBuffer(
            capacity=buffer_capacity,
            device=self.device,
            seed=seed
        )

        self.critic_actor_ratio = self.cfg.algorithm.get("critic_actor_ratio", 1)
        self.critic_subsample_size = self.cfg.algorithm.get("critic_subsample_size", -1)
        self.critic_sample_generator = torch.Generator()
        self.critic_sample_generator.manual_seed(seed)

    def soft_update_target_model(self, tau: float = None):
        """Soft update target model parameters"""
        if tau is None:
            tau = self.cfg.algorithm.get("tau", 0.01)
        
        assert self.target_model_initialized
        
        with torch.no_grad():
            online_params = self.model.named_parameters()
            target_params = self.target_model.named_parameters()
            
            for (name1, online_param), (name2, target_param) in zip(online_params, target_params):
                assert name1 == name2
                if "q_head" not in name1:
                    target_param.data.mul_(0.0)
                    target_param.data.add_(online_param.data)
                else:
                    target_param.data.mul_(1.0 - tau)
                    target_param.data.add_(online_param.data * tau)

    async def recv_rollout_batch(self):
        await super().recv_rollout_batch()
        self.replay_buffer.add_rollout_batch(self.rollout_batch)

    async def recv_demo_data(self):
        demo_data = await self.demo_data_channel.get(async_op=True).async_wait()
        self.demo_buffer = SACReplayBuffer.create_from_buffer(demo_data, seed=self.cfg.actor.seed)

    def compute_advantages_and_returns(self):
        """
        SAC doesn't compute advantages/returns like PPO.
        This method is kept for compatibility but returns empty metrics.
        """
        # SAC uses Q-values directly, no advantages/returns computation needed
        self.log_on_first_rank("SAC algorithm: skipping advantages/returns computation")
        
        # Just compute basic rollout metrics without advantages/returns
        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        return rollout_metrics
    
    def forward_critic(self, batch):
        use_crossq = (self.cfg.algorithm.get("q_head_type", "default") == "crossq")
        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "always")
        agg_q = self.cfg.algorithm.get("agg_q", "min")
        rewards = batch["rewards"].to(self.torch_dtype)
        terminations = batch["terminations"].to(self.torch_dtype)
        dones = batch["dones"].to(self.torch_dtype)

        curr_obs = batch["transitions"]["obs"]
        next_obs = batch["transitions"]["next_obs"]
        with torch.no_grad():
            kwargs = {}
            if self.cfg.actor.model.model_name in ["openvla", "openvla_oft"]:
                kwargs["temperature"] = (
                    self.cfg.algorithm.sampling_params.temperature_train
                )
            next_state_actions, next_state_log_pi, shared_feature = self.model(
                "sac_forward", 
                obs=next_obs, 
                **kwargs
            )
            next_state_log_pi = next_state_log_pi.sum(dim=-1, keepdim=True)
            if not use_crossq:
                all_qf_next_target = self.target_model(
                    "sac_q_forward", 
                    obs=next_obs, actions=next_state_actions, 
                    shared_feature=shared_feature, 
                )
                if self.critic_subsample_size > 0:
                    sample_idx = torch.randint(
                        0, all_qf_next_target.shape[0], self.critic_subsample_size, 
                        generator=self.critic_sample_generator, device=self.device
                    )
                    all_qf_next_target = all_qf_next_target[sample_idx]

                if agg_q == "min":
                    qf_next_target, _ = torch.min(
                        all_qf_next_target, 
                        dim=1, keepdim=True
                    )
                elif agg_q == "mean":
                    qf_next_target = torch.mean(all_qf_next_target, dim=1, keepdim=True)


                if self.cfg.algorithm.get("backup_entropy", True):
                    qf_next_target = qf_next_target - self.alpha * next_state_log_pi
                    qf_next_target = qf_next_target.to(dtype=self.torch_dtype)
                if bootstrap_type == "always":
                    target_q_values = rewards.sum(dim=-1, keepdim=True) \
                        + self.cfg.algorithm.gamma * qf_next_target  # [bsz, 1]
                elif bootstrap_type == "standard":
                    target_q_values = rewards.sum(dim=-1, keepdim=True) \
                        + (~(terminations.any(dim=-1, keepdim=True))) * self.cfg.algorithm.gamma * qf_next_target  # [bsz, 1]
                else:
                    raise NotImplementedError(f"{bootstrap_type=} is not supported!")

        if not use_crossq:
            all_data_q_values = self.model(
                "sac_q_forward", 
                obs=curr_obs, 
                actions=batch["action"] if "action" in batch else batch["action_tokens"]
            )
        else:
            all_data_q_values, all_qf_next = self.model(
                "crossq_q_forward",
                obs=curr_obs,
                actions=batch["action"] if "action" in batch else batch["action_tokens"],
                next_obs=next_obs,
                next_actions=next_state_actions
            )

            all_qf_next = all_qf_next.detach()
            if agg_q == "min":
                qf_next, _ = torch.min(
                    all_qf_next, 
                    dim=1, keepdim=True
                )
            elif agg_q == "mean":
                qf_next = torch.mean(all_qf_next, dim=1, keepdim=True)
            if self.cfg.algorithm.get("backup_entropy", True):
                    qf_next = qf_next - self.alpha * next_state_log_pi
                    qf_next = qf_next.to(dtype=self.torch_dtype)
            
            if bootstrap_type == "always":
                target_q_values = rewards.sum(dim=-1, keepdim=True) \
                    + self.cfg.algorithm.gamma * qf_next  # [bsz, 1]
            elif bootstrap_type == "standard":
                target_q_values = rewards.sum(dim=-1, keepdim=True) \
                    + (~(terminations.any(dim=-1, keepdim=True))) * self.cfg.algorithm.gamma * qf_next  # [bsz, 1]
            else:
                raise NotImplementedError(f"{bootstrap_type=} is not supported!")

        critic_loss = F.mse_loss(
            all_data_q_values, 
            target_q_values.expand_as(all_data_q_values)
        )
        return critic_loss
    
    def forward_actor(self, batch):
        use_crossq = (self.cfg.algorithm.get("q_head_type", "default") == "crossq")
        agg_q = self.cfg.algorithm.get("agg_q", "min")
        curr_obs = batch["transitions"]["obs"]
        kwargs = {}
        if self.cfg.actor.model.model_name in ["openvla", "openvla_oft"]:
            kwargs["temperature"] = (
                self.cfg.algorithm.sampling_params.temperature_train
            )
        pi, log_pi, shared_feature = self.model(
            "sac_forward", obs=curr_obs, **kwargs
        )
        log_pi = log_pi.sum(dim=-1, keepdim=True) # sum over the chunk dimension
        if not use_crossq:
            all_qf_pi = self.model(
                "sac_q_forward", 
                obs=curr_obs, actions=pi, 
                shared_feature=shared_feature, 
                detach_encoder=True
            )
        else:
            all_qf_pi, _ = self.model(
                "crossq_q_forward",
                obs=curr_obs,
                actions=pi,
                next_obs=None,
                next_actions=None,
                shared_feature=shared_feature,
                detach_encoder=True
            )

        if agg_q == "min":
            qf_pi, _ = torch.min(all_qf_pi, dim=1, keepdim=True)
        elif agg_q == "mean":
            qf_pi = torch.mean(all_qf_pi, dim=1, keepdim=True)
        actor_loss = ((self.alpha*log_pi) - qf_pi).mean()

        entropy = -log_pi.mean()
        return actor_loss, entropy
    
    def forward_alpha(self, batch):
        curr_obs = batch["transitions"]["obs"]
        with torch.no_grad():
            kwargs = {}
            if self.cfg.actor.model.model_name in ["openvla", "openvla_oft"]:
                kwargs["temperature"] = (
                    self.cfg.algorithm.sampling_params.temperature_train
                )
            _, log_pi, _ = self.model(
                "sac_forward", obs=curr_obs, **kwargs
            )
            log_pi = log_pi.sum(dim=-1, keepdim=True)

        alpha = self.compute_alpha()
        alpha_loss = (-alpha * (log_pi.mean() + self.target_entropy))
        return alpha_loss
        

    def run_training(self):
        """SAC training using replay buffer"""
        if self.cfg.actor.get("enable_offload", False):
            self.load_fsdp_param_and_grad(self.device)
            self.load_fsdp_optimizer(self.device)

        # Check if replay buffer has enough samples
        min_buffer_size = self.cfg.algorithm.get("min_buffer_size", 100) // self._world_size
        train_actor_steps = self.cfg.algorithm.get("train_actor_steps", 0) // self._world_size
        train_actor_steps = max(min_buffer_size, train_actor_steps)

        if not self.replay_buffer.is_ready(min_buffer_size):
            self.log_on_first_rank(f"Replay buffer size {len(self.replay_buffer)} < {min_buffer_size}, skipping training")
            return {}
        train_actor = self.replay_buffer.is_ready(train_actor_steps)

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
        
        # Number of gradient updates per training call
        num_updates = self.cfg.algorithm.get("num_updates_per_step", 1)
        
        global_batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size

        for update_idx in range(num_updates):
            # Sample batch from replay buffer

            if self.demo_buffer is not None:
                replay_batch = self.replay_buffer.sample(global_batch_size_per_rank // 2)
                demo_batch = self.demo_buffer.sample(global_batch_size_per_rank // 2)
                global_batch = concat_batch(replay_batch, demo_batch)
            else:
                global_batch = self.replay_buffer.sample(global_batch_size_per_rank)
            

            train_micro_batch_list = split_dict_to_chunk(
                global_batch, 
                global_batch_size_per_rank // self.cfg.actor.micro_batch_size
            )

            self.qf_optimizer.zero_grad()
            gbs_critic_loss = []
            for batch in train_micro_batch_list:
                batch = put_tensor_device(batch, device=self.device)
                critic_loss = self.forward_critic(batch)
                gbs_critic_loss.append(critic_loss.item())
                critic_loss = critic_loss / self.gradient_accumulation
                critic_loss.backward()
            qf_grad_norm = self.model.clip_grad_norm_(
                max_norm=self.cfg.actor.optim.clip_grad
            )
            self.qf_optimizer.step()

            metrics_data = {
                "critic/lr": self.qf_optimizer.param_groups[0]["lr"], 
                "sac/critic_loss": np.mean(gbs_critic_loss), 
                "critic/grad_norm": qf_grad_norm, 
            }

            if update_idx % self.critic_actor_ratio == 0 and train_actor:
                self.optimizer.zero_grad()
                gbs_actor_loss = []
                gbs_entropy = 0
                for batch in train_micro_batch_list:
                    batch = put_tensor_device(batch, device=self.device)
                    actor_loss, entropy = self.forward_actor(batch) 
                    gbs_actor_loss.append(actor_loss.item())
                    actor_loss = actor_loss / self.gradient_accumulation
                    actor_loss.backward()
                    gbs_entropy += entropy.item() / self.gradient_accumulation
                actor_grad_norm = self.model.clip_grad_norm_(
                    max_norm=self.cfg.actor.optim.clip_grad
                )
                self.optimizer.step()
                    
                # Update temperature parameter if using automatic entropy tuning
                if hasattr(self, 'base_alpha') and self.base_alpha is not None:
                    self.alpha_optimizer.zero_grad()
                    gbs_alpha_loss = []
                    for batch in train_micro_batch_list:
                        batch = put_tensor_device(batch, device=self.device)
                        alpha_loss = self.forward_alpha(batch)
                        gbs_alpha_loss.append(alpha_loss.item())
                        alpha_loss = alpha_loss / self.gradient_accumulation
                        alpha_loss.backward()
                    torch.distributed.all_reduce(self.base_alpha.grad, op=torch.distributed.ReduceOp.AVG)
                    alpha_grad_norm = torch.nn.utils.clip_grad_norm_(self.base_alpha, self.cfg.actor.optim.clip_grad)
                    self.alpha_optimizer.step()

                self.lr_scheduler.step()
                self.qf_lr_scheduler.step()
                if hasattr(self, 'base_alpha') and self.base_alpha is not None:
                    self.alpha_lr_scheduler.step()
                # Collect metrics
                metrics.update({
                    "sac/alpha": self.alpha, 
                    "sac/actor_loss": np.mean(gbs_actor_loss),
                    "actor/lr": self.optimizer.param_groups[0]["lr"],
                    "sac/alpha_loss": np.mean(gbs_alpha_loss), 
                    "actor/grad_norm": actor_grad_norm, 
                    "alpha/grad_norm": alpha_grad_norm, 
                    "actor/entropy": entropy, 
                    "replay_buffer/size": len(self.replay_buffer),
                    "replay_buffer/utilization": len(self.replay_buffer) / self.replay_buffer.capacity
                })
                
                
            append_to_dict(metrics, metrics_data)

            # # Soft update target network
            if self.target_model_initialized and self.update_step % self.cfg.algorithm.get("target_update_freq", 1) == 0:
                self.soft_update_target_model()    
            
            self.update_step += 1

        # Average metrics across updates
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

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return mean_metric_dict

    def save_checkpoint(self, save_base_path, step):
        super().save_checkpoint(save_base_path, step)
        buffer_path = os.path.join(self.cfg.runner.logger.log_path, f"replay_buffer_{self._rank}.pkl")
        self.replay_buffer.save(buffer_path)