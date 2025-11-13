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

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
# import rlinf.algorithms.advantages_sac  # noqa: F401
from rlinf.data.replay_buffer import SACReplayBuffer, concat_batch
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
                        print(name)
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
                self.base_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            elif self.alpha_type == "softplus":
                self.base_alpha = torch.nn.Parameter(
                    np.log(np.exp(0.01)-1) * torch.ones(1, device=self.device), requires_grad=True
                )
            else:
                raise NotImplementedError
            self.alpha_optimizer = torch.optim.Adam(
                [self.base_alpha], 
                lr=self.cfg.algorithm.get("alpha_lr", 3e-4)
            )
    
    def compute_alpha(self):
        if self.cfg.algorithm.get("auto_entropy_tuning", False):
            if self.alpha_type == "exp":
                alpha = self.base_alpha.exp()
            elif self.alpha_type == "softplus":
                alpha = torch.nn.functional.softplus(self.base_alpha)
        else:
            alpha = self.cfg.algorithm.get("alpha", 0.2)
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
            online_params = self.model.parameters()
            target_params = self.target_model.parameters()
            
            for online_param, target_param in zip(online_params, target_params):
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

    def run_training(self):
        """SAC training using replay buffer"""
        if self.cfg.actor.get("enable_offload", False):
            self.load_fsdp_param_and_grad(self.device)
            self.load_fsdp_optimizer(self.device)

        # Check if replay buffer has enough samples
        min_buffer_size = self.cfg.algorithm.get("min_buffer_size", 100)
        if not self.replay_buffer.is_ready(min_buffer_size):
            self.log_on_first_rank(f"Replay buffer size {len(self.replay_buffer)} < {min_buffer_size}, skipping training")
            return {}

        self.model.train()
        metrics = {}
        
        # Number of gradient updates per training call
        num_updates = self.cfg.algorithm.get("num_updates_per_step", 1)
        batch_size = self.cfg.actor.global_batch_size
        
        for update_idx in range(num_updates):
            # Sample batch from replay buffer

            if self.demo_buffer is not None:
                replay_batch = self.replay_buffer.sample(batch_size // 2)
                demo_batch = self.replay_buffer.sample(batch_size // 2)
                batch = concat_batch(replay_batch, demo_batch)
            else:
                batch = self.replay_buffer.sample(batch_size)
            
            # Move batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)
                elif isinstance(v, dict):
                    batch[k] = {
                        sub_k: sub_v.to(self.device) if isinstance(sub_v, torch.Tensor) else sub_v
                        for sub_k, sub_v in v.items()
                    }
            
            curr_obs = {
                key[len("transitions/obs/"):]: value for key, value in batch.items()
                if "transitions/obs/" in key
            }
            next_obs = {
                key[len("transitions/next_obs/"):]: value for key, value in batch.items()
                if "transitions/next_obs/" in key
            }

            with torch.no_grad():

                next_state_actions, next_state_log_pi, shared_feature = self.model(
                    "sac_forward", 
                    obs=next_obs, 
                )
                next_state_log_pi = next_state_log_pi.sum(dim=-1, keepdim=True)
                
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
                    
                min_qf_next_target, _ = torch.min(
                    all_qf_next_target, 
                    dim=1, keepdim=True
                )

                if self.cfg.algorithm.get("backup_entropy", True):
                    min_qf_next_target = min_qf_next_target - self.alpha * next_state_log_pi
                target_q_values = batch["rewards"] + self.cfg.algorithm.gamma * min_qf_next_target # [bsz, 1]

            
            all_data_q_values = self.model(
                "sac_q_forward", 
                obs=curr_obs, actions=batch["action"] 
            ) # [num-q, bsz, 1]

            critic_loss = F.mse_loss(
                all_data_q_values, 
                target_q_values.expand_as(all_data_q_values)
            )
            self.qf_optimizer.zero_grad()
            critic_loss.backward()
            self.qf_optimizer.step()

            if update_idx % self.critic_actor_ratio == 0:
                pi, log_pi, shared_feature = self.model(
                    "sac_forward", obs=curr_obs
                )
                log_pi = log_pi.sum(dim=-1, keepdim=True)
                all_qf_pi = self.model(
                    "sac_q_forward", 
                    obs=curr_obs, actions=pi, 
                    shared_feature=shared_feature, 
                    detach_encoder=True
                )
                # mean_qf_pi = torch.mean(all_qf_pi, dim=1, keepdim=True)
                # actor_loss = ((self.alpha*log_pi) - mean_qf_pi).mean()

                min_qf_pi = torch.mean(all_qf_pi, dim=1, keepdim=True)
                actor_loss = ((self.alpha*log_pi) - min_qf_pi).mean()
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                actor_loss.backward()
                self.optimizer.step()
                
                # Update temperature parameter if using automatic entropy tuning
                if hasattr(self, 'base_alpha') and self.base_alpha is not None:
                    with torch.no_grad():
                        _, log_pi, _ = self.model(
                            "sac_forward", obs=curr_obs
                        )
                        log_pi = log_pi.sum(dim=-1, keepdim=True)

                    alpha_loss = (-self.compute_alpha() * (log_pi.mean() + self.target_entropy))

                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    torch.distributed.all_reduce(self.base_alpha.grad, op=torch.distributed.ReduceOp.AVG)
                    self.alpha_optimizer.step()
            
            # # Soft update target network
            if self.target_model_initialized and self.update_step % self.cfg.algorithm.get("target_update_freq", 1) == 0:
                self.soft_update_target_model()
                
            loss = actor_loss + critic_loss
            # Collect metrics
            metrics_data = {
                "sac/total_loss": loss.detach().item(),
                "sac/alpha": self.alpha, 
                # "actor/grad_norm": grad_norm.detach().item(),
                "actor/lr": self.optimizer.param_groups[0]["lr"],
                "critic/lr": self.qf_optimizer.param_groups[0]["lr"], 
                "sac/actor_loss": actor_loss.detach().item(), 
                "sac/critic_loss": critic_loss.detach().item(), 
                "sac/qf_values": all_data_q_values.mean().detach().item(), 
                "sac/current_q": min_qf_pi.mean().detach().item(), 
                "replay_buffer/size": len(self.replay_buffer),
                "replay_buffer/utilization": len(self.replay_buffer) / self.replay_buffer.capacity
            }
            
            append_to_dict(metrics, metrics_data)
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