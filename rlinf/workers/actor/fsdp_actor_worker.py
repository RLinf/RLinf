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
from omegaconf import DictConfig
from torch.distributed.device_mesh import init_device_mesh
from tqdm import tqdm

import rlinf.algorithms  # noqa: F401
# import rlinf.algorithms.advantages_sac  # noqa: F401
from rlinf.algorithms.registry import actor_loss
from rlinf.algorithms.replay_buffer import SACReplayBuffer
from rlinf.algorithms.losses import compute_sac_temperature_loss
from rlinf.hybrid_engines.fsdp.utils import get_fsdp_wrap_policy
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import (
    FSDPModelManager,
)
from rlinf.models import get_model
from rlinf.models.embodiment.model_utils import custom_forward
from rlinf.scheduler import Worker
from rlinf.utils.data_iter_utils import get_iterator_k_split
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_loss_mask,
    compute_rollout_metrics,
    compute_split_num,
)
from rlinf.utils.placement import HybridComponentPlacement


class EmbodiedFSDPActor(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor)

        self.cfg = cfg
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.cuda.current_device()
        world_size = self._world_size
        self.device_mesh = init_device_mesh(
            "cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"]
        )

        self._env_group_name = cfg.env.group_name
        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = HybridComponentPlacement(cfg)
        self._weight_dst_rank_in_rollout = self._rank
        if self._weight_dst_rank_in_rollout >= self._component_placement.get_world_size(
            "rollout"
        ):
            self._weight_dst_rank_in_rollout = None

        self._obs_queue_name = cfg.env.channel.queue_name
        self._action_queue_name = cfg.rollout.channel.queue_name
        self._replay_buffer_name = cfg.actor.channel.queue_name
        # stage_num: default to 2, use for pipeline rollout process
        self.stage_num = cfg.rollout.pipeline_stage_num

        self.channel = self.connect_channel(cfg.actor.channel.name)
        self.channel.create_queue(
            cfg.actor.channel.queue_name, maxsize=cfg.actor.channel.queue_size
        )
        
        # SAC-specific initialization
        self.replay_buffer = None
        self.target_model = None
        self.log_alpha = None
        self.alpha_optimizer = None
        self.update_step = 0

    def init_worker(self):
        self.setup_model_and_optimizer()
        self.setup_sac_components()
        # Initialize target model after main model is set up
        self.initialize_target_model()
        if self.cfg.actor.get("enable_offload", False):
            self.offload_fsdp_param_and_grad()
            self.offload_fsdp_optimizer()
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
    
    def setup_sac_components(self):
        """Initialize SAC-specific components"""
        # Initialize replay buffer
        buffer_capacity = self.cfg.algorithm.get("replay_buffer_capacity", 100000)
        self.replay_buffer = SACReplayBuffer(
            capacity=buffer_capacity,
            device=self.device,
            seed=self.cfg.actor.get("seed", 1234)
        )
        
        # Initialize target model (copy of main model)
        # Note: We need to wait until the main model is set up before creating target model
        self.target_model = None
        self.target_model_initialized = False
        
        # Initialize temperature parameter for automatic entropy tuning
        if self.cfg.algorithm.get("auto_entropy_tuning", True):
            target_entropy = self.cfg.algorithm.get(
                "target_entropy", 
                -self.cfg.actor.model.action_dim  # Heuristic: -|A|
            )
            self.target_entropy = target_entropy
            self.log_alpha = torch.tensor(
                np.log(self.cfg.algorithm.get("initial_alpha", 0.2)),
                requires_grad=True,
                device=self.device
            )
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], 
                lr=self.cfg.algorithm.get("alpha_lr", 3e-4)
            )
        else:
            self.alpha = self.cfg.algorithm.get("alpha", 0.2)
    
    def initialize_target_model(self):
        """Initialize target model after main model is set up"""
        if self.target_model is None and self.model is not None:
            # Create target model with same structure as main model
            from rlinf.models import get_model
            self.target_model = get_model(
                self.cfg.actor.checkpoint_load_path, 
                self.cfg.actor.model
            )
            
            # Add Q-value head to target model (same as main model)
            if hasattr(self.model, 'q_value_head'):
                from rlinf.models.embodiment.modules.q_value_head import DoubleQValueHead
                
                # Get model dimensions
                hidden_size = self.target_model.config.hidden_size if hasattr(self.target_model.config, 'hidden_size') else 4096
                action_dim = self.cfg.actor.model.action_dim
                
                # Add Q-value head to the target model
                self.target_model.q_value_head = DoubleQValueHead(hidden_size, action_dim)
                
                # Ensure Q-value head uses the same dtype as the target model
                model_dtype = next(self.target_model.parameters()).dtype
                self.target_model.q_value_head = self.target_model.q_value_head.to(dtype=model_dtype)
                
                self.log_on_first_rank(f"Added DoubleQValueHead to target model: hidden_size={hidden_size}, action_dim={action_dim}, dtype={model_dtype}")
            
            self.target_model.to(self.device)
            self.target_model.eval()
            
            # Copy parameters from main model to target model
            self.soft_update_target_model(tau=1.0)  # Hard copy initially
            self.target_model_initialized = True
            self.log_on_first_rank("Target model initialized successfully")
    
    def soft_update_target_model(self, tau: float = None):
        """Soft update target model parameters"""
        if tau is None:
            tau = self.cfg.algorithm.get("tau", 0.005)
        
        if not self.target_model_initialized:
            self.log_on_first_rank("Target model not initialized, skipping soft update")
            return
        
        with torch.no_grad():
            # Get state dicts for both models
            main_state_dict = self.model.state_dict()
            target_state_dict = self.target_model.state_dict()
            
            # Only update parameters that exist in both models and have matching shapes
            for name, param in main_state_dict.items():
                if name in target_state_dict:
                    target_param = target_state_dict[name]
                    if param.shape == target_param.shape:
                        target_state_dict[name] = tau * param + (1.0 - tau) * target_param
                    else:
                        self.log_on_first_rank(f"Skipping parameter {name} due to shape mismatch: {param.shape} vs {target_param.shape}")
            
            # Load updated state dict to target model
            self.target_model.load_state_dict(target_state_dict)

    # def model_provider_func(self):
    #     model = get_model(self.cfg.actor.checkpoint_load_path, self.cfg.actor.model)
    #     if model is not None:
    #         return model
    #     return super().model_provider_func()
    def model_provider_func(self):
        model = get_model(self.cfg.actor.checkpoint_load_path, self.cfg.actor.model)
        if model is not None:
            # Add Q-value head for SAC
            from rlinf.models.embodiment.modules.q_value_head import DoubleQValueHead
            
            # Get model dimensions
            hidden_size = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 4096
            action_dim = self.cfg.actor.model.action_dim
            
            # Add Q-value head to the model
            model.q_value_head = DoubleQValueHead(hidden_size, action_dim)
            
            # Ensure Q-value head uses the same dtype as the main model
            model_dtype = next(model.parameters()).dtype
            model.q_value_head = model.q_value_head.to(dtype=model_dtype)
            
            self.log_on_first_rank(f"Added DoubleQValueHead to model: hidden_size={hidden_size}, action_dim={action_dim}, dtype={model_dtype}")
            return model
        return super().model_provider_func()

    # def sync_model_to_rollout(self):
    #     if next(self.model.parameters()).is_cpu:
    #         self.load_fsdp_param_and_grad(self.device)
    #         self.load_fsdp_optimizer(self.device)

    #     state_dict = self.get_model_state_dict()
    #     if self._weight_dst_rank_in_rollout is not None:
    #         self.send(
    #             state_dict, self._rollout_group_name, self._weight_dst_rank_in_rollout
    #         )
    #     if self.cfg.actor.get("enable_offload", False):
    #         self.offload_fsdp_param_and_grad()
    #         self.offload_fsdp_optimizer()
    #         torch.cuda.synchronize()
    #         del state_dict
    #         gc.collect()
    #         torch.cuda.empty_cache()
    def sync_model_to_rollout(self):
        if next(self.model.parameters()).is_cpu:
            self.load_fsdp_param_and_grad(self.device)
            self.load_fsdp_optimizer(self.device)

        # Get full state dict and filter out SAC-specific parameters for rollout worker
        full_state_dict = self.get_model_state_dict()
        
        # Filter out q_value_head parameters as rollout worker doesn't need them
        state_dict = {k: v for k, v in full_state_dict.items() if not k.startswith('q_value_head')}
        
        if self._weight_dst_rank_in_rollout is not None:
            self.send(
                state_dict, self._rollout_group_name, self._weight_dst_rank_in_rollout
            )
        if self.cfg.actor.get("enable_offload", False):
            self.offload_fsdp_param_and_grad()
            self.offload_fsdp_optimizer()
            torch.cuda.synchronize()
            del state_dict
            del full_state_dict
            gc.collect()
            torch.cuda.empty_cache()

    async def recv_rollout_batch(self):
        send_num = self._component_placement.get_world_size("rollout") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        self.rollout_batch = {}
        recv_list = []
        for i in range(split_num):
            recv_list.append(
                await self.channel.get(
                    queue_name=self._replay_buffer_name, async_op=True
                ).async_wait()
            )

        # shape [num_chunk, bsz, chunk_size], cat dim 1
        for key in recv_list[0].keys():
            if "env_info/" not in key:
                self.rollout_batch[key] = torch.cat(
                    [recv_list[i][key] for i in range(split_num)], dim=1
                )
            else:
                self.rollout_batch[key] = torch.cat(
                    [recv_list[i][key] for i in range(split_num)], dim=0
                )

        self.rollout_batch = self._process_received_rollout_batch(self.rollout_batch)
        
        # Add transitions to replay buffer
        self.add_to_replay_buffer(self.rollout_batch)
    
    def add_to_replay_buffer(self, rollout_batch):
        """Add rollout data to replay buffer as transitions"""
        n_steps, batch_size, num_chunks = rollout_batch["rewards"].shape
        
        for step in range(n_steps - 1):  # -1 because we need next observations
            for batch_idx in range(batch_size):
                for chunk_idx in range(num_chunks):
                    # Current observations
                    observations = {
                        'input_ids': rollout_batch["input_ids"][step, batch_idx],
                        'pixel_values': rollout_batch["pixel_values"][step, batch_idx],
                        'attention_mask': rollout_batch["attention_mask"][step, batch_idx]
                    }
                    
                    # Next observations
                    next_observations = {
                        'input_ids': rollout_batch["input_ids"][step + 1, batch_idx],
                        'pixel_values': rollout_batch["pixel_values"][step + 1, batch_idx],
                        'attention_mask': rollout_batch["attention_mask"][step + 1, batch_idx]
                    }
                    
                    # Create transition
                    transition = {
                        'observations': observations,
                        'actions': rollout_batch["action_tokens"][step, batch_idx, chunk_idx],
                        'rewards': rollout_batch["rewards"][step, batch_idx, chunk_idx],
                        'next_observations': next_observations,
                        'dones': rollout_batch["dones"][step + 1, batch_idx, chunk_idx],
                        'logprobs': rollout_batch.get("prev_logprobs", torch.zeros_like(rollout_batch["rewards"]))[step, batch_idx, chunk_idx]
                    }
                    self.replay_buffer.add(transition)

    def _process_received_rollout_batch(self, rollout_batch):
        """
        original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
        target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
        """
        rollout_epoch = self.cfg.algorithm.rollout_epoch
        for key, value in rollout_batch.items():
            new_value = value.reshape(
                rollout_epoch, -1, *value.shape[1:]
            )  # [rollout_epoch, n_chunk_step, bsz, ...]
            new_value = new_value.transpose(
                0, 1
            )  # [n_chunk_step, rollout_epoch, bsz, ...]
            new_value = new_value.reshape(new_value.shape[0], -1, *new_value.shape[3:])
            rollout_batch[key] = new_value

        if (
            not self.cfg.env.train.auto_reset
            and not self.cfg.env.train.ignore_terminations
        ):
            dones = rollout_batch[
                "dones"
            ]  # [n_chunk_step, rollout_epoch x bsz, num_action_chunks]
            loss_mask, loss_mask_sum = compute_loss_mask(dones)

            if self.cfg.algorithm.reward_type == "chunk_level":
                loss_mask = loss_mask.any(dim=-1, keepdim=True)
                loss_mask_sum = loss_mask_sum[..., -1:]

            rollout_batch["loss_mask"] = loss_mask
            rollout_batch["loss_mask_sum"] = loss_mask_sum

        # filter data by rewards
        if self.cfg.algorithm.get("filter_rewards", False):
            rewards = rollout_batch[
                "rewards"
            ]  # [n_chunk_step, batch, num_action_chunks]
            if self.rollout_batch.get("loss_mask", None) is not None:
                rewards = rewards * rollout_batch["loss_mask"]
            n_chunk_step, batch_size, num_action_chunks = rewards.shape

            group_size = self.cfg.algorithm.group_size
            assert batch_size % group_size == 0, (
                f"batch {batch_size} not divisible by group_size {group_size}"
            )
            n_prompts = batch_size // group_size

            # calculate rewards by prompt
            rewards = rewards.transpose(
                0, 1
            )  # [batch, n_chunk_step, num_action_chunks]
            rewards = rewards.reshape(rewards.shape[0], -1)  # [batch, n_step]
            reward_matrix = rewards.reshape(
                n_prompts, group_size, rewards.shape[-1]
            )  # [n_prompts, group_size, n_step]
            reward_matrix = reward_matrix.sum(dim=-1)  # [n_prompts, group_size]
            mean_reward_in_group = reward_matrix.mean(dim=1)  # [n_prompts]

            # mask
            reward_filter_mask = (
                mean_reward_in_group >= self.cfg.algorithm.rewards_lower_bound
            ) & (
                mean_reward_in_group <= self.cfg.algorithm.rewards_upper_bound
            )  # [n_prompts]

            # extend mask dimension
            reward_filter_mask = reward_filter_mask.repeat_interleave(
                group_size
            )  # [batch]
            reward_filter_mask = (
                reward_filter_mask.unsqueeze(0).expand(n_chunk_step, -1).unsqueeze(-1)
            )  # [n_chunk_step, batch, 1]

            # update loss_mask
            if self.rollout_batch.get("loss_mask", None) is not None:
                rollout_batch["loss_mask"] = (
                    reward_filter_mask & self.rollout_batch["loss_mask"]
                )
            else:
                rollout_batch["loss_mask"] = reward_filter_mask

        return rollout_batch

    def compute_logprobs(self):
        self.model.eval()
        self.rollout_batch["logprob"] = self.rollout_batch["prev_logprobs"]

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
        batch_size = self.cfg.actor.micro_batch_size
        
        for update_idx in range(num_updates):
            # Sample batch from replay buffer
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
            
            # Forward pass for current observations with Q-network
            current_output = self._forward_model(
                batch['observations'], 
                batch['actions'],
                q_network=True
            )
            
            # Forward pass for next observations using target network
            with torch.no_grad():
                if self.target_model_initialized:
                    next_output = self._forward_target_model(
                        batch['next_observations'],
                        sample_actions=True  # Sample new actions for next state
                    )
                else:
                    # If target model not ready, use main model
                    next_output = self._forward_model(
                        batch['next_observations'],
                        sample_actions=True
                    )
            
            # Compute SAC losses (critic and actor)
            loss_metrics = self._compute_sac_losses(
                current_output, 
                next_output, 
                batch
            )
            
            # Backward pass and optimization
            total_loss = loss_metrics['total_loss']
            self.optimizer.zero_grad()
            total_loss.backward()
            
            grad_norm = self.model.clip_grad_norm_(
                max_norm=self.cfg.actor.optim.clip_grad
            )
            self.optimizer.step()
            
            # Update temperature parameter if using automatic entropy tuning
            if hasattr(self, 'log_alpha') and self.log_alpha is not None:
                alpha_loss, alpha_metrics = compute_sac_temperature_loss(
                    self.log_alpha,
                    current_output['logprobs'],
                    self.target_entropy
                )
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                loss_metrics.update(alpha_metrics)
            
            # Soft update target network
            if self.target_model_initialized and self.update_step % self.cfg.algorithm.get("target_update_freq", 1) == 0:
                self.soft_update_target_model()
            
            # Collect metrics
            loss_metrics.update({
                "sac/total_loss": total_loss.detach().item(),
                "actor/grad_norm": grad_norm.detach().item(),
                "actor/lr": self.optimizer.param_groups[0]["lr"],
                "replay_buffer/size": len(self.replay_buffer),
                "replay_buffer/utilization": len(self.replay_buffer) / self.replay_buffer.capacity
            })
            
            append_to_dict(metrics, loss_metrics)
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
    
    def _forward_model(self, observations, actions=None, q_network=False, sample_actions=False):
        """Forward pass through the model"""
        # 添加调试信息
        self.log_on_first_rank(f"DEBUG: observations type: {type(observations)}")
        self.log_on_first_rank(f"DEBUG: input_ids type: {type(observations['input_ids'])}")
        
        input_ids = observations['input_ids']
        if not isinstance(input_ids, torch.Tensor):
            self.log_on_first_rank(f"ERROR: input_ids is not tensor: {type(input_ids)}")
            return {}
        input_ids = observations['input_ids']
        pixel_values = observations['pixel_values']
        attention_mask = observations['attention_mask']
        
        action_token_len = self.model.action_dim * self.model.num_action_chunks
        
        # Convert action tokens to action features if needed for Q-network
        action_features = None
        if q_network and actions is not None:
            action_features = self._convert_actions_to_features(actions)
        
        if sample_actions:
            # Sample new actions from policy
            logits_processor_args = {
                "vocab_size": self.model.vocab_size,
                "n_action_bins": self.model.config.n_action_bins,
            }
            
            output_dict = custom_forward(
                self.model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                action_token_len=action_token_len,
                q_network=q_network,
                value_head_mode=self.cfg.actor.model.get("vh_mode", "a0"),
                temperature=self.cfg.algorithm.sampling_params.temperature_train,
                top_k=self.cfg.algorithm.sampling_params.top_k,
                logits_processor_args=logits_processor_args,
                action_features=action_features,
                use_gumbel_softmax=self.cfg.algorithm.get("use_gumbel_softmax", True),  # Enable Gumbel-Softmax
                gumbel_temperature=self.cfg.algorithm.get("gumbel_temperature", 1.0),  # Gumbel temperature
            )
        else:
            # Use provided actions
            logits_processor_args = {
                "action_tokens": actions,
                "vocab_size": self.model.vocab_size,
                "n_action_bins": self.model.config.n_action_bins,
            }
            
            output_dict = custom_forward(
                self.model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                action_token_len=action_token_len,
                q_network=q_network,
                value_head_mode=self.cfg.actor.model.get("vh_mode", "a0"),
                temperature=self.cfg.algorithm.sampling_params.temperature_train,
                top_k=self.cfg.algorithm.sampling_params.top_k,
                logits_processor_args=logits_processor_args,
                action_features=action_features,
                use_gumbel_softmax=self.cfg.algorithm.get("use_gumbel_softmax", True),  # Enable Gumbel-Softmax
                gumbel_temperature=self.cfg.algorithm.get("gumbel_temperature", 1.0),  # Gumbel temperature
            )
        
        return output_dict
    
    def _convert_actions_to_features(self, actions):
        """Convert action tokens to features suitable for Q-network input"""
        # Simple conversion: flatten action tokens and convert to model dtype
        batch_size = actions.shape[0]
        model_dtype = next(self.model.parameters()).dtype
        
        # Convert to float first, then to model dtype
        action_features = actions.view(batch_size, -1).float().to(dtype=model_dtype)
        
        # Normalize action features to a reasonable range
        # Ensure n_action_bins is also in the correct dtype
        n_action_bins = torch.tensor(self.model.config.n_action_bins, dtype=model_dtype, device=actions.device)
        action_features = action_features / n_action_bins
        
        return action_features
    
    # def _forward_target_model(self, observations, sample_actions=True):
    #     """Forward pass through target model"""
    #     input_ids = observations['input_ids']
    #     pixel_values = observations['pixel_values']
    #     attention_mask = observations['attention_mask']
        
    #     action_token_len = self.target_model.action_dim * self.target_model.num_action_chunks
        
    #     logits_processor_args = {
    #         "vocab_size": self.target_model.vocab_size,
    #         "n_action_bins": self.target_model.config.n_action_bins,
    #     }
        
    #     # For target model, we sample actions and then compute Q-values
    #     output_dict = custom_forward(
    #         self.target_model,
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         pixel_values=pixel_values,
    #         action_token_len=action_token_len,
    #         q_network=True,
    #         value_head_mode=self.cfg.actor.model.get("vh_mode", "a0"),
    #         temperature=self.cfg.algorithm.sampling_params.temperature_train,
    #         top_k=self.cfg.algorithm.sampling_params.top_k,
    #         logits_processor_args=logits_processor_args,
    #     )
        
    #     return output_dict
    
    def _forward_target_model(self, observations, sample_actions=True):
        """Forward pass through target model"""
        input_ids = observations['input_ids']
        pixel_values = observations['pixel_values']
        attention_mask = observations['attention_mask']
        
        action_token_len = self.target_model.action_dim * self.target_model.num_action_chunks
        
        if sample_actions:
            # First, sample actions from target model (without Q-network)
            logits_processor_args = {
                "vocab_size": self.target_model.vocab_size,
                "n_action_bins": self.target_model.config.n_action_bins,
            }
            
            # Sample actions without Q-network computation
            # output_dict = custom_forward(
            #     self.target_model,
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     pixel_values=pixel_values,
            #     action_token_len=action_token_len,
            #     q_network=False,  # Don't compute Q-values yet
            #     value_head_mode=self.cfg.actor.model.get("vh_mode", "a0"),
            #     temperature=self.cfg.algorithm.sampling_params.temperature_train,
            #     top_k=self.cfg.algorithm.sampling_params.top_k,
            #     logits_processor_args=logits_processor_args,
            # )
            
            from rlinf.models.embodiment.model_utils import sample_logits_processor

            output_dict = custom_forward(
                self.target_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                action_token_len=action_token_len,
                q_network=False,  # Don't compute Q-values yet
                value_head_mode=self.cfg.actor.model.get("vh_mode", "a0"),
                temperature=self.cfg.algorithm.sampling_params.temperature_train,
                top_k=self.cfg.algorithm.sampling_params.top_k,
                logits_processor=sample_logits_processor,  # Use sampling processor
                logits_processor_args=logits_processor_args,
                use_gumbel_softmax=self.cfg.algorithm.get("use_gumbel_softmax", True),  # Enable Gumbel-Softmax
                gumbel_temperature=self.cfg.algorithm.get("gumbel_temperature", 1.0),  # Gumbel temperature
            )
            
            # Now compute Q-values using the sampled actions
            sampled_actions = output_dict['action_tokens']
            action_features = self._convert_actions_to_features(sampled_actions)
            
            # Forward pass with Q-network using sampled actions
            # Update logits_processor_args to include the sampled action_tokens
            q_logits_processor_args = logits_processor_args.copy()
            q_logits_processor_args["action_tokens"] = sampled_actions
            
            q_output_dict = custom_forward(
                self.target_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                action_token_len=action_token_len,
                q_network=True,  # Now compute Q-values
                value_head_mode=self.cfg.actor.model.get("vh_mode", "a0"),
                temperature=self.cfg.algorithm.sampling_params.temperature_train,
                top_k=self.cfg.algorithm.sampling_params.top_k,
                logits_processor_args=q_logits_processor_args,
                action_features=action_features,
                use_gumbel_softmax=self.cfg.algorithm.get("use_gumbel_softmax", True),  # Enable Gumbel-Softmax
                gumbel_temperature=self.cfg.algorithm.get("gumbel_temperature", 1.0),  # Gumbel temperature
            )
            
            # Merge outputs
            output_dict.update(q_output_dict)
        else:
            # Use provided actions (if any)
            logits_processor_args = {
                "vocab_size": self.target_model.vocab_size,
                "n_action_bins": self.target_model.config.n_action_bins,
            }
            
            output_dict = custom_forward(
                self.target_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                action_token_len=action_token_len,
                q_network=True,
                value_head_mode=self.cfg.actor.model.get("vh_mode", "a0"),
                temperature=self.cfg.algorithm.sampling_params.temperature_train,
                top_k=self.cfg.algorithm.sampling_params.top_k,
                logits_processor_args=logits_processor_args,
                use_gumbel_softmax=self.cfg.algorithm.get("use_gumbel_softmax", True),  # Enable Gumbel-Softmax
                gumbel_temperature=self.cfg.algorithm.get("gumbel_temperature", 1.0),  # Gumbel temperature
            )
        
        return output_dict
    
    def _compute_sac_losses(self, current_output, next_output, batch):
        """Compute SAC actor and critic losses"""
        # Get current alpha value and ensure correct dtype
        model_dtype = next(self.model.parameters()).dtype
        if hasattr(self, 'log_alpha') and self.log_alpha is not None:
            alpha = self.log_alpha.exp().to(dtype=model_dtype)
        else:
            alpha = torch.tensor(self.alpha, dtype=model_dtype, device=self.device)
        
        # Ensure all tensors are in the correct dtype
        gamma = torch.tensor(self.cfg.algorithm.get("gamma", 0.99), dtype=model_dtype, device=self.device)
        
        # Prepare loss arguments for SAC
        kwargs = {
            "loss_type": self.cfg.algorithm.loss_type,
            "q1_values": current_output.get("q1_values"),
            "q2_values": current_output.get("q2_values"),
            "target_q_values": next_output.get("q_values"),  # Use min of target Q1, Q2
            "logprobs": current_output["logprobs"],
            "entropy": current_output["entropy"],
            "rewards": batch["rewards"],
            "dones": batch["dones"],
            "gamma": gamma,
            "alpha": alpha,
        }
        
        # Compute SAC loss
        loss, metrics_data = actor_loss(**kwargs)
        metrics_data["total_loss"] = loss
        
        return metrics_data

    def save_checkpoint(self, save_base_path, step):
        torch.distributed.barrier()
        model_state = self.get_model_state_dict()
        optim_state = self.get_optimizer_state_dict()
        if self._rank == 0:
            os.makedirs(save_base_path, exist_ok=True)
            torch.save(model_state, os.path.join(save_base_path, "model.pt"))
            torch.save(optim_state, os.path.join(save_base_path, "optim.pt"))
        torch.distributed.barrier()
