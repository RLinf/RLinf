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

import asyncio
import numpy as np
import torch
# import rlinf.algorithms.advantages_sac  # noqa: F401
from rlinf.data.replay_buffer import concat_batch
from rlinf.data.io_struct.utils import (
    put_tensor_device, 
    split_dict_to_chunk
)
from rlinf.workers.actor.fsdp_actor_worker_sac import EmbodiedSACFSDPActor
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_split_num
)

class AsyncEmbodiedSACFSDPActor(EmbodiedSACFSDPActor):
    async def start_replay_buffer(self, data_channel):
        send_num = self._component_placement.get_world_size("rollout") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)
        replay_buffer_task = asyncio.create_task(
            self.replay_buffer.run(self.cfg, data_channel=data_channel, split_num=split_num)
        )
        await replay_buffer_task
    
    async def run_training(self):
        """SAC training using replay buffer"""
        if self.cfg.actor.get("enable_offload", False):
            self.load_fsdp_param_and_grad(self.device)
            self.load_fsdp_optimizer(self.device)

        # Check if replay buffer has enough samples
        min_buffer_size = self.cfg.algorithm.get("min_buffer_size", 100) // self._world_size
        train_actor_steps = self.cfg.algorithm.get("train_actor_steps", 0) // self._world_size
        train_actor_steps = max(min_buffer_size, train_actor_steps)

        if not (await self.replay_buffer.is_ready_async(min_buffer_size)):
            self.log_on_first_rank(f"Replay buffer size {len(self.replay_buffer)} < {min_buffer_size}, skipping training")
            return False
        train_actor = await self.replay_buffer.is_ready_async(train_actor_steps)
        replay_buffer_stats = self.replay_buffer.get_stats()

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
            gbs_critic_loss = 0
            for batch in train_micro_batch_list:
                batch = put_tensor_device(batch, device=self.device)
                critic_loss = self.forward_critic(batch) / self.gradient_accumulation
                critic_loss.backward()
                gbs_critic_loss += critic_loss.item()
            self.qf_optimizer.step()

            if update_idx % self.critic_actor_ratio == 0 and train_actor:
                self.optimizer.zero_grad()
                gbs_actor_loss = 0
                for batch in train_micro_batch_list:
                    batch = put_tensor_device(batch, device=self.device)
                    actor_loss = self.forward_actor(batch) / self.gradient_accumulation
                    actor_loss.backward()
                    gbs_actor_loss += actor_loss.item()
                self.optimizer.step()
                    
                    # Update temperature parameter if using automatic entropy tuning
                if hasattr(self, 'base_alpha') and self.base_alpha is not None:
                    self.alpha_optimizer.zero_grad()
                    gbs_actor_loss = 0
                    for batch in train_micro_batch_list:
                        batch = put_tensor_device(batch, device=self.device)
                        alpha_loss = self.forward_alpha(batch) / self.gradient_accumulation
                        alpha_loss.backward()
                        gbs_actor_loss += alpha_loss.item()
                    torch.distributed.all_reduce(self.base_alpha.grad, op=torch.distributed.ReduceOp.AVG)
                    self.alpha_optimizer.step()
                    
                
                loss = gbs_actor_loss + gbs_critic_loss
                # Collect metrics
                metrics_data = {
                    "sac/total_loss": loss,
                    "sac/alpha": self.alpha, 
                    # "actor/grad_norm": grad_norm.detach().item(),
                    "actor/lr": self.optimizer.param_groups[0]["lr"],
                    "critic/lr": self.qf_optimizer.param_groups[0]["lr"], 
                    "sac/actor_loss": gbs_actor_loss, 
                    "sac/critic_loss": gbs_critic_loss, 
                    # "sac/qf_values": all_data_q_values.mean().detach().item(), 
                    # "sac/current_q": min_qf_pi.mean().detach().item(), 
                }
                
                append_to_dict(metrics, metrics_data)

            # # Soft update target network
            if self.target_model_initialized and self.update_step % self.cfg.algorithm.get("target_update_freq", 1) == 0:
                self.soft_update_target_model()    
            
            self.update_step += 1

        # self.lr_scheduler.step()
        # self.qf_lr_scheduler.step()
        # if hasattr(self, 'base_alpha') and self.base_alpha is not None:
        #     self.alpha_lr_scheduler.step()

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
        metric_dict = {
            "train": mean_metric_dict, 
            "replay_buffer": replay_buffer_stats
        }

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return metric_dict