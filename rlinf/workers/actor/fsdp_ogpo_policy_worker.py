# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from math import ceil
from typing import Any, Callable

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.algorithms.losses import compute_ppo_actor_loss
from rlinf.data.storage.replay import TrajectoryReplayBuffer
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Channel, Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict, compute_split_num
from rlinf.utils.nested_dict_process import put_tensor_device, split_dict_to_chunk
from rlinf.utils.utils import (
    clear_memory,
    collect_param_names_need_sync,
    seed_everything,
)
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor


class EmbodiedOGPOFSDPPolicy(EmbodiedFSDPActor):
    """Train the ManiSkill state-flow OGPO policy."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.target_model = None
        self.replay_buffer = None
        self.success_buffer = None
        self.qf_optimizer = None
        self.qf_lr_scheduler = None
        self.update_step = 0

    def init_worker(self) -> None:
        seed_everything(int(self.cfg.actor.seed))
        online_module = self.model_provider_func()
        target_module = self.model_provider_func()
        self.param_names_need_sync = collect_param_names_need_sync(target_module)
        self.model = self._strategy.wrap_model(
            model=online_module, device_mesh=self._device_mesh
        )
        self.target_model = self._strategy.wrap_model(
            model=target_module, device_mesh=self._device_mesh
        )
        self.target_model.requires_grad_(False)
        self.target_model.eval()
        if self.torch_dtype is None:
            self.torch_dtype = next(self.model.parameters()).dtype

        self.optimizer, self.qf_optimizer = self.build_optimizers(
            model=self.model,
            main_optim_config=self.cfg.actor.optim,
            param_filters={"critic": ["q_head"]},
            filtered_optim_config={"critic": self.cfg.actor.critic_optim},
        )
        self.lr_scheduler = self.build_lr_scheduler(
            self.optimizer, self.cfg.actor.optim
        )
        self.qf_lr_scheduler = self.build_lr_scheduler(
            self.qf_optimizer, self.cfg.actor.critic_optim
        )
        self.grad_scaler = self.build_grad_scaler(False)
        self.replay_buffer = self._new_buffer("replay_buffer")
        self.success_buffer = self._new_buffer("success_buffer")
        self._update_targets(actor_tau=1.0, critic_tau=1.0)

    def _new_buffer(self, name: str) -> TrajectoryReplayBuffer:
        cfg = self.cfg.algorithm.replay_buffer
        return TrajectoryReplayBuffer(
            seed=int(self.cfg.actor.seed),
            enable_cache=bool(cfg.enable_cache),
            cache_size=int(cfg.cache_size),
            sample_window_size=int(cfg.sample_window_size),
            auto_save=False,
            auto_save_path=os.path.join(self.cfg.runner.logger.log_path, name),
            trajectory_format="pt",
        )

    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        """Receive complete episodes and add their prebuilt training windows."""
        clear_memory(sync=False)
        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        episodes = []
        for _ in range(compute_split_num(send_num, recv_num)):
            episodes.extend(await input_channel.get(async_op=True).async_wait())
        self.replay_buffer.add_trajectories(episodes)
        self.success_buffer.add_trajectories(
            [e for e in episodes if bool(e.forward_inputs["is_success"].any())]
        )

    @torch.no_grad()
    def _update_targets(
        self, actor_tau: float | None = None, critic_tau: float | None = None
    ) -> None:
        actor_tau = float(
            self.cfg.algorithm.ogpo.actor_tau if actor_tau is None else actor_tau
        )
        critic_tau = float(
            self.cfg.algorithm.ogpo.critic_tau if critic_tau is None else critic_tau
        )
        for (online_name, online), (target_name, target) in zip(
            self.model.named_parameters(), self.target_model.named_parameters()
        ):
            if online_name != target_name:
                raise RuntimeError(
                    f"Parameter mismatch: {online_name} != {target_name}"
                )
            tau = critic_tau if "q_head" in online_name else actor_tau
            target.data.lerp_(online.data, tau)

    def get_rollout_state_dict(self) -> dict[str, torch.Tensor]:
        return self._strategy.get_model_state_dict(
            self.target_model, cpu_offload=False, full_state_dict=False
        )

    def _sample_micro_batches(
        self, buffer: TrajectoryReplayBuffer | None = None
    ) -> list[dict[str, Any]]:
        buffer = self.replay_buffer if buffer is None else buffer
        batch_size = int(self.cfg.actor.global_batch_size) // self._world_size
        batch = buffer.sample(num_chunks=batch_size)
        batch["valid"] = batch.pop("forward_inputs")["valid"]
        split_count = max(1, ceil(batch_size / int(self.cfg.actor.micro_batch_size)))
        return [
            put_tensor_device(chunk, device=self.device)
            for chunk in split_dict_to_chunk(batch, split_count)
        ]

    @staticmethod
    def _repeat_obs(
        obs: dict[str, torch.Tensor], repeats: int
    ) -> dict[str, torch.Tensor]:
        return {
            key: value.unsqueeze(0)
            .expand(repeats, *value.shape)
            .reshape(repeats * value.shape[0], *value.shape[1:])
            for key, value in obs.items()
        }

    def _candidate_q(
        self, model, obs: dict[str, torch.Tensor], candidates: torch.Tensor
    ) -> torch.Tensor:
        num_samples, batch_size = candidates.shape[:2]
        return model(
            forward_type=ForwardType.SAC_Q,
            obs=self._repeat_obs(obs, num_samples),
            actions=candidates.reshape(num_samples * batch_size, -1),
        ).reshape(num_samples, batch_size, -1)

    def _sample_actions(self, model, obs, num_samples: int):
        return model(
            forward_type=ForwardType.OGPO_SAMPLE,
            obs=obs,
            num_samples=num_samples,
            noise_std=float(self.cfg.algorithm.ogpo.constant_noise_std),
        )

    @Worker.timer("ogpo_forward_critic")
    def forward_critic(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict]:
        actions = batch["actions"].reshape(batch["actions"].shape[0], -1)
        rewards = batch["rewards"].float()[:, -1, 0]
        terminations = batch["terminations"].bool()[:, -1, 0]
        valid = batch["valid"].bool()[:, -1]
        horizon = int(self.cfg.actor.model.num_action_chunks)

        with torch.no_grad():
            next_actions, _, _ = self._sample_actions(
                self.target_model, batch["next_obs"], 1
            )
            next_q = (
                self.target_model(
                    forward_type=ForwardType.SAC_Q,
                    obs=batch["next_obs"],
                    actions=next_actions[0],
                )
                .float()
                .mean(dim=-1)
            )
            target_q = (
                rewards
                + float(self.cfg.algorithm.gamma) ** horizon * (~terminations) * next_q
            )

        q_values = self.model(
            forward_type=ForwardType.SAC_Q,
            obs=batch["curr_obs"],
            actions=actions,
        ).float()
        target_q = target_q[:, None].expand_as(q_values)
        mask = valid[:, None].expand_as(q_values)
        loss = (q_values - target_q).square()[mask].mean()
        return loss, {
            "critic/total_loss": loss.detach(),
            "critic/q_mean": q_values[mask].mean().detach(),
            "critic/target_q_mean": target_q[mask].mean().detach(),
        }

    def _bc_loss(self, batch_size: int) -> torch.Tensor | None:
        if not self.success_buffer.is_ready(1):
            return None
        batch = self.success_buffer.sample(num_chunks=batch_size)
        valid = batch.pop("forward_inputs")["valid"]
        batch = put_tensor_device(batch, device=self.device)
        valid = valid.to(self.device)
        predicted, target = self.model(
            forward_type=ForwardType.OGPO_BC,
            obs=batch["curr_obs"],
            actions=batch["actions"],
        )
        mask = valid.unsqueeze(-1).expand_as(batch["actions"]).reshape_as(predicted)
        return (predicted - target).square()[mask].mean()

    @Worker.timer("ogpo_forward_actor")
    def forward_actor(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict]:
        batch_size = batch["actions"].shape[0]
        bc_loss = self._bc_loss(batch_size)
        with torch.no_grad():
            candidates, chains, old_logprobs = self._sample_actions(
                self.target_model,
                batch["curr_obs"],
                int(self.cfg.algorithm.group_size),
            )
            q_ensemble = self._candidate_q(
                self.target_model, batch["curr_obs"], candidates
            ).float()
            q_values = q_ensemble.mean(dim=-1)
            per_head = q_ensemble - q_ensemble.mean(dim=0, keepdim=True)
            sorted_advantages = per_head.sort(dim=-1).values
            advantages = torch.where(
                sorted_advantages[..., 0] > 0,
                sorted_advantages[..., 0],
                torch.where(
                    sorted_advantages[..., -1] < 0,
                    sorted_advantages[..., -1],
                    torch.zeros_like(sorted_advantages[..., 0]),
                ),
            )

        logprobs, flow_metrics = self.model(
            forward_type=ForwardType.OGPO_LOG_PROB,
            obs=batch["curr_obs"],
            chains=chains,
            noise_std=float(self.cfg.algorithm.ogpo.constant_noise_std),
        )
        loss, metrics = compute_ppo_actor_loss(
            logprobs=logprobs.float(),
            old_logprobs=old_logprobs.float(),
            advantages=advantages,
            clip_ratio_low=float(self.cfg.algorithm.ogpo.clip_epsilon),
            clip_ratio_high=float(self.cfg.algorithm.ogpo.clip_epsilon),
            loss_mask=torch.ones_like(logprobs, dtype=torch.bool),
        )
        if bc_loss is not None:
            loss = loss + float(self.cfg.algorithm.ogpo.bc_coeff) * bc_loss
            metrics["actor/bc_loss"] = bc_loss.detach()
        metrics.update({f"actor/{key}": value for key, value in flow_metrics.items()})
        metrics["actor/total_loss"] = loss.detach()
        metrics["actor/q_mean"] = q_values.mean().detach()
        metrics["actor/advantage_mean"] = advantages.mean().detach()
        return loss, metrics

    def _update(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler,
        batches: list[dict[str, Any]],
        forward: Callable[[dict[str, Any]], tuple[torch.Tensor, dict]],
        clip_grad: float,
    ) -> dict[str, Any]:
        optimizer.zero_grad(set_to_none=True)
        metrics = {}
        total = sum(batch["actions"].shape[0] for batch in batches)
        for batch in batches:
            loss, batch_metrics = forward(batch)
            weight = batch["actions"].shape[0] / total
            (loss * weight).backward()
            for key, value in batch_metrics.items():
                metrics[key] = metrics.get(key, 0.0) + value * weight
        metrics["grad_norm"] = self.model.clip_grad_norm_(max_norm=clip_grad)
        optimizer.step()
        scheduler.step()
        return metrics

    def update_one_epoch(self) -> dict[str, Any]:
        metrics = {}
        for _ in range(int(self.cfg.algorithm.ogpo.q_utd)):
            values = self._update(
                self.qf_optimizer,
                self.qf_lr_scheduler,
                self._sample_micro_batches(),
                self.forward_critic,
                float(self.cfg.actor.critic_optim.clip_grad),
            )
            values["critic/grad_norm"] = values.pop("grad_norm")
            append_to_dict(metrics, values)

        if self.update_step >= int(self.cfg.algorithm.ogpo.q_warmup_steps):
            for _ in range(int(self.cfg.algorithm.ogpo.pi_utd)):
                values = self._update(
                    self.optimizer,
                    self.lr_scheduler,
                    self._sample_micro_batches(),
                    self.forward_actor,
                    float(self.cfg.actor.optim.clip_grad),
                )
                values["actor/grad_norm"] = values.pop("grad_norm")
                append_to_dict(metrics, values)

        self._update_targets(actor_tau=0.0)
        self._update_targets(critic_tau=0.0)
        return metrics

    def process_train_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        append_to_dict(
            metrics,
            {
                f"replay_buffer/{key}": value
                for key, value in self.replay_buffer.get_stats().items()
            },
        )
        averaged = {
            key: float(
                np.mean(
                    [
                        value.detach().float().cpu().item()
                        if isinstance(value, torch.Tensor)
                        else value
                        for value in (values if isinstance(values, list) else [values])
                    ]
                )
            )
            for key, values in metrics.items()
        }
        return all_reduce_dict(averaged, op=torch.distributed.ReduceOp.AVG)

    @Worker.timer("run_training")
    def run_training(self) -> dict[str, Any]:
        min_size = int(self.cfg.algorithm.replay_buffer.min_buffer_size)
        if not self.replay_buffer.is_ready(min_size):
            self.log_on_first_rank(
                f"OGPO replay size {len(self.replay_buffer)} < {min_size}; skipping update"
            )
            return {}
        self.model.train()
        metrics = {}
        for _ in range(int(self.cfg.algorithm.update_epoch)):
            for key, values in self.update_one_epoch().items():
                metrics.setdefault(key, []).extend(
                    values if isinstance(values, list) else [values]
                )
            self.update_step += 1
        result = self.process_train_metrics(metrics)
        self.torch_platform.synchronize()
        torch.distributed.barrier()
        self.torch_platform.empty_cache()
        return result

    def compute_advantages_and_returns(self) -> dict:
        return {}

    def save_checkpoint(self, save_base_path: str, step: int) -> None:
        self._strategy.save_checkpoint(
            model=self.model,
            optimizers=[self.optimizer, self.qf_optimizer],
            lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
            save_path=save_base_path,
            checkpoint_format="dcp",
        )
        component_path = os.path.join(save_base_path, "ogpo_components")
        os.makedirs(component_path, exist_ok=True)
        torch.save(
            self._strategy.get_model_state_dict(
                self.target_model, cpu_offload=False, full_state_dict=True
            ),
            os.path.join(component_path, f"target_rank_{self._rank}.pt"),
        )
        torch.save(
            self.update_step,
            os.path.join(component_path, f"step_rank_{self._rank}.pt"),
        )

    def load_checkpoint(self, load_base_path: str) -> None:
        self._strategy.load_checkpoint(
            model=self.model,
            optimizers=[self.optimizer, self.qf_optimizer],
            lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
            load_path=load_base_path,
            checkpoint_format="dcp",
        )
        component_path = os.path.join(load_base_path, "ogpo_components")
        self._strategy.load_model_with_state_dict(
            self.target_model,
            torch.load(
                os.path.join(component_path, f"target_rank_{self._rank}.pt"),
                map_location="cpu",
            ),
            cpu_offload=False,
            full_state_dict=True,
        )
        self.update_step = int(
            torch.load(
                os.path.join(component_path, f"step_rank_{self._rank}.pt"),
                map_location="cpu",
            )
        )
