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
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler.worker.worker import Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.nested_dict_process import put_tensor_device, split_dict_to_chunk
from rlinf.utils.utils import clear_memory, masked_mean
from rlinf.workers.actor.fsdp_actor_worker import (
    EmbodiedFSDPActor,
    process_nested_dict_for_train,
)


class EmbodiedNFTFSDPPolicy(EmbodiedFSDPActor):
    """Embodied FSDP policy worker for NFT with off-policy support."""

    @Worker.timer("run_training")
    def run_training(self) -> None:
        """Run NFT training with off-policy decay support."""
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

        self.model.train()
        rollout_size = (
            self.rollout_batch["prev_logprobs"].shape[0]
            * self.rollout_batch["prev_logprobs"].shape[1]
        )
        g = torch.Generator()
        g.manual_seed(self.cfg.actor.seed + self._rank)
        shuffle_id = torch.randperm(rollout_size, generator=g)

        with torch.no_grad():
            self.rollout_batch = process_nested_dict_for_train(
                self.rollout_batch, shuffle_id
            )

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        ), "global_batch_size is not divisible by micro_batch_size * world_size"

        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        rollout_size = self.rollout_batch["prev_logprobs"].size(0)
        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        assert rollout_size % batch_size_per_rank == 0, (
            f"{rollout_size} is not divisible by {batch_size_per_rank}"
        )
        metrics = {}
        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            rollout_dataloader_iter = split_dict_to_chunk(
                self.rollout_batch,
                rollout_size // batch_size_per_rank,
            )
            for train_global_batch in rollout_dataloader_iter:
                train_global_batch_size = train_global_batch["prev_logprobs"].shape[0]
                assert (
                    train_global_batch_size
                    == self.cfg.actor.global_batch_size
                    // torch.distributed.get_world_size()
                )
                assert train_global_batch_size % self.cfg.actor.micro_batch_size == 0, (
                    f"{train_global_batch_size=}, {self.cfg.actor.micro_batch_size}"
                )

                train_micro_batch = split_dict_to_chunk(
                    train_global_batch,
                    train_global_batch_size // self.cfg.actor.micro_batch_size,
                )

                self.optimizer.zero_grad()
                for idx, batch in enumerate(train_micro_batch):
                    batch = put_tensor_device(
                        batch,
                        f"{Worker.torch_device_type}:{int(os.environ['LOCAL_RANK'])}",
                    )
                    backward_ctx = self.before_micro_batch(
                        self.model,
                        is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
                    )

                    loss, metrics_data = self._nft_forward_and_loss(batch)

                    if self.enable_sft_co_train:
                        self._train_sft_epoch(metrics_data, loss)

                    loss /= self.gradient_accumulation
                    with backward_ctx:
                        self.grad_scaler.scale(loss).backward()

                    metrics_data["actor/total_loss"] = loss.detach().item()
                    append_to_dict(metrics, metrics_data)

                self.torch_platform.empty_cache()

                grad_norm, lr_list = self.optimizer_step()
                data = {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                }
                if len(lr_list) > 1:
                    data["critic/lr"] = lr_list[1]
                append_to_dict(metrics, data)
        # put LR scheduler step here
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()
        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )

        return mean_metric_dict

    def _nft_forward_and_loss(self, batch):
        """NFT-specific forward and loss computation."""
        # data input
        forward_inputs = batch["forward_inputs"]
        x_t_input = forward_inputs["nft_x"]
        step_indices = forward_inputs["nft_step_index"]
        num_steps = self.cfg.actor.model.openpi.get("num_steps", 10)
        schedule = torch.linspace(
            1,
            0,
            num_steps + 1,
            device=x_t_input.device,
            dtype=x_t_input.dtype,
        )
        t = schedule[step_indices.long()]
        # compute v_theta
        with self.amp_context:
            output_dict = self.model(
                forward_type=ForwardType.NFT,
                forward_inputs=forward_inputs,
                nft_explicit_inputs={"x_t": x_t_input, "timesteps": t},
                compute_values=False,
            )
        # compute loss
        chunk = output_dict["v_theta"].shape[1]
        action_env_dim = self.cfg.actor.model.openpi.get("action_env_dim", 7)
        loss, metrics_data = self._compute_embodied_nft_loss(
            v_theta=output_dict["v_theta"][:, :chunk, :action_env_dim],
            v_old=forward_inputs["nft_v"][:, :chunk, :action_env_dim],
            x_t=forward_inputs["nft_x"][:, :chunk, :action_env_dim],
            x_next=forward_inputs["nft_xnext"][:, :chunk, :action_env_dim],
            schedule=schedule,
            step_indices=step_indices,
            noise_level=forward_inputs["nft_noise_level"],
            advantages=batch["advantages"],
            beta=self.cfg.algorithm.get("nft_beta", 1.0),
            adv_clip_max=self.cfg.algorithm.get("adv_clip_max", 1.0),
            dpo_beta=self.cfg.algorithm.get("dpo_beta", 1.0),
            max_drift=self.cfg.algorithm.get("max_drift", 0.5),
            loss_mask=batch["loss_mask"],
        )
        return loss, metrics_data

    def _compute_embodied_nft_loss(
        self,
        v_theta: torch.Tensor,  # [batch_size, chunk_len, action_env_dim]
        v_old: torch.Tensor,  # [batch_size, chunk_len, action_env_dim]
        x_t: torch.Tensor,  # [batch_size, chunk_len, action_env_dim]
        x_next: torch.Tensor,  # [batch_size, chunk_len, action_env_dim]
        schedule: torch.Tensor,  # [num_steps+1]
        step_indices: torch.Tensor,
        noise_level: torch.Tensor | float,
        advantages: torch.Tensor,
        beta: float = 1.0,
        adv_clip_max: float = 1.0,
        dpo_beta: float = 1.0,
        max_drift: float = 0.5,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute DPO-style energy-based NFT loss."""
        # shape alignment
        batch_size, chunk_len = x_t.shape[:2]
        sum_dims = tuple(range(2, x_t.ndim))
        loss_mask = loss_mask.expand(batch_size, chunk_len)
        advantages = advantages.expand(batch_size, chunk_len)
        # preference y ∈ [-1, 1]
        y = (
            torch.clamp(advantages * 2.0 - 1.0, -adv_clip_max, adv_clip_max)
            / adv_clip_max
        )
        # clip delta v
        v_old = v_old.detach()
        delta_v = v_theta - v_old
        delta_norm = delta_v.norm(dim=sum_dims, keepdim=True) + 1e-8
        max_drift = float(max_drift)
        clip_coef = (max_drift / delta_norm).clamp(max=1.0)
        delta_v_clipped = delta_v * clip_coef
        # pos and neg candidate velocities
        v_pos = v_old + beta * delta_v_clipped
        v_neg = v_old - beta * delta_v_clipped
        # schedule params
        t_bc, dt_bc, sigma_i, std_t_det = self._build_schedule_params(
            schedule, step_indices, noise_level, x_t
        )
        # flow matching transition mean

        def _flow_mean(x_cur: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
            x0_pred = x_cur - vel * t_bc
            x1_pred = x_cur + vel * (1 - t_bc)
            w0 = 1.0 - (t_bc - dt_bc)
            w1 = t_bc - dt_bc - sigma_i**2 * dt_bc / (2 * t_bc)
            return x0_pred * w0 + x1_pred * w1

        # x-next prediction
        x_next_pos = _flow_mean(x_t, v_pos)
        x_next_neg = _flow_mean(x_t, v_neg)
        var = std_t_det**2 + 1e-4
        e_pos = ((x_next_pos - x_next) ** 2 / var).sum(dim=sum_dims)
        e_neg = ((x_next_neg - x_next) ** 2 / var).sum(dim=sum_dims)
        delta_e = e_pos - e_neg
        # dpo loss

        logit = (float(dpo_beta) / 2.0) * y * delta_e
        nft_loss = masked_mean(F.softplus(logit), loss_mask)
        # metrics
        with torch.no_grad():
            metrics_data = {
                "actor/nft_loss": nft_loss.item(),
                "actor/delta_v_norm": delta_v.norm(dim=sum_dims).mean().item(),
                "actor/clip_frac": (clip_coef < 1).float().mean().item(),
                "actor/E_pos_mean": e_pos.mean().item(),
                "actor/E_neg_mean": e_neg.mean().item(),
                "actor/delta_E_mean": delta_e.mean().item(),
                "actor/logit_mean": logit.mean().item(),
                "actor/pref_acc": (logit < 0).float().mean().item(),
            }
        return nft_loss, metrics_data

    def _build_schedule_params(
        self,
        schedule: torch.Tensor,  # [num_steps+1] linspace 1→0
        step_indices: torch.Tensor,  # [B]
        noise_level: torch.Tensor | float,
        x_t: torch.Tensor,  # reference tensor for ndim/device/dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute timestep & noise params, broadcast to [B, 1, ..., 1] for x_t.ndim.

        Returns: (t_bc, dt_bc, sigma_i, std_t_det)
        """
        ndim = x_t.ndim

        def pad(x):
            return x.view(-1, *([1] * (ndim - 1)))

        idx = step_indices.long()
        # timestep: t_cur and dt = t_cur - t_next
        t_bc = pad(schedule[idx])
        dt_bc = pad(schedule[idx] - schedule[idx + 1])
        # SDE noise scale: σ_i = sqrt(t / (1-t)) * noise_level
        safe_schedule = schedule.clone()
        safe_schedule[0] = safe_schedule[1]  # avoid div-by-zero at t=1
        sigma_i = pad(torch.sqrt(schedule[:-1] / (1 - safe_schedule[:-1]))[idx])
        nl = torch.as_tensor(noise_level, device=x_t.device, dtype=x_t.dtype)
        sigma_i = sigma_i * (pad(nl) if nl.ndim > 0 else nl)
        # transition std
        std_t_det = (torch.sqrt(dt_bc.clamp_min(0)) * sigma_i).detach()
        return t_bc, dt_bc, sigma_i, std_t_det
