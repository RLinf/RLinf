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
import torch.nn.functional as F
from omegaconf import ListConfig

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

    # =======================================================================
    # Initialization & Rollout Model Management
    # =======================================================================

    def init_worker(self) -> None:
        """Initialize actor and rollout model state for off-policy support."""
        super().init_worker()
        self.init_rollout_model()

    def init_rollout_model(self) -> None:
        """Initialize rollout model state for off-policy support."""
        self.rollout_model_state_dict = {}
        tau = self.cfg.algorithm.get("nft_tau", 1.0)
        if isinstance(tau, ListConfig):
            tau = list(tau)
            need_rollout_state = min(float(tau[0]), float(tau[1])) < 1.0
        else:
            need_rollout_state = float(tau) < 1.0
        # init rollout model state dict
        if need_rollout_state:
            src = self.get_model_state_dict(cpu_offload=False, full_state_dict=True)
            for key, value in src.items():
                if torch.is_tensor(value):
                    self.rollout_model_state_dict[key] = value.clone()
                else:
                    self.rollout_model_state_dict[key] = value

    def _get_current_nft_tau(self) -> float:
        """Return scalar tau or linearly annealed tau from [start, end, s0, s1]."""
        tau = self.cfg.algorithm.get("nft_tau", 1.0)
        if isinstance(tau, ListConfig):
            tau = list(tau)
        if isinstance(tau, (list, tuple)):
            start_tau, end_tau = float(tau[0]), float(tau[1])
            start_step, end_step = int(tau[2]), int(tau[3])
            step = int(getattr(self, "version", 0))
            if step <= start_step:
                return start_tau
            if step >= end_step:
                return end_tau
            ratio = (
                0.0
                if start_step == end_step
                else ((step - start_step) / float(end_step - start_step))
            )
            return start_tau + ratio * (end_tau - start_tau)
        return float(tau)

    def get_rollout_state_dict(self) -> dict:
        """Return EMA cache when tau<1, otherwise defer to the on-policy parent behavior."""
        if self._get_current_nft_tau() < 1.0:
            return self.rollout_model_state_dict
        return super().get_rollout_state_dict()

    def soft_update_rollout_model(self) -> None:
        """Soft update rollout model: state = (1-tau)*state + tau*current. No-op when tau=1."""
        # TODO: potential bug on model state dict transfer, need to check
        if not self.rollout_model_state_dict:
            return
        tau = self._get_current_nft_tau()
        # soft update rollout model state dict
        current = self.get_model_state_dict(cpu_offload=False, full_state_dict=True)
        for key, rollout_tensor in self.rollout_model_state_dict.items():
            src = current[key]
            if torch.is_tensor(rollout_tensor) and rollout_tensor.is_floating_point():
                if tau >= 1.0:
                    rollout_tensor.copy_(src)
                else:
                    rollout_tensor.lerp_(src.to(rollout_tensor.dtype), tau)
            elif torch.is_tensor(rollout_tensor) and torch.is_tensor(src):
                rollout_tensor.copy_(src)
            else:
                self.rollout_model_state_dict[key] = src

    # =======================================================================
    # Training Loop & Data Preprocessing
    # =======================================================================

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
            self._precompute_nft_training_inputs()

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

                    loss, metrics_data = self.nft_forward_and_loss(batch)

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
        self.soft_update_rollout_model()
        self.optimizer.zero_grad()
        clear_memory()
        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )
        return mean_metric_dict

    def _precompute_nft_training_inputs(self) -> None:
        """Prepare NFT training tensors before the update loop."""
        forward_inputs = self.rollout_batch["forward_inputs"]
        xcur_source = self.cfg.algorithm.get("nft_xcur_source", "rollout")
        num_steps = self.model.config.num_steps
        recompute_v = bool(self.cfg.algorithm.get("recompute_v", False))

        if xcur_source == "resample":
            # Resample step indices and interpolate xcur from x0
            x0 = forward_inputs["nft_x0"]
            step_indices = torch.randint(0, num_steps, (x0.shape[0],), device=x0.device)
            _, t = self._build_schedule_and_timesteps(step_indices, x0.device, x0.dtype)
            xcur = (1 - t[:, None, None]) * x0 + t[:, None, None] * torch.randn_like(x0)
            forward_inputs["nft_xcur"] = xcur
            forward_inputs["nft_step_index"] = step_indices
            recompute_v = True  # must recompute v_old for resampled xcur

        if recompute_v:
            # recompute v_old: always for resample, opt-in for rollout
            xcur = forward_inputs["nft_xcur"]
            step_indices = forward_inputs["nft_step_index"]
            _, t = self._build_schedule_and_timesteps(
                step_indices, xcur.device, xcur.dtype
            )
            forward_inputs["nft_v"] = self._recompute_v_old(forward_inputs, xcur, t)

    def _recompute_v_old(
        self,
        forward_inputs: dict,
        xcur: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Recompute the old velocity with the rollout model."""
        micro_bs = self.cfg.actor.micro_batch_size
        v_old_buffer = []
        training_was_on_device = False
        cleanup_rollout_model = False

        tau = self._get_current_nft_tau()
        if tau >= 1.0:
            # On-policy: use training model directly
            ref_model = self.model
        else:
            # Off-policy: build a temporary model with lagged rollout weights
            training_was_on_device = not self.is_weight_offloaded
            if training_was_on_device:
                self.offload_param_and_grad()
                clear_memory()

            ref_model = self.model_provider_func()
            ref_model.load_state_dict(self.get_rollout_state_dict(), strict=False)
            ref_model.eval()
            ref_model.requires_grad_(False)
            ref_model.to(self.device)
            cleanup_rollout_model = True

        with torch.no_grad():
            for start in range(0, xcur.shape[0], micro_bs):
                end = min(start + micro_bs, xcur.shape[0])
                fi_slice = put_tensor_device(
                    self._slice_forward_inputs(forward_inputs, start, end),
                    self.device,
                )
                out = ref_model(
                    forward_type=ForwardType.NFT,
                    forward_inputs=fi_slice,
                    nft_inputs={
                        "x_t": xcur[start:end].to(device=self.device),
                        "timesteps": t[start:end].to(device=self.device),
                    },
                    compute_values=False,
                )
                v_old_buffer.append(out["v_theta"].detach().cpu())

        if cleanup_rollout_model:
            del ref_model
            clear_memory()
            if training_was_on_device:
                self.load_param_and_grad(self.device)

        return torch.cat(v_old_buffer, dim=0).to(xcur.device)

    # =======================================================================
    # NFT Forward & Loss
    # =======================================================================

    def nft_forward_and_loss(self, batch):
        """NFT-specific forward and loss computation."""
        # prepare inputs
        forward_inputs = batch["forward_inputs"]
        target_space = self.cfg.algorithm.get("nft_target_space", "xnext")
        x_t_input = forward_inputs["nft_xcur"]
        step_indices = forward_inputs["nft_step_index"]
        sum_type = self.cfg.algorithm.get("nft_sum_type", "action_level")
        schedule, t = self._build_schedule_and_timesteps(
            step_indices, x_t_input.device, x_t_input.dtype
        )
        # forward pass
        with self.amp_context:
            output_dict = self.model(
                forward_type=ForwardType.NFT,
                forward_inputs=forward_inputs,
                nft_inputs={"x_t": x_t_input, "timesteps": t},
                compute_values=False,
            )
        # post-process outputs
        chunk = output_dict["v_theta"].shape[1]
        v_theta = output_dict["v_theta"][:, :chunk, :]
        x_t = forward_inputs["nft_xcur"][:, :chunk, :]
        v_old = forward_inputs["nft_v"][:, :chunk, :].detach()
        batch_size, chunk_len = x_t.shape[:2]
        if sum_type == "action_level":
            sum_dims = tuple(range(2, x_t.ndim))
            loss_mask = batch["loss_mask"].expand(batch_size, chunk_len)
            advantages = batch["advantages"].expand(batch_size, chunk_len)
        elif sum_type == "chunk_level":
            sum_dims = tuple(range(1, x_t.ndim))
            loss_mask = batch["loss_mask"].reshape(batch_size, -1)[:, 0]
            advantages = batch["advantages"].reshape(batch_size, -1)[:, 0]
        else:
            raise ValueError(f"Unsupported nft_sum_type: {sum_type}")
        advantages = self._postprocess_advantages(advantages)
        # clip delta v and get pos/neg candidates
        delta_v, clip_coef, v_pos, v_neg = self._compute_clipped_delta_v(
            v_theta, v_old, sum_dims
        )
        # build schedule params
        t_bc, dt_bc, sigma_i, std_t_det = self._build_schedule_params(
            schedule, step_indices, forward_inputs["nft_noise_level"], x_t
        )
        # compute target and predictions
        target, pred_pos = self._compute_nft_target_and_pred(
            forward_inputs, target_space, x_t, v_pos, t_bc, dt_bc, sigma_i
        )
        _, pred_neg = self._compute_nft_target_and_pred(
            forward_inputs, target_space, x_t, v_neg, t_bc, dt_bc, sigma_i
        )
        # compute weighted energies
        noise_level = forward_inputs["nft_noise_level"]
        weight_mode = self.cfg.algorithm.get("nft_weight_mode", "auto")
        w_pos = self._compute_nft_weight(
            weight_mode,
            t_bc,
            std_t_det,
            noise_level,
            target,
            sum_dims,
            pred=pred_pos,
            sample_type="pos",
        )
        w_neg = self._compute_nft_weight(
            weight_mode,
            t_bc,
            std_t_det,
            noise_level,
            target,
            sum_dims,
            pred=pred_neg,
            sample_type="neg",
        )
        e_pos = ((pred_pos - target) ** 2 * w_pos).sum(dim=sum_dims)
        e_neg = ((pred_neg - target) ** 2 * w_neg).sum(dim=sum_dims)
        # loss
        delta_e = e_pos - e_neg
        loss = self._compute_nft_loss(e_pos, e_neg, delta_e, advantages, loss_mask)
        # metrics
        with torch.no_grad():
            metrics_data = {
                "actor/nft_loss": loss.item(),
                "actor/nft_tau": self._get_current_nft_tau(),
                "actor/delta_v_norm": delta_v.norm(dim=sum_dims).mean().item(),
                "actor/clip_frac": (clip_coef < 1).float().mean().item(),
                "actor/E_pos_mean": e_pos.mean().item(),
                "actor/E_neg_mean": e_neg.mean().item(),
                "actor/E_pos_mean_pos_only": masked_mean(
                    e_pos, (advantages > 0.5) & loss_mask.bool()
                ).item(),
                "actor/E_neg_mean_neg_only": masked_mean(
                    e_neg, (advantages < 0.5) & loss_mask.bool()
                ).item(),
                "actor/delta_E_mean": delta_e.mean().item(),
            }
        return loss, metrics_data

    def _postprocess_advantages(self, advantages: torch.Tensor) -> torch.Tensor:
        """Map advantages into [0, 1] to match NFT loss semantics (r=0 -> neg, r=1 -> pos).

        - adv_type == "raw": success rewards already in [0, 1], no-op.
        - Otherwise (e.g. grpo): clip to [-adv_clip_max, +adv_clip_max] then
          linearly rescale into [0, 1] via (adv + max) / (2 * max). After this
          the subsequent clamp in _compute_nft_loss becomes a no-op.
        """
        adv_type = self.cfg.algorithm.get("adv_type", "raw")
        if adv_type == "raw":
            return advantages
        adv_clip_max = float(self.cfg.algorithm.get("adv_clip_max", 1.0))
        advantages = advantages.clamp(-adv_clip_max, adv_clip_max)
        advantages = (advantages + adv_clip_max) / (2.0 * adv_clip_max)
        return advantages

    def _compute_clipped_delta_v(
        self,
        v_theta: torch.Tensor,
        v_old: torch.Tensor,
        sum_dims: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute clipped delta_v and pos/neg candidate velocities.

        Returns: (delta_v, clip_coef, v_pos, v_neg)
        """
        delta_v = v_theta - v_old
        delta_norm = delta_v.norm(dim=sum_dims, keepdim=True) + 1e-8
        max_drift = float(self.cfg.algorithm.get("max_drift", 0.5))
        clip_coef = (max_drift / delta_norm).clamp(max=1.0)
        beta = float(self.cfg.algorithm.get("nft_beta", 1.0))
        delta_v_clipped = delta_v * clip_coef
        v_pos = v_old + beta * delta_v_clipped
        v_neg = v_old - beta * delta_v_clipped
        return delta_v, clip_coef, v_pos, v_neg

    def _compute_nft_loss(
        self,
        e_pos: torch.Tensor,
        e_neg: torch.Tensor,
        delta_e: torch.Tensor,
        advantages: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute final NFT loss from pos/neg energies.

        Assumes advantages already lies in [0, 1] (enforced by _postprocess_advantages).
        """
        loss_form = self.cfg.algorithm.get("nft_loss_form", "dpo")
        if loss_form == "dpo":
            dpo_beta = float(self.cfg.algorithm.get("dpo_beta", 1.0))
            y = advantages * 2.0 - 1.0
            logit = (dpo_beta / 2.0) * y * delta_e
            return masked_mean(F.softplus(logit), loss_mask)
        elif loss_form == "mse":
            r = advantages
            return masked_mean(r * e_pos + (1.0 - r) * e_neg, loss_mask)
        else:
            raise ValueError(f"Unsupported nft_loss_form: {loss_form}")

    # =======================================================================
    # NFT Math Utilities (schedule, target/pred, weight, slicing)
    # =======================================================================

    def _build_schedule_and_timesteps(
        self,
        step_indices: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build flow-matching schedule and lookup timesteps for given step indices.

        Returns: (schedule, t) where schedule is [num_steps+1] linspace 1->0,
                 t is the timestep values at step_indices.
        """
        num_steps = self.model.config.num_steps
        schedule = torch.linspace(1, 0, num_steps + 1, device=device, dtype=dtype)
        t = schedule[step_indices.long()]
        return schedule, t

    def _build_schedule_params(
        self,
        schedule: torch.Tensor,  # [num_steps+1] linspace 1->0
        step_indices: torch.Tensor,  # [B]
        noise_level: torch.Tensor | float,
        x_t: torch.Tensor,  # reference tensor for ndim/device/dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute timestep & noise params, broadcast to [B, 1, ..., 1] for x_t.ndim.

        Returns: (t_bc, dt_bc, sigma_i, std_t_det)
        """
        # TODO: move into the model file or utils file for better reuse
        ndim = x_t.ndim
        idx = step_indices.long()

        def pad(x):
            return x.view(-1, *([1] * (ndim - 1)))

        # timestep: t_cur and dt = t_cur - t_next
        t_bc = pad(schedule[idx])
        dt_bc = pad(schedule[idx] - schedule[idx + 1])
        # SDE noise scale: sigma_i = sqrt(t / (1-t)) * noise_level
        safe_schedule = schedule.clone()
        safe_schedule[0] = safe_schedule[1]  # avoid div-by-zero at t=1
        sigma_i = pad(torch.sqrt(schedule[:-1] / (1 - safe_schedule[:-1]))[idx])
        nl = torch.as_tensor(noise_level, device=x_t.device, dtype=x_t.dtype)
        sigma_i = sigma_i * (pad(nl) if nl.ndim > 0 else nl)
        # transition std
        std_t_det = (torch.sqrt(dt_bc.clamp_min(0)) * sigma_i).detach()
        return t_bc, dt_bc, sigma_i, std_t_det

    def _compute_nft_target_and_pred(
        self,
        forward_inputs: dict,
        target_space: str,
        x_t: torch.Tensor,
        vel: torch.Tensor,
        t_bc: torch.Tensor,
        dt_bc: torch.Tensor,
        sigma_i: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build target and predicted state for the given NFT target space."""
        # TODO: move into the model file or utils file for better reuse
        x0_pred = x_t - vel * t_bc
        x1_pred = x_t + vel * (1 - t_bc)
        if target_space == "x0":
            target = forward_inputs["nft_x0"][:, : x_t.shape[1], : x_t.shape[2]]
            pred = x0_pred - (sigma_i**2 / 2) * x1_pred
        elif target_space == "xnext":
            target = forward_inputs["nft_xnext"][:, : x_t.shape[1], : x_t.shape[2]]
            w0 = 1.0 - (t_bc - dt_bc)
            w1 = t_bc - dt_bc - sigma_i**2 * dt_bc / (2 * t_bc)
            pred = x0_pred * w0 + x1_pred * w1
        else:
            raise ValueError(f"Unsupported nft_target_space: {target_space}")
        return target, pred

    def _compute_nft_weight(
        self,
        weight_mode: str,
        t_bc: torch.Tensor,
        std_t_det: torch.Tensor,
        noise_level: torch.Tensor,
        target: torch.Tensor,
        sum_dims: tuple[int, ...],
        *,
        pred: torch.Tensor,
        sample_type: str,
    ) -> torch.Tensor | float:
        """Compute per-element multiplicative weight for NFT energy.

        Supported weight_mode values (set via ``nft_weight_mode`` in config):
            "constant"  -- fixed scalar ``nft_weight_scale`` (default 1.0).
            "t"        -- 1 / (t^2 or dt^2), scaled by ``nft_weight_scale``.
            "inv_t2"   -- 1 / (t^2 + eps), compensates x0 target gradient ∝ t^2.
            "inv_t3"   -- 1 / (t^3 + eps), overcompensates to bias toward low-t.
            "sigma"    -- 1 / (std_t_det^2 + eps).
            "adaptive" -- 1 / abs-error-mean (stop-grad), a la DiffusionNFT.
            "auto"     -- "adaptive" when noise_level==0 (ODE), otherwise "sigma" (SDE).
        """
        # auto mode selection
        if weight_mode == "auto":
            if torch.all(noise_level == 0):
                weight_mode = "adaptive"
            else:
                weight_mode = "sigma"
        # weight computation
        weight = self.cfg.algorithm.get("nft_weight_scale", 1.0)
        if isinstance(weight, ListConfig):
            weight = list(weight)
        if isinstance(weight, (list, tuple)):
            if len(weight) != 2:
                raise ValueError(
                    "nft_weight_scale list must be [pos_scale, neg_scale]."
                )
            weight = float(weight[0] if sample_type == "pos" else weight[1])
        else:
            weight = float(weight)
        if weight_mode == "constant":
            pass
        elif weight_mode == "t":
            weight *= t_bc**2
        elif weight_mode == "1-t":
            weight *= (1 - t_bc) ** 2
        elif weight_mode == "inv_t2":
            weight /= t_bc**2 + 1e-4
        elif weight_mode == "inv_t3":
            weight /= t_bc**3 + 1e-4
        elif weight_mode == "sigma":
            # loss in pi-step-nft paper
            weight /= std_t_det**2 + 1e-4
        elif weight_mode == "adaptive":
            # loss in diffusion-nft paper # todo: bug here, need to check
            with torch.no_grad():
                w = (
                    torch.abs(pred.double() - target.double())
                    .mean(dim=sum_dims, keepdim=True)
                    .clamp(min=1e-4)
                    .to(pred.dtype)
                )
            weight /= w
        return weight

    def _slice_forward_inputs(self, forward_inputs: dict, start: int, end: int) -> dict:
        """Slice nested forward inputs along the batch dimension."""
        ret = {}
        for key, value in forward_inputs.items():
            if isinstance(value, torch.Tensor):
                ret[key] = value[start:end]
            elif isinstance(value, dict):
                ret[key] = self._slice_forward_inputs(value, start, end)
            else:
                ret[key] = value
        return ret
