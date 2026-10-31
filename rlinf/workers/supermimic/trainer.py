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

"""In-process SuperMimic loop. The Ray Worker is a thin wrapper around this."""

from __future__ import annotations

import math
import os
import shutil
import sys
import time
import warnings
from pathlib import Path
from types import ModuleType

import torch
from omegaconf import DictConfig

from rlinf.algorithms.supermimic.objectives import (
    actor_objective,
    gae_targets,
    state_prediction_loss,
    termination_bce,
)
from rlinf.algorithms.supermimic.replay import SuperMimicReplay
from rlinf.algorithms.supermimic.reset_sampler import BinResetSampler
from rlinf.algorithms.supermimic.scan import (
    make_dynamics_window,
    make_imagination,
    run_compiled,
)
from rlinf.envs.sim.g1_mimic.constants import (
    DT,
    EPISODE_CONT,
    EPISODE_FAILURE,
    EPISODE_TIMEOUT,
    HISTORY_LEN,
)
from rlinf.envs.sim.g1_mimic.env import compose_end_kind, make_env
from rlinf.envs.sim.g1_mimic.features import integrate_residual
from rlinf.envs.sim.g1_mimic.g1_fk import asset_joint_limits
from rlinf.envs.sim.g1_mimic.motion import GOAL_WINDOW
from rlinf.envs.sim.g1_mimic.obs import add_sensor_noise, analytic_obs
from rlinf.envs.sim.g1_mimic.reward import (
    analytic_failure_prob,
    analytic_reward,
    compose_goal,
    differentiable_fk,
    failure_signals,
    motion_critic_input,
    termination_head_input,
)
from rlinf.envs.sim.g1_mimic.velocity import (
    ALIGN_WEIGHT,
    RESAMPLE_STEPS,
    heading_velocity,
    normalize_command,
    sample_velocity_command,
    velocity_fall,
    velocity_fall_soft,
    velocity_reward,
)
from rlinf.models.embodiment.supermimic import SuperMimicModels, running_state_loss
from rlinf.workers.supermimic import checkpoint as ckpt
from rlinf.workers.supermimic.distributed import DataParallel
from rlinf.workers.supermimic.eval_rollout import (
    RollingHistory,
    summarize_tracking_eval,
)
from rlinf.workers.supermimic.evaluator import build_motion_library, motion_eval_rows


def _host_rss_bytes() -> float:
    try:
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) * 1024.0
    except (OSError, ValueError, IndexError):
        return 0.0
    return 0.0


def gpu_memory_metrics(
    device: torch.device, platform: ModuleType = torch.cuda
) -> dict[str, float]:
    """GPU memory from three counters that overlap; do not add them up.

    ``torch_*`` is PyTorch's caching allocator only. ``warp_mempool_*`` is
    Warp's memory pool (simulator buffers), reported when Warp is loaded.
    ``device_used`` is the whole device as the driver sees it: every process,
    CUDA context and allocator on that GPU.
    """
    if device.type == "cpu":
        return {}
    free, total = platform.mem_get_info(device)
    out = {
        "memory/torch_peak_allocated": float(platform.max_memory_allocated(device)),
        "memory/torch_peak_reserved": float(platform.max_memory_reserved(device)),
        "memory/device_used": float(total - free),
        "memory/device_total": float(total),
    }
    warp = sys.modules.get("warp")
    if warp is not None and device.type == "cuda":
        name = f"cuda:{device.index or 0}"
        if warp.is_mempool_enabled(name):
            out["memory/warp_mempool_current"] = float(
                warp.get_mempool_used_mem_current(name)
            )
            out["memory/warp_mempool_peak"] = float(
                warp.get_mempool_used_mem_high(name)
            )
    return out


def _set_requires_grad(params, flag: bool) -> None:
    for param in params:
        param.requires_grad_(flag)


def _clip_actor_grads(params) -> torch.Tensor:
    norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
    for param in params:
        if param.grad is not None:
            param.grad.clamp_(-0.1, 0.1)
    return norm


def _grad_norm(params) -> torch.Tensor:
    grads = [param.grad for param in params if param.grad is not None]
    if not grads:
        return torch.zeros(())
    return torch.linalg.vector_norm(torch.stack([g.detach().norm() for g in grads]))


def _host_floats(values: list[torch.Tensor | float]) -> list[float]:
    """Read several scalars into Python with one device synchronization."""
    tensors = [value for value in values if torch.is_tensor(value)]
    if not tensors:
        return [float(value) for value in values]
    device = next(
        (value.device for value in tensors if value.is_cuda), tensors[0].device
    )
    ready = []
    for value in values:
        if torch.is_tensor(value):
            ready.append(value.detach().reshape(()).float().to(device))
        else:
            ready.append(torch.tensor(float(value), device=device))
    return [float(item) for item in torch.stack(ready).tolist()]


def _term_grad_norm(term: torch.Tensor, params) -> float:
    """Gradient norm of one loss term. The graph is kept for the real backward."""
    if not term.requires_grad:
        return 0.0
    grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
    norms = [g.norm() for g in grads if g is not None]
    if not norms:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(norms)))


def _finite(x: torch.Tensor | float) -> bool:
    value = float(x) if not torch.is_tensor(x) else float(x.detach())
    return math.isfinite(value)


class SuperMimicTrainer:
    """One rank of SuperMimic: collection, world model, actor and critic updates.

    Args:
        cfg: A validated SuperMimic config.
        device: Device for the env and networks; defaults to the config's.
        dp: Gradient all-reduce over ranks; defaults to a single rank.
        platform: Accelerator module (``Worker.torch_platform``); defaults to ``torch.cuda``.
    """

    def __init__(
        self,
        cfg: DictConfig,
        device: torch.device | None = None,
        dp: DataParallel | None = None,
        platform: ModuleType | None = None,
    ):
        """``platform`` is the worker's ``torch_platform`` (``torch.cuda`` on NVIDIA)."""
        self.cfg = cfg
        algo = cfg.algorithm
        env_cfg = cfg.env
        self.platform = torch.cuda if platform is None else platform
        if device is None:
            requested = str(env_cfg.get("device", "cpu"))
            if requested.startswith("cuda") and self.platform.is_available():
                device = torch.device(requested)
            else:
                device = torch.device("cpu")
        self.device = device
        self.dp = dp if dp is not None else DataParallel(1, 0, device)
        world = self.dp.world_size
        seed = algo.get("seed", None)
        if seed is None and world > 1:
            # torch's default seed is the same in every process, so unseeded
            # ranks would draw identical noise and resets.
            seed = 0
        self.seed = None if seed is None else int(seed)
        if self.seed is not None:
            # Envs and sampling differ per rank. Network weights do not
            # depend on this: rank 0 broadcasts them below.
            torch.manual_seed(self.seed + self.dp.rank)
        self.goal_anchor = bool(algo.goal_anchor)
        self.k = int(algo.k)
        self.H = int(algo.H)
        self.gamma = float(algo.gamma)
        self.lam = float(algo.lam)
        self.goal_dim = int(algo.goal_dim)
        # env.num_envs, wm_window_batch and imagine_batch are global.
        # Each rank takes 1/world of them; validate checks divisibility.
        self.num_envs = int(env_cfg.num_envs) // world
        self.warmup = int(algo.warmup_sim_steps)
        self.wm_only_updates = int(algo.get("wm_only_updates", 0))
        self.updates_per_env_step = float(algo.updates_per_env_step)
        self.wm_batches = int(algo.wm_batches)
        self.wm_window_batch = int(algo.wm_window_batch) // world
        self.imagine_batch = int(algo.imagine_batch) // world
        self.critic_minibatches = int(algo.get("critic_minibatches", 1))
        self.critic_epochs = int(algo.get("critic_epochs", 1))
        self.max_nonfinite_skips = int(algo.get("max_nonfinite_skips", 20))
        self.nonfinite_skips = 0
        self.consecutive_nonfinite = 0
        self.bce_weight = float(algo.get("bce_weight", 1.0))
        # Experimental: "analytic" replaces the learned head with a soft
        # threshold (Sec. 3.2 allows either). The paper baseline is "learned".
        self.learned_termination = (
            str(algo.get("termination_head", "learned")) == "learned"
        )
        # Extra backward passes per term; off for training runs.
        self.grad_diagnostics = bool(algo.get("grad_diagnostics", False))
        self.action_noise_std = float(algo.action_noise_std)
        self.G_max = float(algo.G_max)
        # The simulator's failure labels, the head's BCE inputs, and the
        # imagined d_hat must all come from this one body.
        self.termination_body = str(algo.termination_body)
        self.motion = build_motion_library(cfg, self.device)
        self.env = make_env(
            str(env_cfg.backend),
            num_envs=self.num_envs,
            motion=self.motion,
            device=self.device,
            max_episode_seconds=float(algo.max_episode_seconds),
            termination_body=self.termination_body,
            eval_mode=False,
        )
        self.q0 = self.motion.q0.to(self.device)
        joint_low, joint_high = asset_joint_limits()
        self.joint_low = joint_low.to(device=self.device)
        self.joint_high = joint_high.to(device=self.device)
        self.obs_dim = int(algo.obs_dim)
        self.buffer = SuperMimicReplay(
            self.num_envs,
            int(env_cfg.replay_length),
            self.device,
            obs_dim=self.obs_dim,
        )
        self.goal_encoding = str(algo.get("goal_encoding", "paper"))
        self.include_ref_ori = int(algo.obs_dim) == 99
        self.velocity = bool(algo.get("velocity_tracking", False))
        self.critic_features = str(algo.get("critic_features", "root4"))
        # Omitted keys follow zita-ch. supermimic_base.yaml writes the paper
        # values, so a paper config does not inherit these defaults.
        self.wm_state_loss = str(algo.get("wm_state_loss", "running"))
        self.reward_anchor = str(algo.get("reward_anchor", "pelvis"))
        self.input_norm = str(algo.get("input_norm", "running"))
        self.residual_frame = str(algo.get("residual_frame", "body"))
        self.scale_xi = self.input_norm == "running"
        bias_init = algo.get("termination_bias_init", None)
        self.models = SuperMimicModels(
            self.goal_dim,
            self.obs_dim,
            velocity=self.velocity,
            termination_bias_init=None if bias_init is None else float(bias_init),
            critic_features=self.critic_features,
            input_norm=self.scale_xi,
            output_norm=self.wm_state_loss == "running",
        ).to(self.device)
        matmul = str(algo.get("matmul", "tf32"))
        self.models.set_bf16_matmul(matmul == "bf16")
        self.dp.broadcast_module(self.models)
        lr = float(algo.lr)
        self.opt_wm = torch.optim.Adam(self.models.wm_params(), lr=lr)
        self.opt_actor = torch.optim.Adam(
            self.models.actor_params(), lr=lr, betas=(0.9, 0.95)
        )
        self.opt_critic = torch.optim.Adam(self.models.critic_params(), lr=lr)
        self.opt_velocity = None
        self.opt_velocity_critic = None
        if self.velocity:
            self.opt_velocity = torch.optim.Adam(
                self.models.velocity_actor_params(), lr=lr, betas=(0.9, 0.95)
            )
            self.opt_velocity_critic = torch.optim.Adam(
                self.models.velocity_critic_params(), lr=lr
            )
        decay = int(algo.lr_decay_updates)

        def lr_lambda(step: int) -> float:
            if step >= decay:
                return 0.03
            return 0.03 + 0.97 * 0.5 * (1.0 + math.cos(math.pi * step / decay))

        self.sched_wm = torch.optim.lr_scheduler.LambdaLR(self.opt_wm, lr_lambda)
        self.sched_actor = torch.optim.lr_scheduler.LambdaLR(self.opt_actor, lr_lambda)
        self.sched_critic = torch.optim.lr_scheduler.LambdaLR(
            self.opt_critic, lr_lambda
        )
        self.sched_velocity = None
        self.sched_velocity_critic = None
        if self.opt_velocity is not None:
            self.sched_velocity = torch.optim.lr_scheduler.LambdaLR(
                self.opt_velocity, lr_lambda
            )
            self.sched_velocity_critic = torch.optim.lr_scheduler.LambdaLR(
                self.opt_velocity_critic, lr_lambda
            )
        self.step_acc = 0.0
        self.global_update = 0
        self.sim_vector_steps = 0
        self.env_transitions = 0
        self.wm_optimizer_steps = 0
        self.actor_optimizer_steps = 0
        self.critic_optimizer_steps = 0
        self.prev_action = torch.zeros(
            self.num_envs, int(algo.action_dim), device=self.device
        )
        half = self.num_envs // 2
        self.velocity_env = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        if self.velocity:
            self.velocity_env[half:] = True
        self.command = torch.zeros(self.num_envs, 3, device=self.device)
        self.command_from_ref = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.command_age = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.obs_hist: RollingHistory | None = None
        self._eval_env = None
        self.replay_warm = False
        self.time_replay_sample = 0.0
        self.time_world_model = 0.0
        self.time_bptt = 0.0
        self.time_critic = 0.0
        self.time_sim = 0.0
        max_steps = max(int(round(float(algo.max_episode_seconds) / DT)), 1)
        self.reset_sampler = BinResetSampler(
            self.motion.lengths,
            self.num_envs,
            self.G_max,
            self.gamma,
            max_steps,
            self.device,
        )
        self.reset_sampler.deferred = self.dp.enabled
        # The env constructor draws a uniform frame. Training starts use the
        # return-bin mixture instead, including this first reset.
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.env.reset(env_ids, *self._reset_points(env_ids))
        self._update_commands(env_ids, force=True)
        self._refresh_history()

    def _goal(self, motion_id, motion_time, state):
        if self.goal_encoding == "goal730":
            return self.motion.goal730(motion_id, motion_time)
        goal_ref = self.motion.goal_ref(motion_id, motion_time)
        ref = self.motion.state_at(motion_id, motion_time)
        return compose_goal(goal_ref, state, ref, self.goal_anchor)

    def _ref_quat(self, motion_id, motion_time):
        if not self.include_ref_ori:
            return None
        return self.motion.state_at(motion_id, motion_time)[..., 3:7]

    def _sync(self) -> None:
        """Wait for the device. Off unless ``SUPERMIMIC_PROFILE=1``."""
        if os.environ.get("SUPERMIMIC_PROFILE") != "1":
            return
        if self.device.type != "cpu":
            self.platform.synchronize(self.device)

    def _history_tensor(self) -> torch.Tensor:
        return self.obs_hist.flat()

    def _noisy_obs(self, state, prev, xi, ref_quat, noise: bool) -> torch.Tensor:
        """One new frame. ``noise`` draws uniform sensor noise once."""
        return add_sensor_noise(
            analytic_obs(state, prev, xi, self.q0, ref_quat=ref_quat), noise
        )

    def _new_frames(self, env_ids: torch.Tensor) -> torch.Tensor:
        """One noised frame per listed env, from the env's current state."""
        ref = self._ref_quat(self.env.motion_id[env_ids], self.env.motion_time[env_ids])
        return self._noisy_obs(
            self.env.state[env_ids],
            self.prev_action[env_ids],
            self.env.xi[env_ids],
            ref,
            True,
        ).detach()

    def _refresh_history(self) -> None:
        # Reset fill repeats one noised frame. That single draw is the noise
        # sample for every padded slot; later steps do not redraw it.
        env_ids = torch.arange(self.num_envs, device=self.device)
        frames = self._new_frames(env_ids)
        if self.obs_hist is None:
            self.obs_hist = RollingHistory(frames)
        else:
            self.obs_hist.fill(env_ids, frames)

    def _append_obs(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel():
            self.obs_hist.push(self._new_frames(env_ids), env_ids)

    def _reset_history(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel():
            self.obs_hist.fill(env_ids, self._new_frames(env_ids))

    def _update_collection_norms(self, obs: torch.Tensor, goal: torch.Tensor) -> None:
        """Update policy running norms on the frame the policy is about to see.

        Every rank joins the merge. A velocity rank with no motion rows still
        contributes a zero-count goal batch.
        """
        self.dp.update_norm(self.models.decoder.obs_norm, obs)
        if self.models.encoder.goal_norm is None:
            return
        if self.velocity:
            goal = goal[~self.velocity_env]
        self.dp.update_norm(self.models.encoder.goal_norm, goal)

    def collect_steps(self, n_steps: int, accumulate: bool = True) -> dict[str, float]:
        """Step every env ``n_steps`` times and store the transitions.

        With ``accumulate`` the steps count toward the update-to-data ratio.
        """
        failures = 0
        episodes = 0
        reward_total = torch.zeros((), device=self.device)
        sim_s = 0.0
        for _ in range(n_steps):
            history = self._history_tensor()
            current_obs = history.reshape(self.num_envs, HISTORY_LEN, self.obs_dim)[
                :, -1
            ]
            goal = self._goal(self.env.motion_id, self.env.motion_time, self.env.state)
            self._update_collection_norms(current_obs, goal)
            with torch.no_grad():
                if self.velocity:
                    self._update_commands(
                        torch.arange(self.num_envs, device=self.device), force=False
                    )
                    action = torch.zeros_like(self.prev_action)
                    motion_env = ~self.velocity_env
                    if bool(motion_env.any()):
                        action[motion_env] = self.models.policy(
                            history[motion_env], goal[motion_env]
                        )
                    if bool(self.velocity_env.any()):
                        action[self.velocity_env] = self.models.velocity_policy(
                            history[self.velocity_env],
                            normalize_command(self.command[self.velocity_env]),
                        )
                else:
                    action = self.models.policy(history, goal)
            executed = action + self.action_noise_std * torch.randn_like(action)
            before = self.env.state.clone()
            time_in = self.env.time_in_episode.clone()
            mid = self.env.motion_id.clone()
            mt = self.env.motion_time.clone()
            xi = self.env.xi.clone()
            self._sync()
            t_sim = time.perf_counter()
            result = self.env.step(executed)
            self._sync()
            sim_s += time.perf_counter() - t_sim
            next_state = result["next_state"]
            failure = result["motion_failure"]
            if self.velocity and bool(self.velocity_env.any()):
                fall = velocity_fall(next_state)
                failure = torch.where(self.velocity_env, fall, failure)
            end_kind = compose_end_kind(failure, result["timeout"], result["clip_end"])
            self.models.dynamics.update_norm(
                before,
                executed,
                xi,
                next_state,
                self.residual_frame,
                parallel=self.dp,
            )
            reward = analytic_reward(
                next_state,
                result["ref_state"],
                action,
                self.prev_action,
                self.joint_low,
                self.joint_high,
                reward_anchor=self.reward_anchor,
            )["reward"]
            if self.velocity and bool(self.velocity_env.any()):
                reward = reward.clone()
                reward[self.velocity_env] = velocity_reward(
                    next_state[self.velocity_env],
                    self.command[self.velocity_env],
                    action[self.velocity_env],
                    self.prev_action[self.velocity_env],
                    self.joint_low,
                    self.joint_high,
                )
            self.reset_sampler.log_step(
                reward, mid, mt, rows=~self.velocity_env if self.velocity else None
            )
            reward_total = reward_total + reward.detach().sum()
            env_ids = torch.arange(self.num_envs, device=self.device)
            self.buffer.add(
                env_ids,
                before,
                next_state,
                action,
                executed,
                reward,
                end_kind,
                xi,
                mid,
                mt,
                time_in,
                prev_action=self.prev_action,
                obs_noisy=current_obs,
                task_id=self.velocity_env.long(),
                command=self.command,
                command_from_ref=self.command_from_ref,
            )
            self.prev_action = action.detach()
            finished = end_kind != EPISODE_CONT
            live = (~finished).nonzero(as_tuple=False).squeeze(-1)
            if live.numel():
                self._append_obs(live)
            if finished.any():
                done_ids = finished.nonzero(as_tuple=False).squeeze(-1)
                episodes += int(done_ids.numel())
                failures += int((end_kind == EPISODE_FAILURE).sum())
                done_motion = done_ids
                if self.velocity:
                    done_motion = done_ids[~self.velocity_env[done_ids]]
                self._update_reset_bins(done_motion, end_kind[done_motion])
                self.env.reset(done_ids, *self._reset_points(done_ids))
                self._update_commands(done_ids, force=True)
                self.prev_action[done_ids] = 0
                self._reset_history(done_ids)
            self.reset_sampler.sync(self.dp)
            self.sim_vector_steps += 1
            self.env_transitions += self.num_envs
            if accumulate:
                self.step_acc += self.updates_per_env_step
        valid = self.buffer.num_valid_windows(self.k)
        self.time_sim = sim_s
        bins = self.reset_sampler.bin_return
        reward_mean, bin_mean, bin_min = _host_floats(
            [
                reward_total / max(n_steps * self.num_envs, 1),
                bins.mean(),
                bins.min(),
            ]
        )
        return {
            "sim/terminal_rate": failures / max(n_steps * self.num_envs, 1),
            "sim/reward_mean": reward_mean,
            "sim/episodes_finished": float(episodes),
            "reset/bin_return_mean": bin_mean,
            "reset/bin_return_min": bin_min,
            "buffer/valid_windows": float(valid),
            "time/sim_s": float(sim_s),
        }

    def _reset_points(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return-based starts for motion envs, uniform bins for velocity envs."""
        if not self.velocity:
            return self.reset_sampler.sample(int(env_ids.numel()))
        motion_id = torch.empty_like(env_ids)
        motion_time = torch.empty_like(env_ids)
        is_velocity = self.velocity_env[env_ids]
        for rows, draw in (
            (~is_velocity, self.reset_sampler.sample),
            (is_velocity, self.reset_sampler.sample_uniform),
        ):
            count = int(rows.sum())
            if count:
                motion_id[rows], motion_time[rows] = draw(count)
        return motion_id, motion_time

    def _update_reset_bins(self, env_ids, end_kind) -> None:
        """Fold finished motion episodes into the reset bins.

        Timeout tails bootstrap with the motion critic; failure and clip end
        do not. This is the reset-bin return only. Actor GAE in the
        imagination loop is a separate path.
        """
        if env_ids.numel() == 0:
            return
        bootstrap = None
        if bool((end_kind == EPISODE_TIMEOUT).any()):
            bootstrap = self._bootstrap_value(env_ids)
        self.reset_sampler.finish(env_ids, end_kind, bootstrap)

    def _bootstrap_value(self, env_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._critic_bootstrap(env_ids)

    def _motion_critic_value(self, state, motion_id, motion_time, xi) -> torch.Tensor:
        goal = self._goal(motion_id, motion_time, state)
        ref = self.motion.state_at(motion_id, motion_time)
        return self.models.critic(
            motion_critic_input(
                state,
                goal,
                xi,
                ref,
                self.critic_features,
                scale_xi=self.scale_xi,
            )
        )

    def _critic_bootstrap(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Motion critic at the terminal state and reference time.

        ``env.motion_time`` has already advanced to the terminal frame.
        """
        return self._motion_critic_value(
            self.env.state[env_ids],
            self.env.motion_id[env_ids],
            self.env.motion_time[env_ids],
            self.env.xi[env_ids],
        )

    def _note_nonfinite(self, where: str) -> None:
        self.nonfinite_skips += 1
        self.consecutive_nonfinite += 1
        if self.consecutive_nonfinite > self.max_nonfinite_skips:
            raise RuntimeError(
                f"non-finite gradients in {self.consecutive_nonfinite} consecutive "
                f"optimizer steps (last: {where}); stopping instead of training on NaN"
            )

    def _dp_step(
        self, optimizer, params, weight: float, clip, where: str, mean: bool = True
    ):
        """Reduce, check, clip and step one optimizer on every rank alike.

        ``weight`` is the local sample count behind the gradient. Returns
        whether the step was taken and the pre-clip norm.
        """
        params = list(params)
        total, finite = self.dp.reduce_grads(params, weight, mean=mean)
        if total <= 0:
            optimizer.zero_grad(set_to_none=True)
            return False, 0.0
        if not finite:
            optimizer.zero_grad(set_to_none=True)
            self._note_nonfinite(where)
            return False, float("nan")
        self.consecutive_nonfinite = 0
        norm = clip(params)
        optimizer.step()
        return True, norm

    def fit_world_model(self) -> dict[str, float]:
        """Fit dynamics and the termination head on ``wm_batches`` k-step windows."""
        sample_s = 0.0
        update_s = 0.0
        _set_requires_grad(self.models.wm_params(), True)
        last_loss = float("nan")
        last_norm: torch.Tensor | float = 0.0
        one_step = float("nan")
        k_step = float("nan")
        held_loss: torch.Tensor | None = None
        for _ in range(self.wm_batches):
            t_sample = time.perf_counter()
            batch = self.buffer.sample_windows(self.k, self.wm_window_batch)
            sample_s += time.perf_counter() - t_sample
            self._sync()
            t_update = time.perf_counter()
            self.opt_wm.zero_grad(set_to_none=True)
            weight = 0
            if batch is not None:
                weight = int(batch["state"].shape[0])
                if self.velocity:
                    loss, one, last = self._world_model_open_loop(batch)
                else:
                    loss, one, last = self._world_model_scan(batch)
                loss.backward()
                held_loss = torch.stack([loss.detach(), one.detach(), last.detach()])
            stepped, norm = self._dp_step(
                self.opt_wm,
                self.models.wm_params(),
                weight,
                lambda params: torch.nn.utils.clip_grad_norm_(params, 1.0),
                "world model",
            )
            if stepped:
                self.wm_optimizer_steps += 1
                last_norm = norm
            self._sync()
            update_s += time.perf_counter() - t_update
        if held_loss is not None:
            pieces: list[torch.Tensor | float] = [
                held_loss[0],
                held_loss[1],
                held_loss[2],
                last_norm,
            ]
            last_loss, one_step, k_step, last_norm = _host_floats(pieces)
        self.time_replay_sample = sample_s
        self.time_world_model = update_s
        return {
            "wm/state_loss": last_loss,
            "wm/grad_norm": last_norm,
            "wm/one_step_loss": one_step,
            "wm/k_step_loss": k_step,
        }

    def _world_model_runner(self):
        name = (
            f"wm-{self.k}-{self.wm_state_loss}-{self.residual_frame}-"
            f"{int(self.learned_termination)}-{self.termination_body}"
        )
        if getattr(self, "_wm_scan_name", None) != name:
            self._wm_scan_name = name
            self._wm_scan_fn = make_dynamics_window(
                self.k,
                self.wm_state_loss == "scaled",
                self.learned_termination,
                self.termination_body,
                self.residual_frame,
                self.wm_state_loss,
            )
        return name, self._wm_scan_fn

    def _world_model_scan(self, batch: dict[str, torch.Tensor]):
        """One compiled k-step window. Every row is a motion sample."""
        if self.learned_termination:
            refs = torch.stack(
                [
                    self.motion.state_at(
                        batch["motion_id"], batch["motion_time"] + (t + 1)
                    )
                    for t in range(self.k)
                ],
                dim=1,
            )
        else:
            refs = batch["state"][:, 1:]
        name, fn = self._world_model_runner()
        return run_compiled(
            name,
            fn,
            self.models.dynamics,
            self.models.termination,
            batch["state"][:, 0],
            batch["action_executed"],
            batch["xi"],
            batch["state"][:, 1:],
            refs,
            batch["end_kind"],
            self.bce_weight,
        )

    def _world_model_open_loop(self, batch: dict[str, torch.Tensor]):
        """Velocity mix: BCE only on motion rows, so the scan's mean would differ."""
        pred = batch["state"][:, 0]
        loss = pred.new_zeros(())
        one = pred.new_zeros(())
        last = pred.new_zeros(())
        for t in range(self.k):
            pred = integrate_residual(
                pred,
                self.models.dynamics(pred, batch["action_executed"][:, t], batch["xi"]),
                self.residual_frame,
            )
            true = batch["state"][:, t + 1]
            if self.wm_state_loss == "running":
                step = running_state_loss(
                    pred, true, self.models.dynamics, self.residual_frame
                )
            else:
                step = state_prediction_loss(
                    pred, true, scaled=self.wm_state_loss == "scaled"
                )
            if t == 0:
                one = step
            last = step
            loss = loss + step
            motion_rows = batch["task_id"] == 0
            if self.learned_termination and bool(motion_rows.any()):
                height, tilt, _ = failure_signals(
                    pred[motion_rows],
                    self.motion.state_at(
                        batch["motion_id"][motion_rows],
                        batch["motion_time"][motion_rows] + t + 1,
                    ),
                    termination_body=self.termination_body,
                )
                logit = self.models.termination(termination_head_input(height, tilt))
                loss = loss + self.bce_weight * termination_bce(
                    logit, batch["end_kind"][motion_rows, t]
                )
        return loss, one, last

    def imagine_and_update_actor(self) -> dict[str, float]:
        """Imagine H steps from replay starts, update the actor, then fit the critic."""
        t_sample = time.perf_counter()
        motion_batch = self.imagine_batch // 2 if self.velocity else self.imagine_batch
        starts = self.buffer.sample_starts(
            motion_batch, self.q0, task=0 if self.velocity else None
        )
        self.time_replay_sample += time.perf_counter() - t_sample
        self._sync()
        t_bptt = time.perf_counter()
        _set_requires_grad(self.models.wm_params(), False)
        _set_requires_grad(self.models.critic_params(), False)
        _set_requires_grad(self.models.actor_params(), True)
        self.opt_actor.zero_grad(set_to_none=True)
        metrics: dict[str, float] = {
            "actor/objective": float("nan"),
            "actor/grad_norm": 0.0,
            "imagination/survival": 0.0,
        }
        rollout = None
        if starts is not None:
            rollout = self._imagine_motion(starts)
            (-rollout["objective"]).backward()
        norms = {"encoder": 0.0, "decoder": 0.0}

        def clip(params):
            # The decoder's last layer has gain 0.01, so the encoder sees a
            # much smaller gradient than the decoder itself. Both are logged.
            norms["encoder"] = _grad_norm(self.models.encoder.parameters())
            norms["decoder"] = _grad_norm(self.models.decoder.parameters())
            return _clip_actor_grads(params)

        stepped, grad_norm = self._dp_step(
            self.opt_actor,
            self.models.actor_params(),
            0 if rollout is None else rollout["states"].shape[0],
            clip,
            "actor",
        )
        if stepped:
            self.actor_optimizer_steps += 1
        self._sync()
        self.time_bptt = time.perf_counter() - t_bptt
        t_critic = time.perf_counter()
        if rollout is None:
            critic_metrics = self.fit_critic(*self._empty_motion_rollout())
        else:
            critic_metrics = self.fit_critic(
                rollout["features"].detach(),
                rollout["rewards"].detach(),
                rollout["d"].detach(),
            )
        self.time_critic = time.perf_counter() - t_critic
        if rollout is not None:
            d_t = rollout["d"].detach()
            (
                objective,
                grad_norm,
                encoder_norm,
                decoder_norm,
                survival,
                termination_prob,
                reward_mean,
            ) = _host_floats(
                [
                    rollout["objective"].detach(),
                    grad_norm,
                    norms["encoder"],
                    norms["decoder"],
                    (1.0 - d_t).prod(dim=1).mean(),
                    d_t.mean(),
                    rollout["rewards"].detach().mean(),
                ]
            )
            metrics = {
                "actor/objective": objective,
                "actor/grad_norm": grad_norm,
                "actor/encoder_grad_norm": encoder_norm,
                "actor/decoder_grad_norm": decoder_norm,
                "imagination/survival": survival,
                "imagination/termination_prob": termination_prob,
                "imagination/reward": reward_mean,
            }
        metrics.update(critic_metrics)
        if self.velocity:
            metrics.update(self._velocity_imagine())
        return metrics

    def _imagination_runner(self):
        name = (
            f"imagine-{self.H}-{int(self.learned_termination)}-{self.termination_body}-"
            f"{self.critic_features}-{int(self.include_ref_ori)}-{self.obs_dim}-"
            f"{self.reward_anchor}-{self.residual_frame}-{int(self.scale_xi)}-"
            f"{self.gamma}"
        )
        if getattr(self, "_imagine_scan_name", None) != name:
            self._imagine_scan_name = name
            self._imagine_scan_fn = make_imagination(
                self.H,
                self.learned_termination,
                self.termination_body,
                self.critic_features,
                self.include_ref_ori,
                self.obs_dim,
                self.gamma,
                self.reward_anchor,
                self.residual_frame,
                self.scale_xi,
            )
        return name, self._imagine_scan_fn

    def _body_fk(self, state, motion_id, motion_time):
        """Predicted FK with grad, and cached FK of the reference frame."""
        return differentiable_fk(state), self.motion.kinematics_at(motion_id, motion_time)

    def _prefetch_rollout(self, motion_id: torch.Tensor, time0: torch.Tensor):
        """Reference states, goals and body poses for times ``t .. t+H``.

        Goals ignore the predicted state. One batched query covers every
        future frame; the Python loop over the horizon is not repeated.
        """
        steps = torch.arange(self.H + 1, device=time0.device)
        times = time0.unsqueeze(1) + steps.to(dtype=time0.dtype)
        ids = motion_id.unsqueeze(1).expand(-1, self.H + 1)
        flat_id = ids.reshape(-1)
        flat_time = times.reshape(-1)
        flat = self.motion.state_at(flat_id, flat_time)
        refs = flat.view(motion_id.shape[0], self.H + 1, flat.shape[-1])
        goal = self._goal(flat_id, flat_time, None)
        goals = goal.reshape(motion_id.shape[0], self.H + 1, goal.shape[-1])
        kin = self.motion.kinematics_at(flat_id, flat_time)
        kin = tuple(
            part.reshape(motion_id.shape[0], self.H + 1, *part.shape[1:]).contiguous()
            for part in kin
        )
        return refs, goals, kin

    def _imagine_motion_scanned(
        self, starts: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Same rollout as the open loop, one compiled scan on a large CUDA batch."""
        motion_id = starts["motion_id"]
        time0 = starts["motion_time"]
        refs, goals, ref_kin = self._prefetch_rollout(motion_id, time0)
        name, fn = self._imagination_runner()
        states, goals_t, features, rewards, d_hat, objective = run_compiled(
            name,
            fn,
            self.models.encoder,
            self.models.decoder,
            self.models.dynamics,
            self.models.termination,
            self.models.critic,
            starts["state"],
            starts["prev_action"],
            starts["xi"],
            starts["history"],
            refs,
            goals,
            *ref_kin,
            self.q0,
            self.joint_low,
            self.joint_high,
        )
        return {
            "states": states,
            "goals": goals_t,
            "features": features,
            "rewards": rewards,
            "d": d_hat,
            "objective": objective,
        }

    def _imagine_motion(
        self, starts: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # Goals that depend on the imagined state stay on the open loop.
        # The LAFAN1 runs do not; their goals are a function of the clip only.
        if not self.goal_anchor:
            return self._imagine_motion_scanned(starts)
        state = starts["state"]
        prev = starts["prev_action"]
        xi = starts["xi"]
        history = starts["history"]
        motion_id = starts["motion_id"]
        time0 = starts["motion_time"]
        # One rollout keeps s_0..s_H, g_0..g_H, r_1..r_H and d_1..d_H.
        # history already ends at s_0, so the policy sees it before the step.
        states, goals, features, rewards, d_hats = [], [], [], [], []
        # The reward at step t and the critic features at step t+1 read the
        # same predicted state and the same reference frame. One walk feeds both.
        carried_state = None
        carried_fk = None
        for t in range(self.H):
            if carried_state is state:
                ref, state_fk, ref_fk = carried_fk
            else:
                ref = self.motion.state_at(motion_id, time0 + t)
                state_fk, ref_fk = self._body_fk(state, motion_id, time0 + t)
            goal = self._goal(motion_id, time0 + t, state)
            states.append(state)
            goals.append(goal)
            features.append(
                motion_critic_input(
                    state,
                    goal,
                    xi,
                    ref,
                    self.critic_features,
                    state_fk,
                    ref_fk,
                    scale_xi=self.scale_xi,
                )
            )
            # Stored frames keep the noise they were given at collection.
            # Only the frame produced by this imagined step draws new noise.
            action = self.models.policy(history, goal)
            state = integrate_residual(
                state, self.models.dynamics(state, action, xi), self.residual_frame
            )
            ref_next = self.motion.state_at(motion_id, time0 + t + 1)
            state_fk, ref_fk = self._body_fk(state, motion_id, time0 + t + 1)
            carried_state = state
            carried_fk = (ref_next, state_fk, ref_fk)
            reward_pack = analytic_reward(
                state,
                ref_next,
                action,
                prev,
                self.joint_low,
                self.joint_high,
                state_fk,
                ref_fk,
                reward_anchor=self.reward_anchor,
            )
            height, tilt, _ = failure_signals(
                state, ref_next, termination_body=self.termination_body
            )
            d_hat = self._termination_prob(height, tilt)
            rewards.append(reward_pack["reward"])
            d_hats.append(d_hat)
            ref_quat = ref_next[..., 3:7] if self.include_ref_ori else None
            obs_next = add_sensor_noise(
                analytic_obs(state, action, xi, self.q0, ref_quat=ref_quat), True
            )
            hist2d = history.reshape(-1, HISTORY_LEN, self.obs_dim)
            history = torch.cat((hist2d[:, 1:], obs_next.unsqueeze(1)), dim=1).reshape(
                history.shape
            )
            prev = action
        if carried_state is state:
            ref_h, state_fk, ref_fk = carried_fk
        else:
            ref_h = self.motion.state_at(motion_id, time0 + self.H)
            state_fk, ref_fk = self._body_fk(state, motion_id, time0 + self.H)
        goal_h = self._goal(motion_id, time0 + self.H, state)
        states.append(state)
        goals.append(goal_h)
        features.append(
            motion_critic_input(
                state,
                goal_h,
                xi,
                ref_h,
                self.critic_features,
                state_fk,
                ref_fk,
                scale_xi=self.scale_xi,
            )
        )
        states_t = torch.stack(states, dim=1)
        goals_t = torch.stack(goals, dim=1)
        feat_t = torch.stack(features, dim=1)
        rewards_t = torch.stack(rewards, dim=1)
        d_t = torch.stack(d_hats, dim=1)
        v_last = self.models.critic(feat_t[:, -1])
        return {
            "states": states_t,
            "goals": goals_t,
            "features": feat_t,
            "rewards": rewards_t,
            "d": d_t,
            "objective": actor_objective(rewards_t, d_t, v_last, self.gamma),
        }

    def _termination_prob(
        self, height: torch.Tensor, tilt: torch.Tensor
    ) -> torch.Tensor:
        """d_hat of the motion task: the learned head (paper) or the soft threshold."""
        if self.learned_termination:
            return torch.sigmoid(
                self.models.termination(termination_head_input(height, tilt))
            )
        return analytic_failure_prob(height, tilt)

    def _empty_motion_rollout(self) -> tuple[torch.Tensor, ...]:
        """Zero-row critic inputs for a rank that drew no start this update."""

        def zeros(*shape):
            return torch.zeros(0, *shape, device=self.device)

        width = self.models.critic.net[0].in_features
        return (
            zeros(self.H + 1, width),
            zeros(self.H),
            zeros(self.H),
        )

    def _update_commands(self, env_ids: torch.Tensor, force: bool) -> None:
        if not self.velocity or env_ids.numel() == 0:
            return
        if not force:
            self.command_age[env_ids] += 1
        due = force | (self.command_age[env_ids] >= RESAMPLE_STEPS)
        if bool(due.any()):
            rows = env_ids[due]
            drawn, from_ref = sample_velocity_command(int(rows.numel()), self.device)
            ref = self.motion.reference_velocity_command(
                self.env.motion_id[rows], self.env.motion_time[rows]
            )
            drawn = torch.where(from_ref.unsqueeze(-1), ref, drawn)
            self.command[rows] = drawn
            self.command_from_ref[rows] = from_ref
            self.command_age[rows] = 0
        follow = env_ids[self.command_from_ref[env_ids]]
        if follow.numel():
            self.command[follow] = self.motion.reference_velocity_command(
                self.env.motion_id[follow], self.env.motion_time[follow]
            )

    def _velocity_imagine(self) -> dict[str, float]:
        """Velocity RL on its own starts, reward, critic, and alignment loss."""
        starts = self.buffer.sample_starts(self.imagine_batch // 2, self.q0, task=1)
        _set_requires_grad(self.models.wm_params(), False)
        _set_requires_grad(self.models.actor_params(), False)
        _set_requires_grad(self.models.velocity_actor_params(), True)
        _set_requires_grad(self.models.velocity_critic_params(), False)
        self.opt_velocity.zero_grad(set_to_none=True)
        local = [0.0, 0.0]
        if starts is not None:
            local = [
                float(starts["state"].shape[0]),
                float(starts["command_from_ref"].sum()),
            ]
        # The RL term averages over every start and the alignment term over
        # the from_ref starts, so each rank scales by its share of each.
        rows_total, ref_total = self.dp.all_reduce(local)
        if starts is None:
            self._dp_step(
                self.opt_velocity,
                self.models.velocity_actor_params(),
                0,
                _clip_actor_grads,
                "velocity actor",
                mean=False,
            )
            self._fit_velocity_critic(*self._empty_velocity_rollout())
            _set_requires_grad(self.models.actor_params(), True)
            return {
                "velocity/objective": float("nan"),
                "velocity/grad_norm": 0.0,
                "velocity/align": 0.0,
            }
        state = starts["state"]
        prev = starts["prev_action"]
        xi = starts["xi"]
        history = starts["history"]
        motion_id = starts["motion_id"]
        time0 = starts["motion_time"]
        command = starts["command"]
        from_ref = starts["command_from_ref"]
        states, commands, rewards, d_hats = [], [], [], []
        align_terms = []
        for t in range(self.H):
            ref_cmd = self.motion.reference_velocity_command(motion_id, time0 + t)
            command = torch.where(from_ref.unsqueeze(-1), ref_cmd, command)
            norm = normalize_command(command)
            states.append(state)
            commands.append(command)
            z_vel = self.models.velocity_encoder(norm, history)
            if bool(from_ref.detach().any()):
                z_motion = self.models.encoder(
                    self._goal(motion_id, time0 + t, state)
                ).detach()
                align_terms.append((z_vel[from_ref] - z_motion[from_ref]).pow(2).mean())
            action = self.models.decoder(z_vel, history)
            state = integrate_residual(
                state, self.models.dynamics(state, action, xi), self.residual_frame
            )
            rewards.append(
                velocity_reward(
                    state, command, action, prev, self.joint_low, self.joint_high
                )
            )
            d_hats.append(velocity_fall_soft(state))
            ref_next = self.motion.state_at(motion_id, time0 + t + 1)
            ref_quat = ref_next[..., 3:7] if self.include_ref_ori else None
            obs_next = add_sensor_noise(
                analytic_obs(state, action, xi, self.q0, ref_quat=ref_quat), True
            )
            hist2d = history.reshape(-1, HISTORY_LEN, self.obs_dim)
            history = torch.cat((hist2d[:, 1:], obs_next.unsqueeze(1)), dim=1).reshape(
                history.shape
            )
            prev = action
        ref_cmd = self.motion.reference_velocity_command(motion_id, time0 + self.H)
        command = torch.where(from_ref.unsqueeze(-1), ref_cmd, command)
        states.append(state)
        commands.append(command)
        states_t = torch.stack(states, dim=1)
        commands_t = torch.stack(commands, dim=1)
        rewards_t = torch.stack(rewards, dim=1)
        d_t = torch.stack(d_hats, dim=1)
        v_last = self.models.velocity_critic(states_t[:, -1], commands_t[:, -1], xi)
        objective = actor_objective(rewards_t, d_t, v_last, self.gamma)
        align = (
            torch.stack(align_terms).mean() if align_terms else objective.new_zeros(())
        )
        diagnostics: dict[str, float] = {}
        if self.grad_diagnostics:
            params = self.models.velocity_actor_params()
            reward_part = actor_objective(
                rewards_t, d_t, torch.zeros_like(v_last), self.gamma
            )
            diagnostics = {
                "velocity/grad_norm_reward": _term_grad_norm(-reward_part, params),
                "velocity/grad_norm_bootstrap": _term_grad_norm(
                    -(objective - reward_part), params
                ),
                "velocity/grad_norm_align": _term_grad_norm(
                    ALIGN_WEIGHT * align, params
                ),
            }
        rl_share = local[0] / rows_total
        align_share = local[1] / ref_total if ref_total > 0 else 0.0
        (-rl_share * objective + ALIGN_WEIGHT * align_share * align).backward()
        _, grad_norm = self._dp_step(
            self.opt_velocity,
            self.models.velocity_actor_params(),
            local[0],
            _clip_actor_grads,
            "velocity actor",
            mean=False,
        )
        self._fit_velocity_critic(
            states_t.detach(),
            commands_t.detach(),
            xi.detach(),
            rewards_t.detach(),
            d_t.detach(),
        )
        _set_requires_grad(self.models.actor_params(), True)
        planar, yaw_rate = heading_velocity(states_t[:, 1:].detach())
        commands_next = commands_t[:, 1:].detach()
        (
            objective_v,
            grad_norm,
            align_v,
            survival,
            fall_prob,
            reward_v,
            lin_err,
            yaw_err,
        ) = _host_floats(
            [
                objective.detach(),
                grad_norm,
                align.detach(),
                (1.0 - d_t.detach()).prod(dim=1).mean(),
                d_t.detach().mean(),
                rewards_t.detach().mean(),
                (planar - commands_next[..., :2]).norm(dim=-1).mean(),
                (yaw_rate - commands_next[..., 2]).abs().mean(),
            ]
        )
        return {
            "velocity/objective": objective_v,
            "velocity/grad_norm": grad_norm,
            "velocity/align": align_v,
            "velocity/survival": survival,
            "velocity/fall_prob": fall_prob,
            "velocity/reward": reward_v,
            "velocity/imagined_lin_err": lin_err,
            "velocity/imagined_yaw_err": yaw_err,
            **diagnostics,
        }

    def _empty_velocity_rollout(self) -> tuple[torch.Tensor, ...]:
        state_dim = self.buffer.state.shape[-1]
        xi_dim = self.buffer.xi.shape[-1]
        return (
            torch.zeros(0, self.H + 1, state_dim, device=self.device),
            torch.zeros(0, self.H + 1, 3, device=self.device),
            torch.zeros(0, xi_dim, device=self.device),
            torch.zeros(0, self.H, device=self.device),
            torch.zeros(0, self.H, device=self.device),
        )

    def _critic_chunks(self, batch: int) -> list[torch.Tensor]:
        """``critic_minibatches`` index chunks, possibly empty, per epoch.

        Every rank takes the same number of optimizer steps whatever its
        local batch size.
        """
        perm = torch.randperm(batch, device=self.device)
        return list(torch.tensor_split(perm, max(self.critic_minibatches, 1)))

    def _fit_velocity_critic(self, states_t, commands_t, xi, rewards_t, d_t) -> None:
        _set_requires_grad(self.models.velocity_critic_params(), True)
        batch = states_t.shape[0]
        targets = None
        self.models.velocity_critic.update_norm(
            states_t.reshape(-1, states_t.shape[-1]),
            commands_t.reshape(-1, commands_t.shape[-1]),
            xi.unsqueeze(1).expand(-1, states_t.shape[1], -1).reshape(-1, xi.shape[-1]),
            parallel=self.dp,
        )
        if batch > 0:
            with torch.no_grad():
                flat_xi = (
                    xi.unsqueeze(1).expand(-1, self.H + 1, -1).reshape(-1, xi.shape[-1])
                )
                values = self.models.velocity_critic(
                    states_t.reshape(-1, states_t.shape[-1]),
                    commands_t.reshape(-1, commands_t.shape[-1]),
                    flat_xi,
                ).reshape(batch, self.H + 1)
                targets = gae_targets(rewards_t, d_t, values, self.gamma, self.lam)
        for _ in range(self.critic_epochs):
            for idx in self._critic_chunks(batch):
                self.opt_velocity_critic.zero_grad(set_to_none=True)
                if idx.numel():
                    pred = self.models.velocity_critic(
                        states_t[idx, : self.H].reshape(-1, states_t.shape[-1]),
                        commands_t[idx, : self.H].reshape(-1, commands_t.shape[-1]),
                        xi[idx]
                        .unsqueeze(1)
                        .expand(-1, self.H, -1)
                        .reshape(-1, xi.shape[-1]),
                    ).reshape(idx.numel(), self.H)
                    loss = (pred - targets[idx]).pow(2).mean()
                    loss.backward()
                stepped, _ = self._dp_step(
                    self.opt_velocity_critic,
                    self.models.velocity_critic_params(),
                    idx.numel(),
                    lambda params: torch.nn.utils.clip_grad_norm_(params, 1.0),
                    "velocity critic",
                )
                if stepped:
                    self.critic_optimizer_steps += 1
        _set_requires_grad(self.models.velocity_critic_params(), False)

    def fit_critic(
        self,
        features_t: torch.Tensor,
        rewards_t: torch.Tensor,
        d_t: torch.Tensor,
    ) -> dict[str, float]:
        """Regress V onto R_t from the same imagined trajectory.

        ``features_t`` is ``[B, H+1, F]``. Targets are ``R_0 .. R_{H-1}``.
        ``B`` may be 0 on a rank that drew no start; it still joins every
        optimizer step.
        """
        _set_requires_grad(self.models.critic_params(), True)
        _set_requires_grad(self.models.actor_params(), False)
        _set_requires_grad(self.models.wm_params(), False)
        width = self.models.critic.net[0].in_features
        if (
            features_t.shape[1] != self.H + 1
            or features_t.shape[-1] != width
            or rewards_t.shape[1] != self.H
        ):
            raise RuntimeError(
                "critic targets must align with s_0..s_{H-1} from one rollout, "
                f"got features {tuple(features_t.shape)} and rewards {tuple(rewards_t.shape)}"
            )
        batch = features_t.shape[0]
        targets = None
        value_mean = target_mean = float("nan")
        held_stats: torch.Tensor | None = None
        if self.models.critic.norm is not None:
            self.dp.update_norm(self.models.critic.norm, features_t.reshape(-1, width))
        if batch > 0:
            with torch.no_grad():
                values = self.models.critic(features_t.reshape(-1, width)).reshape(
                    batch, self.H + 1
                )
                targets = gae_targets(rewards_t, d_t, values, self.gamma, self.lam)
                held_stats = torch.stack([values[:, : self.H].mean(), targets.mean()])
        feat_fit = features_t[:, : self.H]
        last_loss = float("nan")
        held_loss: torch.Tensor | None = None
        for _ in range(self.critic_epochs):
            for idx in self._critic_chunks(batch):
                self.opt_critic.zero_grad(set_to_none=True)
                if idx.numel():
                    pred = self.models.critic(feat_fit[idx].reshape(-1, width)).reshape(
                        idx.numel(), self.H
                    )
                    loss = (pred - targets[idx]).pow(2).mean()
                    loss.backward()
                    held_loss = loss.detach()
                stepped, _ = self._dp_step(
                    self.opt_critic,
                    self.models.critic_params(),
                    idx.numel(),
                    lambda params: torch.nn.utils.clip_grad_norm_(params, 1.0),
                    "critic",
                )
                if stepped:
                    self.critic_optimizer_steps += 1
        pieces: list[torch.Tensor | float] = []
        if held_loss is not None:
            pieces.append(held_loss)
        if held_stats is not None:
            pieces.extend([held_stats[0], held_stats[1]])
        if pieces:
            hosted = _host_floats(pieces)
            cursor = 0
            if held_loss is not None:
                last_loss = hosted[cursor]
                cursor += 1
            if held_stats is not None:
                value_mean, target_mean = hosted[cursor], hosted[cursor + 1]
        self._sync()
        _set_requires_grad(self.models.actor_params(), True)
        return {
            "critic/loss": last_loss,
            "critic/value_mean": value_mean,
            "critic/target_mean": target_mean,
        }

    def maybe_update(self) -> dict[str, float]:
        """Run every update the update-to-data ratio owes, on all ranks together."""
        metrics: dict[str, float] = {}
        updated = False
        while self.step_acc >= 1.0:
            # Ranks without a window yet still join every collective with
            # weight 0, so only the global count gates the update.
            local = self.buffer.num_valid_windows(self.k)
            if self.dp.all_reduce([float(local)])[0] == 0:
                break
            updated = True
            metrics.update(self.fit_world_model())
            # The first wm_only_updates iterations match zita-ch: the world
            # model trains, and the actor and critic wait. Actor and critic
            # schedulers step only with their optimizers. Over the 250k
            # cosine those 500 steps change the rate by far less than 1%.
            # The paper default is 0, which updates the actor every iteration.
            actor_updated = self.global_update >= self.wm_only_updates
            metrics["train/wm_only"] = 0.0 if actor_updated else 1.0
            if actor_updated:
                metrics.update(self.imagine_and_update_actor())
            self.sched_wm.step()
            if actor_updated:
                self.sched_actor.step()
                self.sched_critic.step()
                if self.sched_velocity is not None:
                    self.sched_velocity.step()
                    self.sched_velocity_critic.step()
            self.global_update += 1
            self.step_acc -= 1.0
        interval = int(self.cfg.algorithm.get("param_check_interval", 100))
        if (
            updated
            and self.dp.enabled
            and interval > 0
            and self.global_update % interval == 0
        ):
            metrics["train/param_divergence"] = self.dp.param_divergence(self.models)
        metrics["train/nonfinite_skips"] = float(self.nonfinite_skips)
        metrics["train/global_update"] = float(self.global_update)
        metrics["sim_vector_steps"] = float(self.sim_vector_steps)
        metrics["env_transitions"] = float(self.env_transitions)
        metrics["wm_optimizer_steps"] = float(self.wm_optimizer_steps)
        metrics["actor_optimizer_steps"] = float(self.actor_optimizer_steps)
        metrics["critic_optimizer_steps"] = float(self.critic_optimizer_steps)
        metrics["time/replay_sample_s"] = float(self.time_replay_sample)
        metrics["time/world_model_s"] = float(self.time_world_model)
        metrics["time/bptt_s"] = float(self.time_bptt)
        metrics["time/critic_s"] = float(self.time_critic)
        metrics["time/sim_s"] = float(self.time_sim)
        metrics.update(gpu_memory_metrics(self.device, self.platform))
        metrics["memory/host_rss"] = _host_rss_bytes()
        return metrics

    def run_iteration(self) -> dict[str, float]:
        # Resume restores weights only. The new replay still needs the warmup.
        """Warm the replay once, collect one vector step, and run owed updates."""
        if not self.replay_warm:
            if self.warmup > 0:
                self.collect_steps(self.warmup, accumulate=False)
            self.replay_warm = True
        metrics = self.collect_steps(1)
        metrics.update(self.maybe_update())
        metrics["time/comm_s"] = float(self.dp.comm_seconds)
        self.dp.comm_seconds = 0.0
        return metrics

    def run_until_updates(self, n_updates: int) -> dict[str, float]:
        """Iterate until ``global_update`` reaches ``n_updates``."""
        metrics: dict[str, float] = {}
        while self.global_update < n_updates:
            metrics = self.run_iteration()
        return metrics

    def evaluate(self) -> dict[str, float]:
        """Every clip once, from frame 0, in batches of ``num_envs``.

        Only rank 0 evaluates; the other ranks hold the same weights.
        """
        if not self.dp.is_main:
            return {}
        if self._eval_env is None:
            # One eval simulator for the whole run, not a new one per evaluation.
            self._eval_env = make_env(
                str(self.cfg.env.backend),
                num_envs=max(1, min(self.num_envs, int(self.motion.num_clips))),
                motion=self.motion,
                device=self.device,
                max_episode_seconds=1.0e6,
                termination_body=str(self.cfg.algorithm.termination_body),
                eval_mode=True,
            )
        rows = motion_eval_rows(
            self.cfg,
            self.motion,
            self.models,
            self._goal,
            self.device,
            self.num_envs,
            env=self._eval_env,
        )
        return summarize_tracking_eval(rows)

    def _checkpoint_meta(self) -> dict:
        return {
            "algorithm": "supermimic",
            "world_size": self.dp.world_size,
            "global_update": self.global_update,
            "goal_dim": self.goal_dim,
            "obs_dim": self.obs_dim,
            "goal_encoding": self.goal_encoding,
            "goal_window": list(GOAL_WINDOW[self.goal_encoding]),
            "velocity_tracking": self.velocity,
            "critic_features": self.critic_features,
            "wm_state_loss": self.wm_state_loss,
            "reward_anchor": self.reward_anchor,
            "input_norm": self.input_norm,
            "residual_frame": self.residual_frame,
            "matmul": "bf16" if self.models.bf16_matmul else "tf32",
            "termination_head": "learned" if self.learned_termination else "analytic",
            "termination_bias_init": self.cfg.algorithm.get(
                "termination_bias_init", None
            ),
            "num_clips": int(self.motion.num_clips),
            "reset_bins": int(self.reset_sampler.num_bins),
            "global_num_envs": self.num_envs * self.dp.world_size,
        }

    def save_checkpoint(self, directory: str) -> None:
        """Write ``directory`` atomically. Every rank must call this together.

        Rank 0 writes the shared state, each rank its RNG streams, then rank
        0 records sizes and sums and renames ``<dir>.tmp`` to ``<dir>``.
        """
        final = Path(directory)
        tmp = ckpt.temp_dir(final)
        if self.dp.is_main:
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True)
        self.dp.barrier()
        tmp.mkdir(parents=True, exist_ok=True)
        if self.dp.is_main:
            blob = {
                "models": self.models.state_dict(),
                "opt_wm": self.opt_wm.state_dict(),
                "opt_actor": self.opt_actor.state_dict(),
                "opt_critic": self.opt_critic.state_dict(),
                "sched_wm": self.sched_wm.state_dict(),
                "sched_actor": self.sched_actor.state_dict(),
                "sched_critic": self.sched_critic.state_dict(),
                "global_update": self.global_update,
                "sim_vector_steps": self.sim_vector_steps,
                "env_transitions": self.env_transitions * self.dp.world_size,
                "wm_optimizer_steps": self.wm_optimizer_steps,
                "actor_optimizer_steps": self.actor_optimizer_steps,
                "critic_optimizer_steps": self.critic_optimizer_steps,
                "step_acc": self.step_acc,
                "nonfinite_skips": self.nonfinite_skips,
                "reset_sampler": self.reset_sampler.state_dict(),
            }
            if self.opt_velocity is not None:
                blob["opt_velocity"] = self.opt_velocity.state_dict()
                blob["opt_velocity_critic"] = self.opt_velocity_critic.state_dict()
                blob["sched_velocity"] = self.sched_velocity.state_dict()
                blob["sched_velocity_critic"] = self.sched_velocity_critic.state_dict()
            torch.save(blob, tmp / ckpt.MODEL_FILE)
        torch.save(
            {"rank": self.dp.rank, "rng": ckpt.capture_rng(self.device)},
            tmp / ckpt.rank_file(self.dp.rank),
        )
        self.dp.barrier()
        if self.dp.is_main:
            ckpt.write_meta(tmp, self._checkpoint_meta())
            ckpt.publish(tmp, final)
        self.dp.barrier()

    def load_checkpoint(self, directory: str) -> None:
        """Restore weights, optimizers, schedulers, counters, reset bins, RNG.

        Replay, simulator state, observation history and velocity commands
        are not saved, so the next iteration warms a new replay first. This
        is a continuation, not an exact restore of the interrupted run.
        """
        path = Path(directory)
        meta = ckpt.verify_checkpoint(path)
        expected = self._checkpoint_meta()
        for key in ckpt.POLICY_KEYS:
            found = ckpt.policy_value(meta, key)
            want = ckpt.policy_value(expected, key)
            if found != want:
                raise ValueError(
                    f"checkpoint {path} has {key}={found!r}, this config has {want!r}"
                )
        if int(meta["world_size"]) != self.dp.world_size:
            warnings.warn(
                f"checkpoint {path} was written by {meta['world_size']} ranks, "
                f"resuming on {self.dp.world_size}; weights and optimizers load, "
                "RNG streams restart from the seed",
                stacklevel=2,
            )
        blob = torch.load(
            path / ckpt.MODEL_FILE,
            map_location=self.device,
            weights_only=True,
        )
        self.models.load_state_dict(blob["models"])
        self.opt_wm.load_state_dict(blob["opt_wm"])
        self.opt_actor.load_state_dict(blob["opt_actor"])
        self.opt_critic.load_state_dict(blob["opt_critic"])
        self.sched_wm.load_state_dict(blob["sched_wm"])
        self.sched_actor.load_state_dict(blob["sched_actor"])
        self.sched_critic.load_state_dict(blob["sched_critic"])
        self.global_update = int(blob["global_update"])
        self.sim_vector_steps = int(blob["sim_vector_steps"])
        self.env_transitions = int(blob["env_transitions"]) // self.dp.world_size
        self.wm_optimizer_steps = int(blob["wm_optimizer_steps"])
        self.actor_optimizer_steps = int(blob["actor_optimizer_steps"])
        self.critic_optimizer_steps = int(blob["critic_optimizer_steps"])
        self.step_acc = float(blob.get("step_acc", 0.0))
        self.nonfinite_skips = int(blob.get("nonfinite_skips", 0))
        if self.opt_velocity is not None and blob.get("opt_velocity") is not None:
            self.opt_velocity.load_state_dict(blob["opt_velocity"])
            self.opt_velocity_critic.load_state_dict(blob["opt_velocity_critic"])
            self.sched_velocity.load_state_dict(blob["sched_velocity"])
            self.sched_velocity_critic.load_state_dict(blob["sched_velocity_critic"])
        self.replay_warm = False
        if "reset_sampler" in blob:
            self.reset_sampler.load_state_dict(blob["reset_sampler"])
        else:
            self.reset_sampler.reset_statistics()
        rank_path = path / ckpt.rank_file(self.dp.rank)
        if int(meta["world_size"]) == self.dp.world_size:
            rank_blob = torch.load(rank_path, map_location="cpu", weights_only=True)
            ckpt.restore_rng(rank_blob["rng"], self.device)
        elif self.seed is not None:
            torch.manual_seed(self.seed + self.dp.rank)
