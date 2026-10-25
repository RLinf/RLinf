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
import time
from pathlib import Path

import torch
from omegaconf import DictConfig

from rlinf.envs.sim.g1_mimic.objectives import (
    actor_objective,
    gae_targets,
    state_prediction_loss,
    termination_bce,
)
from rlinf.envs.sim.g1_mimic.constants import (
    DT,
    EPISODE_CONT,
    EPISODE_FAILURE,
    EPISODE_TIMEOUT,
    HISTORY_LEN,
)
from rlinf.envs.sim.g1_mimic.env import make_env
from rlinf.envs.sim.g1_mimic.eval_rollout import (
    run_tracking_eval,
    summarize_tracking_eval,
)
from rlinf.envs.sim.g1_mimic.reset_sampler import BinResetSampler
from rlinf.envs.sim.g1_mimic.features import compose_goal, integrate_residual, root_error_4d
from rlinf.envs.sim.g1_mimic.g1_fk import asset_joint_limits
from rlinf.envs.sim.g1_mimic.bones import load_bones_library
from rlinf.envs.sim.g1_mimic.motion import load_motion_library
from rlinf.envs.sim.g1_mimic.obs import add_sensor_noise, analytic_obs, pad_history
from rlinf.envs.sim.g1_mimic.replay import SuperMimicReplay
from rlinf.envs.sim.g1_mimic.velocity import (
    ALIGN_WEIGHT,
    RESAMPLE_STEPS,
    normalize_command,
    sample_velocity_command,
    velocity_fall,
    velocity_fall_soft,
    velocity_reward,
)
from rlinf.envs.sim.g1_mimic.reward import (
    analytic_reward,
    failure_signals,
    termination_head_input,
)
from rlinf.envs.sim.g1_mimic.networks import SuperMimicModels


def _host_rss_bytes() -> float:
    try:
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) * 1024.0
    except (OSError, ValueError, IndexError):
        return 0.0
    return 0.0


def _set_requires_grad(params, flag: bool) -> None:
    for param in params:
        param.requires_grad_(flag)


def _clip_actor_grads(params) -> float:
    norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
    for param in params:
        if param.grad is not None:
            param.grad.clamp_(-0.1, 0.1)
    return float(norm)


def _finite(x: torch.Tensor | float) -> bool:
    value = float(x) if not torch.is_tensor(x) else float(x.detach())
    return math.isfinite(value)


class SuperMimicTrainer:
    def __init__(self, cfg: DictConfig, device: torch.device | None = None):
        self.cfg = cfg
        algo = cfg.algorithm
        env_cfg = cfg.env
        if device is None:
            requested = str(env_cfg.get("device", "cpu"))
            if requested.startswith("cuda") and torch.cuda.is_available():
                device = torch.device(requested)
            else:
                device = torch.device("cpu")
        self.device = device
        self.goal_anchor = bool(algo.goal_anchor)
        self.k = int(algo.k)
        self.H = int(algo.H)
        self.gamma = float(algo.gamma)
        self.lam = float(algo.lam)
        self.goal_dim = int(algo.goal_dim)
        self.num_envs = int(env_cfg.num_envs)
        self.warmup = int(algo.warmup_sim_steps)
        self.updates_per_env_step = float(algo.updates_per_env_step)
        self.wm_batches = int(algo.wm_batches)
        self.wm_window_batch = int(algo.wm_window_batch)
        self.imagine_batch = int(algo.imagine_batch)
        self.bce_weight = float(algo.get("bce_weight", 1.0))
        self.action_noise_std = float(algo.action_noise_std)
        self.G_max = float(algo.G_max)
        motion_format = str(env_cfg.get("motion_format", "lafan1"))
        if motion_format == "bones_seed":
            self.motion = load_bones_library(
                str(env_cfg.get("motion_path", "")),
                int(env_cfg.get("num_clips", 40)),
                self.device,
            )
        elif motion_format == "lafan1":
            self.motion = load_motion_library(
                env_cfg.get("motion_path", None),
                int(env_cfg.get("num_clips", 40)),
                self.device,
                allow_synthetic=str(env_cfg.get("backend", "analytic")) == "analytic",
            )
        else:
            raise ValueError(f"unknown env.motion_format {motion_format!r}")
        self.env = make_env(
            str(env_cfg.backend),
            num_envs=self.num_envs,
            motion=self.motion,
            device=self.device,
            max_episode_seconds=float(algo.max_episode_seconds),
            termination_body=str(algo.termination_body),
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
        self.models = SuperMimicModels(
            self.goal_dim, self.obs_dim, velocity=self.velocity
        ).to(self.device)
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
        self.sched_critic = torch.optim.lr_scheduler.LambdaLR(self.opt_critic, lr_lambda)
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
        self.velocity_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.velocity:
            self.velocity_env[half:] = True
        self.command = torch.zeros(self.num_envs, 3, device=self.device)
        self.command_from_ref = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.command_age = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.obs_hist: list[list[torch.Tensor]] = [[] for _ in range(self.num_envs)]
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
        # The env constructor draws a uniform frame. Training starts use the
        # return-bin mixture instead, including this first reset.
        env_ids = torch.arange(self.num_envs, device=self.device)
        motion_id, motion_time = self.reset_sampler.sample(self.num_envs)
        self.env.reset(env_ids, motion_id, motion_time)
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
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _history_tensor(self) -> torch.Tensor:
        return torch.stack(
            [pad_history(frames) for frames in self.obs_hist], dim=0
        )

    def _noisy_obs(self, state, prev, xi, ref_quat, noise: bool) -> torch.Tensor:
        """One new frame. ``noise`` draws uniform sensor noise once."""
        return add_sensor_noise(
            analytic_obs(state, prev, xi, self.q0, ref_quat=ref_quat), noise
        )

    def _refresh_history(self) -> None:
        # Reset fill repeats one noised frame. That single draw is the noise
        # sample for every padded slot; later steps do not redraw it.
        ref = self._ref_quat(self.env.motion_id, self.env.motion_time)
        for i in range(self.num_envs):
            ref_i = None if ref is None else ref[i]
            frame = self._noisy_obs(
                self.env.state[i], self.prev_action[i], self.env.xi[i], ref_i, True
            ).detach()
            self.obs_hist[i] = [frame] * HISTORY_LEN

    def _append_obs(self, env_ids: torch.Tensor | None = None) -> None:
        ids = (
            range(self.num_envs)
            if env_ids is None
            else env_ids.tolist()
        )
        if not ids:
            return
        id_tensor = torch.tensor(ids, dtype=torch.long, device=self.device)
        ref = self._ref_quat(self.env.motion_id[id_tensor], self.env.motion_time[id_tensor])
        for n, i in enumerate(ids):
            ref_i = None if ref is None else ref[n]
            frame = self._noisy_obs(
                self.env.state[i], self.prev_action[i], self.env.xi[i], ref_i, True
            ).detach()
            frames = self.obs_hist[i]
            frames.append(frame)
            self.obs_hist[i] = frames[-HISTORY_LEN:]

    def collect_steps(self, n_steps: int, accumulate: bool = True) -> dict[str, float]:
        failures = 0
        sim_s = 0.0
        for _ in range(n_steps):
            history = self._history_tensor()
            current_obs = history.reshape(self.num_envs, HISTORY_LEN, self.obs_dim)[:, -1]
            goal = self._goal(self.env.motion_id, self.env.motion_time, self.env.state)
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
            end_kind = result["end_kind"]
            reward = analytic_reward(
                next_state,
                result["ref_state"],
                action,
                self.prev_action,
                self.joint_low,
                self.joint_high,
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
                fall = velocity_fall(next_state)
                undo = self.velocity_env & (end_kind == EPISODE_FAILURE) & ~fall
                end_kind = torch.where(undo, torch.full_like(end_kind, EPISODE_CONT), end_kind)
                end_kind = torch.where(
                    self.velocity_env & fall,
                    torch.full_like(end_kind, EPISODE_FAILURE),
                    end_kind,
                )
            self.reset_sampler.log_step(reward, mid, mt)
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
                failures += int((end_kind == EPISODE_FAILURE).sum())
                self._update_reset_bins(done_ids, end_kind[done_ids])
                self.env.reset(done_ids, *self.reset_sampler.sample(int(done_ids.numel())))
                self._update_commands(done_ids, force=True)
                self.prev_action[done_ids] = 0
                for i in done_ids.tolist():
                    self.obs_hist[i] = []
                self._append_obs(done_ids)
            self.sim_vector_steps += 1
            self.env_transitions += self.num_envs
            if accumulate:
                self.step_acc += self.updates_per_env_step
        valid = self.buffer.num_valid_windows(self.k)
        self.time_sim = sim_s
        return {
            "sim/terminal_rate": failures / max(n_steps * self.num_envs, 1),
            "buffer/valid_windows": float(valid),
            "time/sim_s": float(sim_s),
        }

    def _update_reset_bins(self, env_ids, end_kind) -> None:
        bootstrap = None
        if bool((end_kind == EPISODE_TIMEOUT).any()):
            with torch.no_grad():
                bootstrap = self._bootstrap_value(env_ids)
        self.reset_sampler.finish(env_ids, end_kind, bootstrap)

    def _bootstrap_value(self, env_ids: torch.Tensor) -> torch.Tensor:
        state = self.env.state[env_ids]
        goal = self._goal(
            self.env.motion_id[env_ids], self.env.motion_time[env_ids], state
        )
        ref = self.motion.state_at(
            self.env.motion_id[env_ids], self.env.motion_time[env_ids]
        )
        return self.models.critic(
            state, goal, self.env.xi[env_ids], root_error_4d(state, ref)
        )

    def fit_world_model(self) -> dict[str, float]:
        sample_s = 0.0
        update_s = 0.0
        t_sample = time.perf_counter()
        batch = self.buffer.sample_windows(self.k, self.wm_window_batch)
        sample_s += time.perf_counter() - t_sample
        if batch is None:
            self.time_replay_sample = sample_s
            self.time_world_model = 0.0
            return {"wm/state_loss": float("nan"), "wm/grad_norm": 0.0}
        _set_requires_grad(self.models.wm_params(), True)
        last_loss = 0.0
        last_norm = 0.0
        for _ in range(self.wm_batches):
            t_sample = time.perf_counter()
            batch = self.buffer.sample_windows(self.k, self.wm_window_batch)
            sample_s += time.perf_counter() - t_sample
            if batch is None:
                break
            self._sync()
            t_update = time.perf_counter()
            pred = batch["state"][:, 0]
            loss = pred.new_zeros(())
            for t in range(self.k):
                residual = self.models.dynamics(
                    pred, batch["action_executed"][:, t], batch["xi"]
                )
                pred = integrate_residual(pred, residual)
                loss = loss + state_prediction_loss(pred, batch["state"][:, t + 1])
                motion_rows = batch["task_id"] == 0
                if bool(motion_rows.any()):
                    height, tilt, _ = failure_signals(
                        pred[motion_rows],
                        self.motion.state_at(
                            batch["motion_id"][motion_rows],
                            batch["motion_time"][motion_rows] + t + 1,
                        ),
                    )
                    logit = self.models.termination(termination_head_input(height, tilt))
                    loss = loss + self.bce_weight * termination_bce(
                        logit, batch["end_kind"][motion_rows, t]
                    )
            self.opt_wm.zero_grad(set_to_none=True)
            loss.backward()
            last_norm = float(
                torch.nn.utils.clip_grad_norm_(self.models.wm_params(), 1.0)
            )
            self.opt_wm.step()
            self.wm_optimizer_steps += 1
            self._sync()
            last_loss = float(loss.detach())
            update_s += time.perf_counter() - t_update
        self.time_replay_sample = sample_s
        self.time_world_model = update_s
        return {"wm/state_loss": last_loss, "wm/grad_norm": last_norm}

    def imagine_and_update_actor(self) -> dict[str, float]:
        t_sample = time.perf_counter()
        motion_batch = self.imagine_batch // 2 if self.velocity else self.imagine_batch
        starts = self.buffer.sample_starts(
            motion_batch, self.q0, task=0 if self.velocity else None
        )
        self.time_replay_sample += time.perf_counter() - t_sample
        if starts is None and not self.velocity:
            self.time_bptt = 0.0
            return {
                "actor/objective": float("nan"),
                "actor/grad_norm": 0.0,
                "imagination/survival": 0.0,
            }
        if starts is None:
            metrics = {
                "actor/objective": float("nan"),
                "actor/grad_norm": 0.0,
                "imagination/survival": 0.0,
            }
            metrics.update(self._velocity_imagine())
            return metrics
        self._sync()
        t_bptt = time.perf_counter()
        _set_requires_grad(self.models.wm_params(), False)
        _set_requires_grad(self.models.critic_params(), False)
        _set_requires_grad(self.models.actor_params(), True)
        state = starts["state"]
        prev = starts["prev_action"]
        xi = starts["xi"]
        history = starts["history"]
        motion_id = starts["motion_id"]
        time0 = starts["motion_time"]
        # One rollout keeps s_0..s_H, g_0..g_H, r_1..r_H and d_1..d_H.
        # history already ends at s_0, so the policy sees it before the step.
        states, goals, root_errs, rewards, d_hats = [], [], [], [], []
        for t in range(self.H):
            ref = self.motion.state_at(motion_id, time0 + t)
            goal = self._goal(motion_id, time0 + t, state)
            states.append(state)
            goals.append(goal)
            root_errs.append(root_error_4d(state, ref))
            # Stored frames keep the noise they were given at collection.
            # Only the frame produced by this imagined step draws new noise.
            action = self.models.policy(history, goal)
            state = integrate_residual(state, self.models.dynamics(state, action, xi))
            ref_next = self.motion.state_at(motion_id, time0 + t + 1)
            reward_pack = analytic_reward(
                state, ref_next, action, prev, self.joint_low, self.joint_high
            )
            height, tilt, _ = failure_signals(state, ref_next)
            d_hat = torch.sigmoid(
                self.models.termination(termination_head_input(height, tilt))
            )
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
        ref_h = self.motion.state_at(motion_id, time0 + self.H)
        goal_h = self._goal(motion_id, time0 + self.H, state)
        states.append(state)
        goals.append(goal_h)
        root_errs.append(root_error_4d(state, ref_h))
        states_t = torch.stack(states, dim=1)
        goals_t = torch.stack(goals, dim=1)
        err_t = torch.stack(root_errs, dim=1)
        rewards_t = torch.stack(rewards, dim=1)
        d_t = torch.stack(d_hats, dim=1)
        v_last = self.models.critic(states_t[:, -1], goals_t[:, -1], xi, err_t[:, -1])
        objective = actor_objective(rewards_t, d_t, v_last, self.gamma)
        self.opt_actor.zero_grad(set_to_none=True)
        (-objective).backward()
        grad_norm = _clip_actor_grads(self.models.actor_params())
        self.opt_actor.step()
        self._sync()
        self.actor_optimizer_steps += 1
        self.time_bptt = time.perf_counter() - t_bptt
        self._sync()
        t_critic = time.perf_counter()
        critic_metrics = self.fit_critic(
            states_t.detach(),
            goals_t.detach(),
            err_t.detach(),
            xi.detach(),
            rewards_t.detach(),
            d_t.detach(),
        )
        self.time_critic = time.perf_counter() - t_critic
        survival = float((1.0 - d_t.detach()).prod(dim=1).mean())
        metrics = {
            "actor/objective": float(objective.detach()),
            "actor/grad_norm": grad_norm,
            "imagination/survival": survival,
        }
        metrics.update(critic_metrics)
        if self.velocity:
            metrics.update(self._velocity_imagine())
        return metrics

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
        if starts is None:
            return {
                "velocity/objective": float("nan"),
                "velocity/grad_norm": 0.0,
                "velocity/align": 0.0,
            }
        _set_requires_grad(self.models.wm_params(), False)
        _set_requires_grad(self.models.actor_params(), False)
        _set_requires_grad(self.models.velocity_actor_params(), True)
        _set_requires_grad(self.models.velocity_critic_params(), False)
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
            state = integrate_residual(state, self.models.dynamics(state, action, xi))
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
            torch.stack(align_terms).mean()
            if align_terms
            else objective.new_zeros(())
        )
        self.opt_velocity.zero_grad(set_to_none=True)
        (-objective + ALIGN_WEIGHT * align).backward()
        grad_norm = _clip_actor_grads(self.models.velocity_actor_params())
        self.opt_velocity.step()
        self._fit_velocity_critic(
            states_t.detach(), commands_t.detach(), xi.detach(), rewards_t.detach(), d_t.detach()
        )
        _set_requires_grad(self.models.actor_params(), True)
        return {
            "velocity/objective": float(objective.detach()),
            "velocity/grad_norm": grad_norm,
            "velocity/align": float(align.detach()),
        }

    def _fit_velocity_critic(self, states_t, commands_t, xi, rewards_t, d_t) -> None:
        _set_requires_grad(self.models.velocity_critic_params(), True)
        with torch.no_grad():
            flat_xi = xi.unsqueeze(1).expand(-1, self.H + 1, -1).reshape(-1, xi.shape[-1])
            values = self.models.velocity_critic(
                states_t.reshape(-1, states_t.shape[-1]),
                commands_t.reshape(-1, commands_t.shape[-1]),
                flat_xi,
            ).reshape(states_t.shape[0], self.H + 1)
            targets = gae_targets(rewards_t, d_t, values, self.gamma, self.lam)
        batch = states_t.shape[0]
        mb = max(1, batch // int(self.cfg.algorithm.critic_minibatches))
        for _ in range(int(self.cfg.algorithm.critic_epochs)):
            perm = torch.randperm(batch, device=self.device)
            for start in range(0, batch, mb):
                idx = perm[start : start + mb]
                pred = self.models.velocity_critic(
                    states_t[idx, : self.H].reshape(-1, states_t.shape[-1]),
                    commands_t[idx, : self.H].reshape(-1, commands_t.shape[-1]),
                    xi[idx].unsqueeze(1).expand(-1, self.H, -1).reshape(-1, xi.shape[-1]),
                ).reshape(idx.numel(), self.H)
                loss = (pred - targets[idx]).pow(2).mean()
                self.opt_velocity_critic.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.models.velocity_critic_params(), 1.0)
                self.opt_velocity_critic.step()
                self.critic_optimizer_steps += 1
        _set_requires_grad(self.models.velocity_critic_params(), False)

    def fit_critic(
        self, states_t, goals_t, err_t, xi, rewards_t, d_t
    ) -> dict[str, float]:
        """Regress V(s_t, g_t) onto R_t from the same imagined trajectory.

        ``states_t`` is ``[B, H+1]``. Targets are ``R_0 .. R_{H-1}``.
        """
        _set_requires_grad(self.models.critic_params(), True)
        _set_requires_grad(self.models.actor_params(), False)
        _set_requires_grad(self.models.wm_params(), False)
        if states_t.shape[1] != self.H + 1 or rewards_t.shape[1] != self.H:
            raise RuntimeError(
                "critic targets must align with s_0..s_{H-1} from one rollout"
            )
        with torch.no_grad():
            flat_xi = xi.unsqueeze(1).expand(-1, self.H + 1, -1).reshape(-1, xi.shape[-1])
            values = self.models.critic(
                states_t.reshape(-1, states_t.shape[-1]),
                goals_t.reshape(-1, goals_t.shape[-1]),
                flat_xi,
                err_t.reshape(-1, 4),
            ).reshape(states_t.shape[0], self.H + 1)
            targets = gae_targets(rewards_t, d_t, values, self.gamma, self.lam)
        state_fit = states_t[:, : self.H]
        goal_fit = goals_t[:, : self.H]
        err_fit = err_t[:, : self.H]
        last_loss = 0.0
        batch = state_fit.shape[0]
        mb = max(1, batch // int(self.cfg.algorithm.critic_minibatches))
        for _ in range(int(self.cfg.algorithm.critic_epochs)):
            perm = torch.randperm(batch, device=self.device)
            for start in range(0, batch, mb):
                idx = perm[start : start + mb]
                pred = self.models.critic(
                    state_fit[idx].reshape(-1, state_fit.shape[-1]),
                    goal_fit[idx].reshape(-1, goal_fit.shape[-1]),
                    xi[idx].unsqueeze(1).expand(-1, self.H, -1).reshape(-1, xi.shape[-1]),
                    err_fit[idx].reshape(-1, 4),
                ).reshape(idx.numel(), self.H)
                loss = (pred - targets[idx]).pow(2).mean()
                self.opt_critic.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.models.critic_params(), 1.0)
                self.opt_critic.step()
                self.critic_optimizer_steps += 1
                last_loss = float(loss.detach())
        self._sync()
        _set_requires_grad(self.models.actor_params(), True)
        return {"critic/loss": last_loss}

    def maybe_update(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        while self.step_acc >= 1.0:
            if self.buffer.num_valid_windows(self.k) == 0:
                break
            metrics.update(self.fit_world_model())
            metrics.update(self.imagine_and_update_actor())
            self.sched_wm.step()
            self.sched_actor.step()
            self.sched_critic.step()
            if self.sched_velocity is not None:
                self.sched_velocity.step()
                self.sched_velocity_critic.step()
            self.global_update += 1
            self.step_acc -= 1.0
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
        if self.device.type == "cuda":
            metrics["memory/peak_allocated"] = float(
                torch.cuda.max_memory_allocated(self.device)
            )
        else:
            metrics["memory/peak_allocated"] = 0.0
        metrics["memory/host_rss"] = _host_rss_bytes()
        return metrics

    def run_iteration(self) -> dict[str, float]:
        # Resume restores weights only. The new replay still needs the warmup.
        if not self.replay_warm:
            if self.warmup > 0:
                self.collect_steps(self.warmup, accumulate=False)
            self.replay_warm = True
        metrics = self.collect_steps(1)
        metrics.update(self.maybe_update())
        return metrics

    def run_until_updates(self, n_updates: int) -> dict[str, float]:
        metrics: dict[str, float] = {}
        while self.global_update < n_updates:
            metrics = self.run_iteration()
        return metrics

    def evaluate(self) -> dict[str, float]:
        """Every clip once, from frame 0, in batches of ``num_envs``."""
        n_clips = int(self.motion.num_clips)
        batch = min(self.num_envs, n_clips)
        eval_env = make_env(
            str(self.cfg.env.backend),
            num_envs=batch,
            motion=self.motion,
            device=self.device,
            max_episode_seconds=1.0e6,
            termination_body=str(self.cfg.algorithm.termination_body),
            eval_mode=True,
        )

        def policy(history, goal):
            with torch.no_grad():
                return self.models.policy(history, goal)

        parts: list[dict[str, torch.Tensor]] = []
        for start in range(0, n_clips, batch):
            ids = torch.arange(
                start, min(start + batch, n_clips), device=self.device
            )
            count = int(ids.shape[0])
            eval_env.reset(
                torch.arange(count, device=self.device),
                ids,
                torch.zeros(count, dtype=torch.long, device=self.device),
            )
            parts.append(
                run_tracking_eval(
                    eval_env,
                    policy,
                    self.q0,
                    self._goal,
                    joint_low=self.joint_low,
                    joint_high=self.joint_high,
                    active=count,
                    include_ref_ori=self.include_ref_ori,
                )
            )
        merged = {
            key: torch.cat([part[key] for part in parts], dim=0) for key in parts[0]
        }
        if int(merged["clip_id"].unique().numel()) != n_clips:
            raise RuntimeError("evaluation did not cover every clip once")
        return summarize_tracking_eval(merged)

    def save_checkpoint(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
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
            "env_transitions": self.env_transitions,
            "wm_optimizer_steps": self.wm_optimizer_steps,
            "actor_optimizer_steps": self.actor_optimizer_steps,
            "critic_optimizer_steps": self.critic_optimizer_steps,
        }
        if self.opt_velocity is not None:
            blob["opt_velocity"] = self.opt_velocity.state_dict()
            blob["opt_velocity_critic"] = self.opt_velocity_critic.state_dict()
            blob["sched_velocity"] = self.sched_velocity.state_dict()
            blob["sched_velocity_critic"] = self.sched_velocity_critic.state_dict()
        torch.save(blob, path / "supermimic.pt")

    def load_checkpoint(self, directory: str) -> None:
        blob = torch.load(
            Path(directory) / "supermimic.pt",
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
        self.env_transitions = int(blob["env_transitions"])
        self.wm_optimizer_steps = int(blob["wm_optimizer_steps"])
        self.actor_optimizer_steps = int(blob["actor_optimizer_steps"])
        self.critic_optimizer_steps = int(blob["critic_optimizer_steps"])
        if self.opt_velocity is not None and blob.get("opt_velocity") is not None:
            self.opt_velocity.load_state_dict(blob["opt_velocity"])
            self.opt_velocity_critic.load_state_dict(blob["opt_velocity_critic"])
            self.sched_velocity.load_state_dict(blob["sched_velocity"])
            self.sched_velocity_critic.load_state_dict(blob["sched_velocity_critic"])
        self.replay_warm = False
        self.reset_sampler.reset_statistics()
