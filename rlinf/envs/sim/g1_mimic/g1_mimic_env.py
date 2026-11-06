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

"""G1 motion- and velocity-tracking training vector for SuperMimic.

The SuperMimic env worker drives this env one control step at a time. Inside
a step the rollout worker returns the clean policy action and, when motion
episodes finish, the critic values the reset bins bootstrap from. A gym
``chunk_step`` cannot carry those round trips, so the env exposes the step in
three calls: :meth:`observe`, :meth:`step` and :meth:`finish_step`.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from omegaconf import DictConfig

from rlinf.algorithms.supermimic.reset_sampler import BinResetSampler
from rlinf.algorithms.supermimic.scan import CompileCache, run_compiled
from rlinf.envs.sim.g1_mimic.constants import (
    ACTION_DIM,
    DT,
    EPISODE_CONT,
    EPISODE_FAILURE,
)
from rlinf.envs.sim.g1_mimic.g1_fk import asset_joint_limits, set_kinematics_compile
from rlinf.envs.sim.g1_mimic.mjlab_env import MjlabG1Env, compose_end_kind
from rlinf.envs.sim.g1_mimic.motion import MotionLibrary, load_motion_library
from rlinf.envs.sim.g1_mimic.obs import RollingHistory, add_sensor_noise, analytic_obs
from rlinf.envs.sim.g1_mimic.reward import (
    analytic_reward,
    compose_goal,
    differentiable_fk,
    motion_critic_input,
)
from rlinf.envs.sim.g1_mimic.velocity import (
    RESAMPLE_STEPS,
    normalize_command,
    sample_velocity_command,
    velocity_fall,
    velocity_reward,
)
from rlinf.models.embodiment.supermimic.normalization import merged_batch_stats
from rlinf.scheduler import Worker
from rlinf.utils.utils import seed_everything


def env_device() -> torch.device:
    """The accelerator this worker process uses, or the CPU.

    Float32 matmuls in the kinematics and rewards use TF32 tensor cores, as in
    the policy workers; the switch is process-wide.
    """
    device_type = Worker.torch_device_type
    platform = Worker.torch_platform
    if device_type is None or platform is None or not platform.is_available():
        return torch.device("cpu")
    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    return torch.device(device_type, platform.current_device())


def build_motion_library(env_cfg: DictConfig, device: torch.device) -> MotionLibrary:
    """Load the LAFAN1 or BONES-SEED library ``env_cfg`` names."""
    motion_format = str(env_cfg.get("motion_format", "lafan1"))
    # null loads every clip in the directory (after filtering, for BONES).
    num_clips = env_cfg.get("num_clips", None)
    num_clips = None if num_clips is None else int(num_clips)
    if motion_format == "bones_seed":
        from rlinf.envs.sim.g1_mimic.bones import load_bones_library

        return load_bones_library(
            str(env_cfg.get("motion_path", "")), num_clips, device
        )
    if motion_format == "lafan1":
        # motion_on_cpu keeps the frames in host memory; queries gather rows
        # there and copy only those rows to the simulator device.
        store = "cpu" if bool(env_cfg.get("motion_on_cpu", False)) else device
        return load_motion_library(env_cfg.get("motion_path", None), num_clips, store)
    raise ValueError(f"unknown env motion_format {motion_format!r}")


def make_goal(
    motion: MotionLibrary,
    encoding: str,
    anchor: bool,
    motion_id: torch.Tensor,
    motion_time: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    """Policy goal of one encoding at ``(motion_id, motion_time)``."""
    if encoding == "goal730":
        return motion.goal730(motion_id, motion_time)
    goal_ref = motion.goal_ref(motion_id, motion_time)
    ref = motion.state_at(motion_id, motion_time)
    return compose_goal(goal_ref, state, ref, anchor)


def _collect_post(
    reward_anchor: str,
    include_ref_ori: bool,
    next_state: torch.Tensor,
    ref_state: torch.Tensor,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    xi: torch.Tensor,
    q0: torch.Tensor,
    joint_low: torch.Tensor,
    joint_high: torch.Tensor,
    ref_pos: torch.Tensor,
    ref_rot: torch.Tensor,
    ref_lin: torch.Tensor,
    ref_ang: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Motion reward and the next noised observation frame of every env.

    ``ref_state`` is the clip frame the env just stepped to, so its root
    quaternion is the next frame's reference.
    """
    reward = analytic_reward(
        next_state,
        ref_state,
        action,
        prev_action,
        joint_low,
        joint_high,
        differentiable_fk(next_state),
        (ref_pos, ref_rot, ref_lin, ref_ang),
        reward_anchor=reward_anchor,
    )["reward"]
    ref_quat = ref_state[..., 3:7] if include_ref_ori else None
    frame = add_sensor_noise(
        analytic_obs(next_state, action, xi, q0, ref_quat=ref_quat), True
    )
    return reward, frame


class G1MimicEnv:
    """One rank's training envs: motion tracking, and velocity tracking if enabled.

    With velocity tracking the first half of the rows track motion and the
    second half track velocity commands. Motion rows restart from the
    return-weighted reset bins; velocity rows restart from uniform bins.
    """

    def __init__(
        self,
        cfg: DictConfig,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info=None,
    ):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.seed = int(cfg.seed) + int(seed_offset)
        self.total_num_processes = int(total_num_processes)
        self.worker_info = worker_info
        self.device = env_device()
        seed_everything(self.seed)
        set_kinematics_compile(bool(cfg.torch_compile))

        self.obs_dim = int(cfg.obs_dim)
        self.include_ref_ori = self.obs_dim == 99
        self.goal_encoding = str(cfg.goal_encoding)
        self.goal_anchor = bool(cfg.goal_anchor)
        self.velocity = bool(cfg.velocity_tracking)
        self.critic_features = str(cfg.critic_features)
        self.scale_xi = str(cfg.input_norm) == "running"
        self.reward_anchor = str(cfg.reward_anchor)
        self.action_noise_std = float(cfg.action_noise_std)
        self.motion = build_motion_library(cfg, self.device)
        self.sim = MjlabG1Env(
            num_envs=self.num_envs,
            motion=self.motion,
            device=self.device,
            max_episode_seconds=float(cfg.max_episode_seconds),
            termination_body=str(cfg.termination_body),
            eval_mode=False,
            com_sampling=str(cfg.get("com_sampling", "per_env")),
        )
        self.q0 = self.motion.q0.to(self.device)
        joint_low, joint_high = asset_joint_limits()
        self.joint_low = joint_low.to(self.device)
        self.joint_high = joint_high.to(self.device)
        self._compile_cache = CompileCache(enabled=bool(cfg.torch_compile))
        self.group: dist.ProcessGroup | None = None

        self.num_motion_envs = self.num_envs // 2 if self.velocity else self.num_envs
        self.velocity_env = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.velocity_env[self.num_motion_envs :] = self.velocity
        self.prev_action = torch.zeros(self.num_envs, ACTION_DIM, device=self.device)
        self.command = torch.zeros(self.num_envs, 3, device=self.device)
        self.command_from_ref = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.command_age = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        max_steps = max(int(round(float(cfg.max_episode_seconds) / DT)), 1)
        self.reset_sampler = BinResetSampler(
            self.motion.lengths,
            self.num_envs,
            float(cfg.G_max),
            float(cfg.gamma),
            max_steps,
            self.device,
        )
        self.reset_sampler.deferred = self.total_num_processes > 1
        self._end_kind: torch.Tensor | None = None
        self._done_ids: torch.Tensor | None = None
        self._done_motion: torch.Tensor | None = None
        self._clear_metrics()

        # The simulator constructor draws uniform frames. Training starts use
        # the return-bin mixture instead, including this first reset.
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.sim.reset(env_ids, *self._reset_points(env_ids))
        self._update_commands(env_ids, force=True)
        self.history = RollingHistory(self._new_frames(env_ids))

    def set_process_group(self, group: dist.ProcessGroup | None) -> None:
        """Group of every training env rank; the reset bins are pooled over it."""
        self.group = group

    def _goal(self, motion_id, motion_time, state) -> torch.Tensor:
        return make_goal(
            self.motion,
            self.goal_encoding,
            self.goal_anchor,
            motion_id,
            motion_time,
            state,
        )

    def _ref_quat(self, motion_id, motion_time) -> torch.Tensor | None:
        if not self.include_ref_ori:
            return None
        return self.motion.state_at(motion_id, motion_time)[..., 3:7]

    def _new_frames(self, env_ids: torch.Tensor) -> torch.Tensor:
        """One noised frame per listed env, from the env's current state."""
        sim = self.sim
        ref_quat = self._ref_quat(sim.motion_id[env_ids], sim.motion_time[env_ids])
        frame = analytic_obs(
            sim.state[env_ids],
            self.prev_action[env_ids],
            sim.xi[env_ids],
            self.q0,
            ref_quat=ref_quat,
        )
        return add_sensor_noise(frame, True).detach()

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
                self.sim.motion_id[rows], self.sim.motion_time[rows]
            )
            drawn = torch.where(from_ref.unsqueeze(-1), ref, drawn)
            self.command[rows] = drawn
            self.command_from_ref[rows] = from_ref
            self.command_age[rows] = 0
        follow = env_ids[self.command_from_ref[env_ids]]
        if follow.numel():
            self.command[follow] = self.motion.reference_velocity_command(
                self.sim.motion_id[follow], self.sim.motion_time[follow]
            )

    def observe(self) -> dict[str, torch.Tensor]:
        """Policy inputs of the coming step.

        Returns ``history`` ``[N, 10 * obs_dim]``, ``goal`` ``[N, goal_dim]``
        and, with velocity tracking, the normalized ``command`` ``[N, 3]`` and
        the velocity-row flag ``task`` ``[N]``. Motion rows read ``goal``;
        velocity rows read ``command``.
        """
        sim = self.sim
        goal = run_compiled(
            "collect-goal",
            self._goal,
            sim.motion_id,
            sim.motion_time,
            sim.state,
            cache=self._compile_cache,
        )
        self._update_commands(
            torch.arange(self.num_envs, device=self.device), force=False
        )
        obs = {"history": self.history.flat().contiguous(), "goal": goal}
        if self.velocity:
            obs["command"] = normalize_command(self.command)
            obs["task"] = self.velocity_env.clone()
        return obs

    def policy_input_stats(
        self, obs: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """Statistics of the frame and goal the policy is about to see.

        Merged over every env rank, so this is a collective. The rollout and
        actor workers fold the same statistics into their input norms. Empty
        without running input norms. Only motion rows count toward the goal.
        """
        if not self.scale_xi:
            return {}, {}
        obs_stats, obs_count = merged_batch_stats(
            self.history.last(), self.group, self.obs_dim
        )
        goal = obs["goal"][: self.num_motion_envs]
        goal_stats, goal_count = merged_batch_stats(goal, self.group, goal.shape[-1])
        return (
            {"obs_stats": obs_stats, "goal_stats": goal_stats},
            {"obs_count": obs_count, "goal_count": goal_count},
        )

    def step(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Execute ``action`` with Gaussian noise and return the replay transition.

        The clean ``action`` scores the action terms of the reward and becomes
        the next frame's previous action; the world model learns from the
        noisy one the simulator executed. Every tensor in the result is owned
        by the caller: the env does not write to it.
        """
        executed = action + self.action_noise_std * torch.randn_like(action)
        sim = self.sim
        before = sim.state.clone()
        time_in = sim.time_in_episode.clone()
        motion_id = sim.motion_id.clone()
        motion_time = sim.motion_time.clone()
        xi = sim.xi.clone()
        obs_frame = self.history.last().contiguous()
        prev_action = self.prev_action.clone()
        result = sim.step(executed)
        next_state = result["next_state"]
        failure = result["motion_failure"]
        if self.velocity:
            failure = torch.where(self.velocity_env, velocity_fall(next_state), failure)
        end_kind = compose_end_kind(failure, result["timeout"], result["clip_end"])
        with torch.no_grad():
            reward, frames = run_compiled(
                "collect-post",
                _collect_post,
                self.reward_anchor,
                self.include_ref_ori,
                next_state,
                result["ref_state"],
                action,
                prev_action,
                xi,
                self.q0,
                self.joint_low,
                self.joint_high,
                *self.motion.kinematics_at(motion_id, motion_time + 1),
                cache=self._compile_cache,
            )
        if self.velocity:
            rows = slice(self.num_motion_envs, None)
            reward = reward.clone()
            reward[rows] = velocity_reward(
                next_state[rows],
                self.command[rows],
                action[rows],
                prev_action[rows],
                self.joint_low,
                self.joint_high,
            )
        self.reset_sampler.log_step(
            reward,
            motion_id,
            motion_time,
            rows=~self.velocity_env if self.velocity else None,
        )
        transition = {
            "state": before,
            "next_state": next_state.clone(),
            "action": action.clone(),
            "executed": executed,
            "reward": reward.contiguous(),
            "end_kind": end_kind,
            "xi": xi,
            "motion_id": motion_id,
            "motion_time": motion_time,
            "time_in_episode": time_in,
            "prev_action": prev_action,
            "obs": obs_frame.clone(),
            "task": self.velocity_env.long(),
            "command": self.command.clone(),
            "command_from_ref": self.command_from_ref.clone(),
        }
        self.prev_action = action.detach().clone()
        self.history.push(frames.detach())
        self._end_kind = end_kind
        self._reward_total = self._reward_total + reward.detach().sum()
        self._failures = self._failures + (end_kind == EPISODE_FAILURE).sum()
        self._steps += 1
        return transition

    def bootstrap_features(self) -> torch.Tensor | None:
        """Motion-critic inputs of the motion episodes that just ended.

        ``None`` when no motion episode ended this step. Timeout tails
        bootstrap from these values; the reset sampler drops the others.
        """
        end_kind = self._end_kind
        done_ids = (end_kind != EPISODE_CONT).nonzero(as_tuple=False).squeeze(-1)
        self._done_ids = done_ids
        done_motion = done_ids
        if self.velocity:
            done_motion = done_ids[done_ids < self.num_motion_envs]
        self._done_motion = done_motion
        if done_motion.numel() == 0:
            return None
        # Every env goes through one compiled call; a few finished rows would
        # stay eager and launch many small kernels per step.
        sim = self.sim
        with torch.no_grad():
            features = run_compiled(
                "collect-bootstrap",
                self._critic_features,
                sim.state,
                sim.motion_id,
                sim.motion_time,
                sim.xi,
                cache=self._compile_cache,
            )
        return features[done_motion].contiguous()

    def _critic_features(self, state, motion_id, motion_time, xi) -> torch.Tensor:
        goal = self._goal(motion_id, motion_time, state)
        ref = self.motion.state_at(motion_id, motion_time)
        return motion_critic_input(
            state, goal, xi, ref, self.critic_features, scale_xi=self.scale_xi
        )

    def finish_step(self, values: torch.Tensor | None) -> None:
        """Fold finished motion episodes into the reset bins and restart every done env.

        ``values`` are the motion-critic values of :meth:`bootstrap_features`,
        in the same row order. Every env rank calls this once per step.
        """
        done_ids = self._done_ids
        if done_ids.numel():
            self._episodes += int(done_ids.numel())
            done_motion = self._done_motion
            if done_motion.numel():
                self.reset_sampler.finish(
                    done_motion, self._end_kind[done_motion], values
                )
            self.sim.reset(done_ids, *self._reset_points(done_ids))
            self._update_commands(done_ids, force=True)
            self.prev_action[done_ids] = 0
            self.history.fill(done_ids, self._new_frames(done_ids))
        self.reset_sampler.sync(self.group)

    def _clear_metrics(self) -> None:
        self._reward_total = torch.zeros((), device=self.device)
        self._failures = torch.zeros((), dtype=torch.long, device=self.device)
        self._episodes = 0
        self._steps = 0

    def consume_metrics(self) -> dict[str, float]:
        """Means over every env rank since the last call, then a reset."""
        totals = torch.stack(
            (
                self._reward_total.float(),
                self._failures.float(),
                torch.tensor(float(self._episodes), device=self.device),
            )
        )
        world = 1
        if self.group is not None and dist.get_world_size(self.group) > 1:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM, group=self.group)
            world = dist.get_world_size(self.group)
        bins = self.reset_sampler.bin_return
        reward, failures, episodes, bin_mean, bin_min = torch.cat(
            (totals, bins.mean().reshape(1), bins.min().reshape(1))
        ).tolist()
        transitions = max(self._steps * self.num_envs * world, 1)
        self._clear_metrics()
        return {
            "sim/terminal_rate": failures / transitions,
            "sim/reward_mean": reward / transitions,
            "sim/episodes_finished": episodes,
            "reset/bin_return_mean": bin_mean,
            "reset/bin_return_min": bin_min,
        }

    def state_dict(self) -> dict:
        """Reset-bin statistics. Simulator state and histories are not saved."""
        return {"reset_sampler": self.reset_sampler.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.reset_sampler.load_state_dict(state["reset_sampler"])
