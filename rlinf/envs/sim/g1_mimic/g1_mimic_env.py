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

"""G1 motion-tracking training vector for MuJoCo Warp.

Exposes both the standard RLinf embodied env contract (:meth:`reset` and
:meth:`chunk_step`) and the fine-grained transition interface (:meth:`observe`
and :meth:`step`) used by :class:`SuperMimicEnvWorker`. Reset sampling defaults
to uniform clip frames and may be replaced via :meth:`set_reset_sampler`.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig

from rlinf.envs.sim.g1_mimic.constants import ACTION_DIM
from rlinf.envs.sim.g1_mimic.kinematics.features import quat_to_rotmat
from rlinf.envs.sim.g1_mimic.kinematics.g1_fk import set_kinematics_compile
from rlinf.envs.sim.g1_mimic.kinematics.representations import make_representation
from rlinf.envs.sim.g1_mimic.mjlab_env import MjlabG1Env, env_device
from rlinf.envs.sim.g1_mimic.tasks.motion import build_motion_library, root_goals
from rlinf.envs.sim.g1_mimic.tasks.reward import tracking_reward
from rlinf.utils.compile_cache import CompileCache, run_compiled
from rlinf.utils.utils import seed_everything


class G1MimicEnv:
    """One rank's G1 motion-tracking training envs.

    Resets draw a library frame (uniformly by default, or from an injected
    reset sampler), write that frame with mjlab's reset noise, and point the
    reference clock at the next frame, the target of the first step.
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
        self.device = env_device()
        seed_everything(self.seed)
        set_kinematics_compile(bool(cfg.torch_compile))

        self.representation = make_representation(str(cfg.representation)).to(
            self.device
        )
        self.action_noise_std = float(cfg.action_noise_std)
        self.motion = build_motion_library(cfg, self.device)
        self.sim = MjlabG1Env(
            num_envs=self.num_envs,
            motion=self.motion,
            device=self.device,
            max_episode_seconds=float(cfg.max_episode_seconds),
        )
        self._compile_cache = CompileCache(
            enabled=bool(cfg.torch_compile), dynamic=False
        )
        self.reset_sampler = None
        self.prev_action = torch.zeros(self.num_envs, ACTION_DIM, device=self.device)
        self._observed: dict[str, torch.Tensor] = {}
        self._clear_metrics()
        self._reset(torch.arange(self.num_envs, device=self.device))

    def set_reset_sampler(self, reset_sampler) -> None:
        """Attach an algorithm-owned reset sampler (``sample``/``record``)."""
        self.reset_sampler = reset_sampler

    def update_reset_state_ids(self) -> None:
        """No-op hook for compatibility with generic :class:`EnvWorker` rollouts."""

    def _sample_reset_frames(self, count: int) -> torch.Tensor:
        if self.reset_sampler is not None:
            return self.reset_sampler.sample(count)
        frame_count = self.motion.num_frames
        if frame_count <= 1:
            return torch.zeros(count, dtype=torch.long, device=self.device)
        frames = torch.randint(0, frame_count - 1, (count,), device=self.device)
        ends = self.motion.ends.to(self.device)
        offsets = self.motion.offsets.to(self.device)
        clip = torch.searchsorted(ends, frames, right=True)
        on_last = (frames == ends[clip] - 1) & (frames > offsets[clip])
        return torch.where(on_last, frames - 1, frames)

    def _reset(self, env_ids: torch.Tensor) -> None:
        motion_id, time = self.motion.locate(
            self._sample_reset_frames(int(env_ids.numel()))
        )
        self.sim.reset(env_ids, motion_id, time)
        self.sim.motion_time[env_ids] += 1
        self.prev_action[env_ids] = 0

    def _observation(self, state, pos, quat, lin, ang, frame):
        full = self.representation.stored_state(state, (pos, quat, lin, ang))
        root = self.representation.root(full)
        _, goal = root_goals(root, *self.motion.goal_window(frame))
        return full, self.representation.encode(full), goal

    def observe(self) -> dict[str, torch.Tensor]:
        """Encoding ``enc`` and actor goal ``goal`` of the coming step."""
        frame = self.sim.frame()
        full, enc, goal = run_compiled(
            "collect-observe",
            self._observation,
            self.sim.state,
            *self.sim.body_state(),
            frame,
            cache=self._compile_cache,
        )
        self._observed = {"state": full, "frame": frame}
        return {"enc": enc, "goal": goal}

    def reset(self, *args, **kwargs) -> tuple[dict[str, torch.Tensor], dict]:
        """Reset all envs and return packed ``{"states": cat(enc, goal)}``."""
        del args, kwargs
        self._reset(torch.arange(self.num_envs, device=self.device))
        obs = self.observe()
        return {"states": torch.cat((obs["enc"], obs["goal"]), dim=-1)}, {}

    def _score(
        self,
        state,
        pos,
        quat,
        lin,
        ang,
        ref_pos,
        ref_rot,
        ref_lin,
        ref_ang,
        action,
        prev_action,
    ):
        return tracking_reward(
            state[:, :3],
            quat_to_rotmat(state[:, 3:7]),
            (pos, quat_to_rotmat(quat), lin, ang),
            (ref_pos, ref_rot, ref_lin, ref_ang),
            state[:, 7:36],
            action,
            prev_action,
            self.sim.soft_limits,
        )

    def step(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Execute ``action`` plus Gaussian noise; return the replay transition.

        The transition holds the state and frame of :meth:`observe`, the
        executed action, the action executed one step earlier, the episode
        end flags, the sim context, the scalar reward, and the executed PD
        targets under ``pd_target``. Envs that finished are reset before this
        returns. The caller owns every returned tensor.
        """
        sim = self.sim
        executed = action + self.action_noise_std * torch.randn_like(action)
        prev_action = self.prev_action.clone()
        context = sim.sim_context.clone()
        result = sim.step(executed)
        with torch.no_grad():
            # The library's kinematics are gathered before the call: filling
            # their cache inside the compiled piece would trace every frame.
            reward = run_compiled(
                "collect-score",
                self._score,
                sim.state,
                *sim.body_state(),
                *self.motion.body_state(result["frame"]),
                executed,
                prev_action,
                cache=self._compile_cache,
            )
        # A fall on the step that also reaches the clip end or the time
        # limit counts as the time-out.
        truncated = result["timeout"] | result["clip_end"]
        failed = result["failure"] & ~truncated
        transition = {
            "state": self._observed["state"],
            "action": executed,
            "prev_action": prev_action,
            "motion_time": self._observed["frame"],
            "terminated": failed,
            "truncated": truncated,
            "sim_context": context,
            "reward": reward,
            "pd_target": sim.pd_offset + sim.action_scale * executed,
        }
        # The transition keeps ``executed``; resetting envs zeroes prev_action.
        self.prev_action.copy_(executed)
        done = (failed | truncated).nonzero(as_tuple=True)[0]
        if done.numel():
            self._reset(done)
        # Episodes that ended on this step draw their start before this step
        # can refresh the reset distribution.
        if self.reset_sampler is not None:
            self.reset_sampler.record(
                result["frame"],
                reward,
                failed,
                result["clip_end"],
                result["timeout"] & ~result["clip_end"],
            )
        self._reward_total += reward.sum()
        self._failures += failed.sum()
        self._episodes += failed.sum() + truncated.sum()
        self._steps += 1
        return transition

    def chunk_step(self, chunk_actions):
        """Standard RLinf single-action chunk step over all envs."""
        actions = torch.as_tensor(chunk_actions, dtype=torch.float32)
        actions = actions.reshape(self.num_envs, -1, ACTION_DIM)
        if actions.shape[1] != 1:
            raise ValueError(
                f"G1MimicEnv steps one action per call, got chunks of {actions.shape[1]}"
            )
        if not self._observed:
            self.observe()
        transition = self.step(actions[:, 0].to(self.device))
        obs = self.observe()
        packed = {"states": torch.cat((obs["enc"], obs["goal"]), dim=-1)}
        rewards = transition["reward"].detach().cpu().unsqueeze(1)
        terminations = transition["terminated"].detach().cpu().unsqueeze(1)
        truncations = transition["truncated"].detach().cpu().unsqueeze(1)
        return [packed], rewards, terminations, truncations, [{}]

    def _clear_metrics(self) -> None:
        self._reward_total = torch.zeros((), device=self.device)
        self._failures = torch.zeros((), dtype=torch.long, device=self.device)
        self._episodes = torch.zeros((), dtype=torch.long, device=self.device)
        self._steps = 0

    def consume_metrics(self) -> dict[str, float]:
        """This rank's statistics since the last call, then a reset."""
        totals = torch.stack(
            (self._reward_total, self._failures.float(), self._episodes.float())
        )
        transitions = max(self._steps * self.num_envs, 1)
        if self.reset_sampler is not None:
            value = self.reset_sampler.value
            reward, failures, episodes, value_mean, value_min = torch.cat(
                (totals, value.mean().reshape(1), value.min().reshape(1))
            ).tolist()
            self._clear_metrics()
            return {
                "sim/reward_mean": reward / transitions,
                "sim/terminal_rate": failures / transitions,
                "sim/episodes_finished": episodes,
                "reset/bin_value_mean": value_mean,
                "reset/bin_value_min": value_min,
            }
        reward, failures, episodes = totals.tolist()
        self._clear_metrics()
        return {
            "sim/reward_mean": reward / transitions,
            "sim/terminal_rate": failures / transitions,
            "sim/episodes_finished": episodes,
        }

    def state_dict(self) -> dict:
        """Reset-sampler statistics when attached. Simulator state is not saved."""
        if self.reset_sampler is None:
            return {}
        return {"reset_sampler": self.reset_sampler.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        if self.reset_sampler is not None and "reset_sampler" in state:
            self.reset_sampler.load_state_dict(state["reset_sampler"])
