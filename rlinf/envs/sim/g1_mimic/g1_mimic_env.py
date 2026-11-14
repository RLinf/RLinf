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

"""G1 motion-tracking training vector for SuperMimic.

The SuperMimic env worker drives this env one control step at a time and
streams every transition to the actor's replay. A gym ``chunk_step`` cannot
carry the per-step normalizer statistics, so the env exposes the step as
:meth:`G1MimicEnv.observe` and :meth:`G1MimicEnv.step`.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from omegaconf import DictConfig

from rlinf.algorithms.supermimic.representations import make_representation
from rlinf.algorithms.supermimic.reset_sampler import ReturnResetSampler
from rlinf.algorithms.supermimic.scan import root_goals
from rlinf.envs.sim.g1_mimic.compile_cache import CompileCache, run_compiled
from rlinf.envs.sim.g1_mimic.constants import ACTION_DIM
from rlinf.envs.sim.g1_mimic.features import quat_to_rotmat
from rlinf.envs.sim.g1_mimic.g1_fk import set_kinematics_compile
from rlinf.envs.sim.g1_mimic.mjlab_env import MjlabG1Env
from rlinf.envs.sim.g1_mimic.motion import MotionLibrary, load_motion_library
from rlinf.envs.sim.g1_mimic.reward import tracking_reward
from rlinf.models.embodiment.supermimic.normalization import merged_moments
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


class G1MimicEnv:
    """One rank's training envs.

    Resets draw a library frame from the return-based sampler, write that
    frame (with mjlab's reset noise) and point the reference clock at the
    next frame, the target of the first step.
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
        self._compile_cache = CompileCache(enabled=bool(cfg.torch_compile))
        self.group: dist.ProcessGroup | None = None
        self.reset_sampler = ReturnResetSampler(
            self.motion.ends, self.num_envs, self.device
        )
        self.prev_action = torch.zeros(self.num_envs, ACTION_DIM, device=self.device)
        self._observed: dict[str, torch.Tensor] = {}
        self._clear_metrics()
        self._reset(torch.arange(self.num_envs, device=self.device))

    def set_process_group(self, group: dist.ProcessGroup | None) -> None:
        """Group of every training env rank; statistics are merged over it."""
        self.group = group
        self.reset_sampler.group = group

    def _reset(self, env_ids: torch.Tensor) -> None:
        motion_id, time = self.motion.locate(
            self.reset_sampler.sample(int(env_ids.numel()))
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

    def policy_input_stats(
        self, obs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Moments of the encodings and goals of :meth:`observe`, over every env rank.

        A collective: every env rank calls it once per step.
        """
        return {
            "enc": merged_moments(obs["enc"], self.group),
            "goal": merged_moments(obs["goal"], self.group),
        }

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
        end flags and the sim context, plus the merged moments of the
        executed PD targets under ``target_stats``. Envs that finished are
        reset before this returns. The caller owns every returned tensor.
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
        self.reset_sampler.record(
            result["frame"],
            reward,
            failed,
            result["clip_end"],
            result["timeout"] & ~result["clip_end"],
        )
        transition = {
            "state": self._observed["state"],
            "action": executed,
            "prev_action": prev_action,
            "motion_time": self._observed["frame"],
            "terminated": failed,
            "truncated": truncated,
            "sim_context": context,
            "target_stats": merged_moments(
                sim.pd_offset + sim.action_scale * executed, self.group
            ),
        }
        # The transition keeps ``executed``; resetting envs zeroes prev_action.
        self.prev_action.copy_(executed)
        done = (failed | truncated).nonzero(as_tuple=True)[0]
        if done.numel():
            self._reset(done)
        self._reward_total += reward.sum()
        self._failures += failed.sum()
        self._episodes += failed.sum() + truncated.sum()
        self._steps += 1
        return transition

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
        value = self.reset_sampler.value
        reward, failures, episodes, value_mean, value_min = torch.cat(
            (totals, value.mean().reshape(1), value.min().reshape(1))
        ).tolist()
        transitions = max(self._steps * self.num_envs, 1)
        self._clear_metrics()
        return {
            "sim/reward_mean": reward / transitions,
            "sim/terminal_rate": failures / transitions,
            "sim/episodes_finished": episodes,
            "reset/bin_value_mean": value_mean,
            "reset/bin_value_min": value_min,
        }

    def state_dict(self) -> dict:
        """Reset-sampler statistics. Simulator state is not saved."""
        return {"reset_sampler": self.reset_sampler.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.reset_sampler.load_state_dict(state["reset_sampler"])
