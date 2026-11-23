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

"""State-based IsaacLab dexterous-manipulation environments for SAPG.

These tasks (AllegroHand / ShadowHand in-hand reorientation) expose a low
dimensional proprioceptive observation instead of images, so they pair with the
MLP policies rather than the VLA policies used by ``stack_cube``. The
observation is returned verbatim under ``states``; the SAPG exploration
embedding is concatenated later by the policy, keeping the simulator wrapper
free of algorithm-specific logic.

The official SAPG codebase targets IsaacGym, which is no longer maintained.
IsaacLab (Isaac Sim backend) ships maintained equivalents of these tasks, so we
reuse RLinf's existing IsaacLab interface instead of porting the IsaacGym code.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace

import gymnasium as gym
import torch

from rlinf.utils.nested_dict_process import clone_nested_to_cpu

from ..isaaclab_env import IsaaclabBaseEnv

# RLinf-registered task name -> real IsaacLab gym id. RLinf registers short,
# backend-agnostic names, while IsaacLab's gym registry uses the direct
# in-hand reorientation task ids.
ISAACLAB_TASK_ID_MAP = {
    "Isaac-Allegro-Hand-v0": "Isaac-Repose-Cube-Allegro-Direct-v0",
    "Isaac-Shadow-Hand-v0": "Isaac-Repose-Cube-Shadow-Direct-v0",
}

# Python package dirs of a standalone Isaac Sim install, mirroring
# setup_python_env.sh. Ray workers do not inherit the driver's PYTHONPATH, so a
# standalone (non-pip) Isaac Sim must be added to sys.path inside the worker.
_ISAAC_SIM_PYTHON_SUBDIRS = [
    "kit/python/lib/python3.11/site-packages",
    "python_packages",
    "exts/isaacsim.simulation_app",
    "extsDeprecated/omni.isaac.kit",
    "kit/kernel/py",
    "kit/plugins/bindings-python",
    "exts/isaacsim.robot_motion.lula/pip_prebundle",
    "exts/isaacsim.asset.exporter.urdf/pip_prebundle",
    "extscache/omni.kit.pip_archive-0.0.0+69cbf6ad.lx64.cp311/pip_prebundle",
    "exts/omni.isaac.core_archive/pip_prebundle",
    "exts/omni.isaac.ml_archive/pip_prebundle",
    "exts/omni.pip.compute/pip_prebundle",
    "exts/omni.pip.cloud/pip_prebundle",
]


class _ExternalResetMixin:
    """Let RLinf retain terminal observations and own environment resets."""

    def __init__(self, *args, **kwargs):
        self._inside_step = False
        super().__init__(*args, **kwargs)

    def step(self, action):
        self._inside_step = True
        try:
            return super().step(action)
        finally:
            self._inside_step = False

    def _reset_idx(self, env_ids):
        if not self._inside_step:
            return super()._reset_idx(env_ids)


class _DexHandSubprocessAdapter(gym.Wrapper):
    """Expose partial resets and exchange state tensors across CUDA namespaces."""

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
        env_ids: torch.Tensor | None = None,
    ):
        if env_ids is None:
            return clone_nested_to_cpu(self.env.reset(seed=seed, options=options))
        env = self.env
        if seed is not None:
            env.seed(seed)
        env._reset_idx(env_ids.to(env.device))
        env.scene.write_data_to_sim()
        env.sim.forward()
        return clone_nested_to_cpu((env._get_observations(), env.extras))

    def step(self, action: torch.Tensor):
        obs, reward, terminated, truncated, infos = self.env.step(action)
        infos["success_count"] = self.env.successes.clone()
        return clone_nested_to_cpu((obs, reward, terminated, truncated, infos))


def _setup_standalone_isaac_sim():
    """Make a standalone Isaac Sim importable inside a Ray worker."""
    isaac_path = os.environ.get("ISAAC_PATH")
    if not isaac_path or not os.path.isdir(isaac_path):
        return
    os.environ["ISAAC_PATH"] = isaac_path
    os.environ.setdefault("CARB_APP_PATH", os.path.join(isaac_path, "kit"))
    os.environ.setdefault("EXP_PATH", os.path.join(isaac_path, "apps"))
    for sub in _ISAAC_SIM_PYTHON_SUBDIRS:
        p = os.path.join(isaac_path, sub)
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


class IsaaclabDexHandEnv(IsaaclabBaseEnv):
    """State-based IsaacLab dexterous-hand task (e.g. AllegroHand, ShadowHand)."""

    def _make_env_function(self):
        def make_env_isaaclab():
            # Force headless mode to avoid GLX errors on servers without display.
            os.environ.pop("DISPLAY", None)
            _setup_standalone_isaac_sim()

            from isaaclab.app import AppLauncher

            # Kit probes physical GPUs. Only the child uses the physical
            # namespace, while the parent keeps RLinf's assigned visibility.
            device = f"cuda:{self.worker_info.accelerator_rank}"
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            launcher = AppLauncher(
                headless=True, enable_cameras=False, multi_gpu=False, device=device
            )
            asset_root = self.cfg.init_params.get("asset_root")
            if asset_root:
                import carb

                carb.settings.get_settings().set(
                    "/persistent/isaac/asset_root/cloud", asset_root
                )
            from isaaclab_tasks.utils import load_cfg_from_registry

            task_id = ISAACLAB_TASK_ID_MAP.get(
                self.isaaclab_env_id, self.isaaclab_env_id
            )
            isaac_env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
            isaac_env_cfg.sim.device = device
            isaac_env_cfg.seed = self.seed
            isaac_env_cfg.scene.num_envs = self.cfg.init_params.num_envs
            isaac_env_cfg.episode_length_s = (
                self.cfg.max_episode_steps
                * isaac_env_cfg.sim.dt
                * isaac_env_cfg.decimation
            )

            from gymnasium.envs.registration import load_env_creator

            task_spec = gym.spec(task_id)
            task_class = load_env_creator(task_spec.entry_point)

            class ExternallyResetHandEnv(_ExternalResetMixin, task_class):
                pass

            env = gym.make(
                replace(task_spec, entry_point=ExternallyResetHandEnv),
                cfg=isaac_env_cfg,
                render_mode="rgb_array",
            ).unwrapped
            return _DexHandSubprocessAdapter(env), launcher.app

        return make_env_isaaclab

    def _wrap_obs(self, obs):
        # IsaacLab direct tasks expose the low-dimensional observation under
        # the "policy" key, as a [num_envs, obs_dim] tensor on the sim device.
        states = obs["policy"]
        return {"states": states}

    def step(self, actions=None, auto_reset=True):
        obs, reward, terminated, truncated, infos = super().step(
            actions, auto_reset=auto_reset
        )
        return obs, reward * self.cfg.reward_coef, terminated, truncated, infos

    def _record_metrics(self, step_reward, terminations, infos):
        self.returns += step_reward
        successes = infos["success_count"]
        self.success_once |= successes > 0
        infos["episode"] = {
            "success_once": self.success_once.clone(),
            "success_count": successes.clone(),
            "return": self.returns.clone(),
            "episode_len": self.elapsed_steps.clone(),
            "reward": self.returns / self.elapsed_steps,
        }
        return infos
