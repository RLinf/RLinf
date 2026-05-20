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

import os
import sys

import gymnasium as gym
import numpy as np
import torch

from rlinf.envs.isaaclab.isaaclab_env import IsaaclabBaseEnv


class PolarisEnv(IsaaclabBaseEnv):
    """RLinf wrapper for PolaRiS environments.

    The wrapper:
    - Launches PolaRiS inside a subprocess via ``SubProcIsaacLabEnv``.
    - Exposes the canonical RLinf observation dict (``main_images``,
      ``wrist_images``, ``states``, ``task_descriptions``).
    - Supports ``chunk_step``, metrics tracking, auto-reset and
      ``ignore_terminations`` inherited from ``IsaaclabBaseEnv``.
    """

    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
    ):
        # PolaRiS currently supports only num_envs=1 per subprocess because
        # the Gaussian-splat renderer is not vectorised.  Guard against
        # accidental misconfiguration.
        if num_envs != 1:
            raise ValueError(
                f"PolarisEnv only supports num_envs=1 per worker, got {num_envs}. "
                "Adjust total_num_envs and env_world_size so that each env "
                "worker receives exactly 1 environment."
            )
        super().__init__(
            cfg,
            num_envs,
            seed_offset,
            total_num_processes,
            worker_info,
        )

    def _make_env_function(self):
        """Return a factory that creates the PolaRiS env inside a subprocess.

        The factory is pickled and shipped to a child process by
        ``SubProcIsaacLabEnv``.  It must:
        1. Start the Isaac Sim ``AppLauncher``.
        2. Create a ``ManagerBasedRLSplatEnv`` wrapped inside a thin
           ``InnerPolarisEnv`` adapter that translates the standard
           ``reset(seed, env_ids)`` / ``step(actions)`` protocol used by
           ``SubProcIsaacLabEnv._torch_worker`` into PolaRiS-specific calls.
        3. Return ``(inner_env, sim_app)``.
        """
        cfg = self.cfg
        task_name = self.isaaclab_env_id
        num_envs = self.num_envs
        seed = self.seed
        open_loop_horizon = getattr(cfg.init_params, "open_loop_horizon", None)

        def _make_polaris():
            for key in list(sys.modules.keys()):
                if (
                    key.startswith("polaris.")
                    or key.startswith("isaaclab.")
                    or key.startswith("isaacsim")
                    or key.startswith("omni.")
                ):
                    del sys.modules[key]

            os.environ.pop("DISPLAY", None)
            for _key in ("ISAAC_PATH", "EXP_PATH", "CARB_APP_PATH"):
                os.environ.pop(_key, None)

            from isaaclab.app import AppLauncher

            sim_app = AppLauncher(headless=True, enable_cameras=True).app

            import polaris.environments  # noqa: F401
            from polaris.utils import load_eval_initial_conditions, parse_env_cfg

            dataset_path = getattr(cfg.init_params, "dataset_path", None)
            if dataset_path is None:
                dataset_path = os.environ.get("POLARIS_DATA_PATH", None)
            if dataset_path is not None:
                os.environ["POLARIS_DATA_PATH"] = str(dataset_path)

            usd_file = cfg.init_params.usd_file
            env_cfg = parse_env_cfg(
                task_name,
                usd_file=usd_file,
                device="cuda",
                num_envs=num_envs,
                use_fabric=True,
            )
            env_cfg.seed = seed

            if hasattr(cfg.init_params, "episode_length_s"):
                env_cfg.episode_length_s = cfg.init_params.episode_length_s
            if hasattr(cfg.init_params, "wrist_cam"):
                env_cfg.scene.wrist_cam.height = cfg.init_params.wrist_cam.height
                env_cfg.scene.wrist_cam.width = cfg.init_params.wrist_cam.width
            if hasattr(cfg.init_params, "table_cam"):
                for cam_name in ("external_cam",):
                    if hasattr(env_cfg.scene, cam_name):
                        getattr(
                            env_cfg.scene, cam_name
                        ).height = cfg.init_params.table_cam.height
                        getattr(
                            env_cfg.scene, cam_name
                        ).width = cfg.init_params.table_cam.width

            real_env = gym.make(task_name, cfg=env_cfg)

            language_instruction, initial_conditions = load_eval_initial_conditions(
                real_env.usd_file
            )

            class _InnerPolarisEnv:
                """Adapts PolaRiS's custom API to the protocol expected by
                ``SubProcIsaacLabEnv._torch_worker`` (``reset`` / ``step`` /
                ``close`` / ``device``)."""

                def __init__(self, open_loop_horizon=None):
                    self.env = real_env
                    self.language_instruction = language_instruction
                    self.initial_conditions = initial_conditions
                    self._ic_idx = 0
                    self.device = "cuda"
                    self._chunk_step_counter = 0
                    self._open_loop_horizon = open_loop_horizon

                def reset(self, seed=None, env_ids=None):
                    self._chunk_step_counter = 0
                    ic = self.initial_conditions[self._ic_idx]
                    obs, info = self.env.reset(object_positions=ic, expensive=True)
                    self._ic_idx = (self._ic_idx + 1) % len(self.initial_conditions)
                    return obs, info

                def step(self, actions):
                    if not isinstance(actions, torch.Tensor):
                        actions = torch.as_tensor(actions, device=self.device)
                    else:
                        actions = actions.to(self.device)
                    try:
                        if self._open_loop_horizon is not None:
                            needs_render = (
                                self._chunk_step_counter == 0
                                or self._chunk_step_counter >= self._open_loop_horizon
                            )
                            if needs_render:
                                self._chunk_step_counter = 0
                            result = self.env.step(actions, expensive=needs_render)
                            self._chunk_step_counter += 1
                            return result
                        return self.env.step(actions, expensive=True)
                    except RuntimeError:
                        return self.env.step(actions, expensive=False)

                def close(self):
                    self.env.close()

            inner_env = _InnerPolarisEnv(open_loop_horizon=open_loop_horizon)

            return inner_env, sim_app

        return _make_polaris

    def _wrap_obs(self, obs):
        """Convert raw PolaRiS observations into the canonical RLinf dict.

        Keys returned
        -------------
        main_images : torch.Tensor  [num_envs, H, W, 3]
            External (table) camera RGB.
        wrist_images : torch.Tensor  [num_envs, H, W, 3]
            Wrist camera RGB.
        states : torch.Tensor  [num_envs, 8]
            Robot proprioception: ``[arm_joint_pos (7), gripper_pos (1)]``.
        task_descriptions : list[str]
            Natural-language instruction repeated for every env.
        """
        splat = obs.get("splat", {})
        main_img = splat.get("external_cam", splat.get("camera", None))
        wrist_img = splat.get("wrist_cam", None)

        def _to_tensor(img):
            if img is None:
                return None
            if isinstance(img, np.ndarray):
                if np.issubdtype(img.dtype, np.floating):
                    if img.max() <= 1.0:
                        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                    else:
                        img = np.where(img <= 1.0, img * 255, img)
                        img = img.clip(0, 255).astype(np.uint8)
                elif img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                return torch.from_numpy(img.copy()).to(self.device).unsqueeze(0)
            if isinstance(img, torch.Tensor):
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                return img.to(self.device)
            return None

        main_images = _to_tensor(main_img)
        wrist_images = _to_tensor(wrist_img)

        policy = obs.get("policy", {})
        arm_joint_pos = policy.get("arm_joint_pos", None)
        gripper_pos = policy.get("gripper_pos", None)

        if arm_joint_pos is not None and gripper_pos is not None:
            if isinstance(arm_joint_pos, np.ndarray):
                arm_joint_pos = torch.from_numpy(arm_joint_pos).to(self.device)
            if isinstance(gripper_pos, np.ndarray):
                gripper_pos = torch.from_numpy(gripper_pos).to(self.device)
            if arm_joint_pos.dim() == 1:
                arm_joint_pos = arm_joint_pos.unsqueeze(0)
            if gripper_pos.dim() == 1:
                gripper_pos = gripper_pos.unsqueeze(0)
            states = torch.cat([arm_joint_pos, gripper_pos], dim=-1).float()
        else:
            states = torch.zeros(
                (self.num_envs, 8), dtype=torch.float32, device=self.device
            )

        instruction = [self.task_description] * self.num_envs

        env_obs = {
            "task_descriptions": instruction,
            "states": states,
        }
        if main_images is not None:
            env_obs["main_images"] = main_images
        if wrist_images is not None:
            env_obs["wrist_images"] = wrist_images

        return env_obs

    def step(self, actions=None, auto_reset=True):
        obs, step_reward, terminations, truncations, infos = self.env.step(actions)

        terminations = terminations.clone()
        truncations = truncations.clone()

        if isinstance(infos, dict) and "rubric" in infos:
            rubric = infos["rubric"]
            rubric_success = rubric.get("success", False)
            if isinstance(rubric_success, bool):
                if rubric_success:
                    terminations[:] = True
            elif isinstance(rubric_success, (torch.Tensor, np.ndarray)):
                t = torch.as_tensor(rubric_success, device=terminations.device)
                terminations = t.bool().reshape_as(terminations)

        step_reward = self._calc_step_reward(terminations)

        obs = self._wrap_obs(obs)

        self._elapsed_steps += 1

        truncations = (self.elapsed_steps >= self.cfg.max_episode_steps) | truncations

        dones = terminations | truncations

        merged_infos = {}
        if isinstance(infos, dict):
            merged_infos.update(infos)
        infos = self._record_metrics(step_reward, terminations, merged_infos)

        if isinstance(merged_infos, dict) and "rubric" in merged_infos:
            rubric = merged_infos["rubric"]
            progress = rubric.get("progress", 0.0)
            rubric_metrics = rubric.get("metrics", {})
            criteria_reached = rubric_metrics.get("criteria_ever_reached", 0)
            criteria_total = rubric_metrics.get("criteria_total", 0)

            infos["episode"]["progress"] = torch.tensor(
                [progress], dtype=torch.float32, device=self.device
            )
            infos["episode"]["criteria_ever_reached"] = torch.tensor(
                [criteria_reached], dtype=torch.float32, device=self.device
            )
            infos["episode"]["criteria_total"] = torch.tensor(
                [criteria_total], dtype=torch.float32, device=self.device
            )

        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = terminations
            terminations[:] = False

        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)

        return (
            obs,
            step_reward,
            terminations,
            truncations,
            infos,
        )

    @property
    def total_num_group_envs(self):
        return self.num_envs // self.cfg.group_size
