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

"""Ray actor that owns the RoboLab Isaac Sim runtime."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import ray
import torch

from rlinf.utils.logging import get_logger


@ray.remote(
    num_cpus=1, num_gpus=0, max_concurrency=1, max_restarts=0, max_task_retries=0
)
class RoboLabProcess:
    """Own one RoboLab Isaac application in a synchronous Ray actor."""

    def __init__(self, robolab_dir: str, settings: dict):
        # RoboLab resolves assets relative to its source checkout. Change the
        # actor's private working directory before importing RoboLab or Isaac.
        os.chdir(robolab_dir)
        import cv2  # noqa: F401 -- RoboLab imports OpenCV before Isaac Lab.
        from isaaclab.app import AppLauncher

        self.logger = get_logger()
        self.app = None
        self.env = None
        self._closed = False
        try:
            self.app = AppLauncher(
                headless=True,
                enable_cameras=True,
                device=settings.get("device", "cuda:0"),
                renderer=settings.get("renderer", "realtime"),
                kit_args=settings.get("kit_args", ""),
            ).app
            self._init_env(settings)
        except BaseException:
            self.close()
            raise

    def _init_env(self, settings: dict) -> None:
        import omni.kit.app
        import omni.timeline
        import robolab.constants
        from openpi_client import image_tools
        from robolab.constants import set_output_dir
        from robolab.core.environments.runtime import create_env, end_episode
        from robolab.core.logging.results import get_current_subtask_info
        from robolab.core.task.instruction import instruction_text
        from robolab.registrations.droid.auto_env_registrations_jointpos import (
            auto_register_droid_envs,
        )

        self._image_tools = image_tools
        self._end_episode = end_episode
        self._get_current_subtask_info = get_current_subtask_info
        self.enable_subtask = bool(settings.get("enable_subtask", True))
        robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = self.enable_subtask
        self.task = settings["task"]
        auto_register_droid_envs(task=self.task)
        set_output_dir(settings["output_dir"])
        self.env, env_cfg = create_env(
            self.task,
            device=settings.get("device", "cuda:0"),
            seed=settings["seed"],
            num_envs=settings["num_envs"],
            instruction_type=settings.get("instruction_type"),
            policy="rlinf",
            renderer=settings.get("renderer", "realtime"),
            rendering_mode=settings.get("rendering_mode"),
        )
        self.prompt = instruction_text(env_cfg.instruction)
        self.timeline = omni.timeline.get_timeline_interface()
        self.kit_app = omni.kit.app.get_app()
        self.num_envs = settings["num_envs"]
        self.max_steps = settings["max_episode_steps"]
        self.auto_reset = bool(settings.get("auto_reset", False))
        self.steps = np.zeros(self.num_envs, dtype=np.int32)
        self.finished = np.zeros(self.num_envs, dtype=bool)
        self.success = np.zeros(self.num_envs, dtype=np.float32)
        self.score = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_indices = np.arange(self.num_envs, dtype=np.int64)
        self.last_obs = None
        self.run_index = -1
        self.episode_open = False

    def ready(self) -> dict:
        """Return metadata after Isaac and the RoboLab environment are ready."""
        return {
            "task_descriptions": [self.prompt] * self.num_envs,
            "max_episode_length": int(self.env.max_episode_length),
            "action_bounds": {
                "low": np.asarray(self.env.single_action_space.low),
                "high": np.asarray(self.env.single_action_space.high),
            },
        }

    def _finish_episode(self) -> None:
        if getattr(self, "episode_open", False):
            self._end_episode(self.env)
            self.episode_open = False

    def _reset_selected(self, env_ids: list[int]):
        """Reset selected slots through RoboLab's pinned IsaacLab internals."""
        ids = torch.as_tensor(env_ids, device=self.env.device, dtype=torch.long)
        if not ids.numel():
            return self.last_obs
        self.env._frozen_envs[ids] = False
        self.env._pre_step_frozen[ids] = False
        for env_id in ids.tolist():
            self.env._env_results.pop(env_id, None)
            self.env._env_term_step.pop(env_id, None)
        has_stepped = self.env._has_stepped
        self.env._has_stepped = False
        try:
            self.env._reset_idx(ids)
        finally:
            self.env._has_stepped = has_stepped
        if self.env.recorder_manager is not None:
            for env_id in ids.tolist():
                self.episode_indices[env_id] += self.num_envs
                self.env.recorder_manager.set_episode_index(
                    int(self.episode_indices[env_id]), env_ids=[env_id]
                )
        self.last_obs = self.env.observation_manager.compute()
        indices = ids.cpu().numpy()
        self.steps[indices] = 0
        self.finished[indices] = False
        self.success[indices] = 0
        self.score[indices] = 0
        return self.last_obs

    def _pack_obs(self, raw) -> dict:
        active_mask = np.zeros(self.num_envs, dtype=bool)
        active_mask[self.env.active_env_ids] = True
        active_mask &= ~self.finished
        images = raw["image_obs"]
        proprio = raw["proprio_obs"]
        main_images = images["over_shoulder_left_camera"].detach().cpu().numpy()
        wrist_images = images["wrist_cam"].detach().cpu().numpy()
        return {
            "main_images": np.stack(
                [
                    self._image_tools.resize_with_pad(image, 224, 224)
                    for image in main_images
                ]
            ),
            "wrist_images": np.stack(
                [
                    self._image_tools.resize_with_pad(image, 224, 224)
                    for image in wrist_images
                ]
            ),
            "states": np.concatenate(
                [
                    proprio["arm_joint_pos"].detach().cpu().numpy(),
                    proprio["gripper_pos"].detach().cpu().numpy(),
                ],
                axis=-1,
            ).astype(np.float32),
            "task_descriptions": [self.prompt] * self.num_envs,
            "active_mask": active_mask,
        }

    def reset(self, seed: int | None = None, env_ids: list[int] | None = None) -> dict:
        """Reset all or selected environment slots."""
        if env_ids is None:
            self._finish_episode()
            self.env.reset_eval_state()
            # RoboLab's evaluator resets twice before taking the first action.
            self.last_obs, _ = self.env.reset(seed=seed)
            self.last_obs, _ = self.env.reset()
            self.run_index += 1
            self.episode_indices[:] = np.arange(self.num_envs)
            if self.env.recorder_manager is not None:
                self.env.recorder_manager.set_hdf5_file(
                    f"run_{self.run_index}.hdf5"
                )
                for env_id in range(self.num_envs):
                    self.env.recorder_manager.set_episode_index(
                        env_id, env_ids=[env_id]
                    )
            self.episode_open = True
            self.steps[:] = 0
            self.finished[:] = False
            self.success[:] = 0
            self.score[:] = 0
        else:
            if self.last_obs is None:
                raise RuntimeError("Reset the full RoboLab batch before a partial reset")
            self._reset_selected(env_ids)
        return self._pack_obs(self.last_obs)

    def chunk_step(self, actions: np.ndarray) -> dict:
        """Execute one action chunk and return RLinf-compatible arrays."""
        horizon = actions.shape[1]
        rewards = np.zeros((self.num_envs, horizon), dtype=np.float32)
        terminations = np.zeros((self.num_envs, horizon), dtype=bool)
        truncations = np.zeros((self.num_envs, horizon), dtype=bool)
        executed_steps = np.zeros(self.num_envs, dtype=np.int32)
        newly_finished = np.zeros(self.num_envs, dtype=bool)
        for index in range(horizon):
            if self.finished.all():
                break
            while not self.timeline.is_playing():
                self.kit_app.update()
            active = ~self.finished
            action = torch.as_tensor(actions[:, index], device=self.env.device)
            action[~torch.as_tensor(active, device=self.env.device)] = 0
            self.last_obs, _, _, _, _ = self.env.step(action)
            self.steps[active] += 1
            executed_steps[active] += 1
            for env_id, result in enumerate(self.env.get_env_results()):
                if self.finished[env_id]:
                    continue
                native_success = result["success"]
                if native_success is None and (
                    self.max_steps is None
                    or self.steps[env_id] < self.max_steps
                ):
                    continue
                self.finished[env_id] = True
                newly_finished[env_id] = True
                self.success[env_id] = float(native_success is True)
                if self.enable_subtask:
                    subtask_info = self._get_current_subtask_info(
                        self.env, env_id=env_id
                    )
                    if subtask_info is None or subtask_info.get("score") is None:
                        raise RuntimeError(
                            f"Expected a subtask score for {self.task}, env "
                            f"{env_id}, but received {subtask_info}"
                        )
                    self.score[env_id] = (
                        1.0
                        if native_success is True
                        else float(subtask_info["score"])
                    )
                rewards[env_id, index] = self.success[env_id]
                terminations[env_id, index] = bool(native_success is True)
                truncations[env_id, index] = not terminations[env_id, index]

        final_observation = self._pack_obs(self.last_obs)
        episode = {
            "success_once": self.success.copy(),
            "return": self.success.copy(),
            "episode_len": self.steps.astype(np.float32),
            "reward": self.success / np.maximum(self.steps, 1),
        }
        if self.enable_subtask:
            episode["score"] = self.score.copy()
        if newly_finished.any():
            terminations[newly_finished, -1] |= self.success[
                newly_finished
            ].astype(bool)
            truncations[newly_finished, -1] |= ~self.success[
                newly_finished
            ].astype(bool)
        if self.auto_reset and newly_finished.any():
            self._reset_selected(np.flatnonzero(newly_finished).tolist())
        elif self.finished.all():
            self._finish_episode()
        return {
            "observation": self._pack_obs(self.last_obs),
            "final_observation": final_observation,
            "final_mask": newly_finished,
            "executed_steps": executed_steps,
            "rewards": rewards,
            "terminations": terminations,
            "truncations": truncations,
            "episode": episode,
        }

    def render_camera(self, camera_name: str, env_id: int) -> np.ndarray:
        """Return one native camera frame without stepping."""
        if self.last_obs is None:
            raise RuntimeError("Reset RoboLab before reading a camera")
        return (
            self.last_obs["image_obs"][camera_name][env_id]
            .detach()
            .cpu()
            .numpy()
        )

    def get_status(self) -> list[dict[str, Any]]:
        """Return native and RLinf termination status for every slot."""
        statuses = []
        for env_id, result in enumerate(self.env.get_env_results()):
            native_success = result["success"]
            statuses.append(
                {
                    "steps": int(result["step"])
                    if native_success is not None
                    else int(self.env.episode_length_buf[env_id].item()),
                    "step_limit": int(self.env.max_episode_length),
                    "success": native_success,
                    "terminated": bool(
                        self.finished[env_id] and self.success[env_id]
                    ),
                    "truncated": bool(
                        self.finished[env_id] and not self.success[env_id]
                    ),
                }
            )
        return statuses

    def close(self) -> None:
        """Close recording, the environment, and the Isaac application."""
        if self._closed:
            return
        self._closed = True
        if self.env is not None:
            try:
                self._finish_episode()
                self.env.close()
            except Exception:
                self.logger.exception("Failed to close the RoboLab environment")
            finally:
                self.env = None
        if self.app is not None:
            try:
                self.app.close()
            finally:
                self.app = None
