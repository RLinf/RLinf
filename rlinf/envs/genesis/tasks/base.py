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

"""Abstract base class for Genesis tasks."""

from __future__ import annotations

import abc
from typing import Any

import numpy as np
import torch


class GenesisTaskBase(abc.ABC):
    """Interface that every Genesis task must implement.

    A *task* is responsible for:
    * populating a Genesis scene with a robot, objects, and cameras;
    * resetting (randomizing) the scene state;
    * computing per-step rewards;
    * extracting raw observation data that ``GenesisEnv`` will wrap into
      the canonical RLinf observation dict.

    Attributes:
        robot: The Genesis robot entity created during :meth:`build_scene`.
        eef_link: The end-effector link on the robot.
        camera: The main camera sensor (may be ``None`` if images are
            not required).
        motor_dofs: Index array for the arm motor DOFs.
        finger_dofs: Index array for the gripper finger DOFs.
        task_description: Natural-language description of the task.
    """

    # Subclasses should set these in ``build_scene``.
    robot: Any = None
    eef_link: Any = None
    camera: Any = None
    motor_dofs: Any = None
    finger_dofs: Any = None
    task_description: str = ""

    def seed(self, seed: int) -> None:
        """Set task-level random seed.

        Subclasses can override when they maintain their own RNG.
        """

    def total_num_reset_states(self) -> int:
        """Return the number of available reset states.

        Tasks without a discrete reset-state table can keep the default.
        """
        return int(np.iinfo(np.uint32).max // 2)

    @abc.abstractmethod
    def build_scene(self, scene, cfg) -> None:
        """Add entities (robot, objects, cameras) to *scene*.

        Called **before** ``scene.build()`` so that Genesis can compile
        the scene graph.

        Args:
            scene: A ``genesis.Scene`` instance.
            cfg: The ``env.train`` (or ``env.eval``) Hydra config section.
        """

    @abc.abstractmethod
    def reset(
        self,
        scene,
        num_envs: int,
        envs_idx: torch.Tensor | None = None,
        reset_state_ids: torch.Tensor | None = None,
    ) -> None:
        """Reset the task state (robot + objects) for the given env indices.

        After this call ``scene.step()`` should be called at least once to
        settle the physics state before observations are read.

        Args:
            scene: The built Genesis scene.
            num_envs: Total number of parallel environments.
            envs_idx: Optional tensor of environment indices to reset.
                ``None`` means reset all environments.
            reset_state_ids: Optional tensor of reset-state IDs aligned
                with ``envs_idx`` (or all envs when ``envs_idx`` is None).
        """

    def apply_action(self, actions: torch.Tensor) -> None:
        """Apply policy actions to the task robot.

        The default implementation controls arm and gripper position DOFs.
        Subclasses can override for custom control spaces.
        """
        if self.robot is None:
            raise RuntimeError("Task robot is not initialized.")

        if self.motor_dofs is not None:
            self.robot.control_dofs_position(
                actions[:, : len(self.motor_dofs)],
                self.motor_dofs,
            )
        if self.finger_dofs is not None:
            self.robot.control_dofs_position(
                actions[
                    :,
                    len(self.motor_dofs) : len(self.motor_dofs) + len(self.finger_dofs),
                ],
                self.finger_dofs,
            )

    def compute_step_outcomes(
        self,
        scene,
        num_envs: int,
        elapsed_steps: torch.Tensor,
        max_episode_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Compute reward, done signals and infos for one step.

        Default behavior preserves current GenesisEnv semantics:
        termination by task success, truncation by max episode steps.
        """
        reward, success = self.compute_reward(scene, num_envs)
        terminations = success.bool()
        truncations = elapsed_steps >= max_episode_steps
        infos: dict[str, Any] = {"success": success}
        return reward, terminations, truncations, infos

    def get_task_descriptions(self, num_envs: int) -> list[str]:
        """Return per-env language descriptions."""
        return [self.task_description] * num_envs

    @abc.abstractmethod
    def compute_reward(self, scene, num_envs: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute rewards and success flags for the current state.

        Args:
            scene: The built Genesis scene.
            num_envs: Total number of parallel environments.

        Returns:
            A ``(reward, success)`` tuple where both tensors have shape
            ``(num_envs,)``.  ``reward`` is float and ``success`` is bool.
        """

    @abc.abstractmethod
    def get_obs(
        self,
        scene,
        num_envs: int,
    ) -> dict[str, Any]:
        """Return raw observation data for the current state.

        The returned dict will be post-processed by ``GenesisEnv._wrap_obs``
        into the canonical RLinf format (``main_images``, ``states``,
        ``task_descriptions``, ...).

        Expected keys:
        * ``"images"``: ``np.ndarray`` of shape ``(B, H, W, 3)`` uint8.
        * ``"states"``: ``torch.Tensor`` of shape ``(B, state_dim)`` float.

        Args:
            scene: The built Genesis scene.
            num_envs: Total number of parallel environments.
        """
