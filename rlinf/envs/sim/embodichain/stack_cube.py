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

"""Panda Task Program execution through IsaacLab-style relative EEF commands."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.envs.types import ControllerAction
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnvCfg
    from embodichain.lab.sim.types import EnvObs


def rotation_vector(matrix: torch.Tensor) -> torch.Tensor:
    """Return the shortest rotation vector for batched rotation matrices."""
    vectors = Rotation.from_matrix(matrix.detach().cpu().numpy()).as_rotvec()
    return torch.as_tensor(vectors, dtype=matrix.dtype, device=matrix.device)


class StackCubeExpertEnv(EmbodiedEnv):
    """Execute planned joint targets through a shared relative EEF controller.

    Task Program supplies joint targets. They are converted to 7-D relative
    commands before execution and recording. Ordinary 7-D policy actions use
    the same scale-0.5 damped least-squares controller, updated each physics
    substep. Only actuator targets are written during execution.
    """

    def __init__(
        self, cfg: EmbodiedEnvCfg, *, protocol: str | None = None, **kwargs: Any
    ) -> None:
        self._eef_target = None
        self.last_relative_action = None
        self._relative_command = None
        self._stack_protocol = protocol
        self._stack_task_state = None
        super().__init__(cfg, **kwargs)
        self.arm_ids = self.robot.get_joint_ids("arm")
        self.hand_ids = self.robot.get_joint_ids("hand")
        self.sim.set_ground_plane_visibility(False)
        if protocol is not None:
            self._reset_stack_task_state()

    def _reset_stack_task_state(self) -> None:
        self._stack_stable_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._stack_terminated = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._stack_task_state = (
            torch.zeros_like(self._stack_terminated),
            torch.zeros_like(self._stack_terminated),
            {},
        )

    def reset(self, *args: Any, **kwargs: Any) -> tuple[EnvObs, dict[str, Any]]:
        """Reset the physical scene and discard the previous control target."""
        options = kwargs.get("options") or {}
        reset_ids = options.get("reset_ids")
        if reset_ids is not None and list(reset_ids) != list(range(self.num_envs)):
            raise NotImplementedError(
                "Stack Cube requires synchronized full-batch reset."
            )
        self._eef_target = None
        self.last_relative_action = None
        self._relative_command = None
        if self._stack_task_state is not None:
            self._reset_stack_task_state()
        return super().reset(*args, **kwargs)

    def _update_sim_state(self, **kwargs: Any) -> None:
        super()._update_sim_state(**kwargs)
        if self._stack_task_state is None:
            return
        metrics = (
            self.batch_stack_metrics()
            if self._stack_protocol == "isaaclab"
            else self.stack_metrics()
        )
        stable_frame = torch.as_tensor(
            metrics["stable_frame"], device=self.device, dtype=torch.bool
        ).reshape(self.num_envs)
        self._stack_stable_steps = torch.where(
            stable_frame, self._stack_stable_steps + 1, 0
        )
        stable = self._stack_stable_steps >= 20
        if self._stack_protocol == "isaaclab":
            active = ~self._stack_terminated
            success = metrics["baseline_success"] & active
            fail = metrics["failure"] & active
            if not self.cfg.ignore_terminations:
                self._stack_terminated |= success | fail
        else:
            success = stable
            fail = torch.zeros_like(success)
        self._stack_task_state = (
            success,
            fail,
            {"stack_metrics": metrics, "stable_success": stable},
        )

    def compute_task_state(
        self, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Return measured success, failure, and stability for the native step.

        State advances once after physics. Repeated info reads do not advance
        the twenty-step dwell. PPO reports the first success/drop transition;
        full-duration evaluation keeps physical stability as a separate metric.
        Expert collection retains the common Task Program completion behavior.
        """
        if self._stack_task_state is None:
            return super().compute_task_state(**kwargs)
        return self._stack_task_state

    def reset_request(
        self, request: dict[str, Any], *, initial_state: dict[str, Any] | None = None
    ) -> tuple[EnvObs, dict[str, Any]]:
        """Reset one environment to a frozen request with zero initial velocities.

        One unrecorded physics tick with lifted cubes clears cached contacts
        before placing the requested initial poses. It is initialization only.
        Saved initial state, when provided for replay, takes precedence over
        sampled object poses and joint offsets. This operation is for episode
        initialization only; policy execution never writes physical state.
        """
        if self.num_envs != 1:
            raise ValueError("A frozen request requires a single environment.")
        return self.reset_requests(
            [request],
            initial_states=[initial_state] if initial_state is not None else None,
        )

    def initial_state(self) -> dict:
        """Return exact reset matrices and joint state for replay provenance."""
        if self.num_envs != 1:
            raise ValueError("Initial-state capture requires a single environment.")
        return {
            "object_poses": {
                uid: self.sim.get_rigid_object(uid)
                .get_local_pose(to_matrix=True)[0]
                .cpu()
                .tolist()
                for uid in ("blue", "red", "green")
            },
            "qpos": self.robot.get_qpos()[0].cpu().tolist(),
            "qvel": self.robot.get_qvel()[0].cpu().tolist(),
            "object_linear_velocities": {
                uid: self.sim.get_rigid_object(uid).body_data.lin_vel[0].cpu().tolist()
                for uid in ("blue", "red", "green")
            },
            "object_angular_velocities": {
                uid: self.sim.get_rigid_object(uid).body_data.ang_vel[0].cpu().tolist()
                for uid in ("blue", "red", "green")
            },
            "velocity_contract": "zero_at_reset",
            "contact_cache_reset": "lift_0.1m_one_physics_tick_before_placement",
            "arm_joint_ids": list(self.arm_ids),
            "hand_joint_ids": list(self.hand_ids),
        }

    def reset_requests(
        self,
        requests: list[dict[str, Any]],
        *,
        initial_states: list[dict[str, Any]] | None = None,
    ) -> tuple[EnvObs, dict[str, Any]]:
        """Reset every scene to its request, clearing contacts synchronously.

        Partial resets are deliberately unsupported: the contact-cache flush
        advances the shared physics world. Resetting a subset this way would
        advance the other scenes outside their recorded control steps.
        """
        if len(requests) != self.num_envs:
            raise ValueError("A synchronized reset requires one request per scene.")
        if initial_states is not None and len(initial_states) != self.num_envs:
            raise ValueError("Saved initial states must match the scene count.")
        if any(request.get("visual") is not None for request in requests):
            raise ValueError("Visual OOD reset is not implemented.")
        self.reset(seed=int(requests[0]["seed"]))
        qpos = self.robot.get_qpos().clone()
        desired_poses = {}
        for uid in ("blue", "red", "green"):
            obj = self.sim.get_rigid_object(uid)
            pose = obj.get_local_pose(to_matrix=True).clone()
            if initial_states is not None:
                pose[:] = torch.as_tensor(
                    [state["object_poses"][uid] for state in initial_states],
                    device=self.device,
                )
            else:
                pose[:, :3, 3] = torch.as_tensor(
                    [request["objects"][uid]["position"] for request in requests],
                    device=self.device,
                )
                angles = np.asarray(
                    [request["objects"][uid]["yaw_rad"] for request in requests]
                ).reshape(-1, 1)
                pose[:, :3, :3] = torch.as_tensor(
                    Rotation.from_euler("z", angles).as_matrix(), device=self.device
                )
            desired_poses[uid] = pose
            lifted = pose.clone()
            lifted[:, 2, 3] += 0.1
            obj.set_local_pose(lifted)
            obj.clear_dynamics()
        # Separating contacts for one tick invalidates cached friction anchors.
        self.sim.update(self.physics_dt, 1, render_final_step=False)
        for uid, pose in desired_poses.items():
            obj = self.sim.get_rigid_object(uid)
            obj.set_local_pose(pose)
            obj.clear_dynamics()
        if initial_states is not None:
            qpos[:] = torch.as_tensor(
                [state["qpos"] for state in initial_states], device=self.device
            )
        else:
            qpos[:, self.arm_ids] += torch.as_tensor(
                [request["arm_joint_offsets"] for request in requests],
                device=self.device,
            )
        limits = self.robot.get_qpos_limits()
        if (
            not torch.isfinite(qpos).all()
            or torch.any(qpos < limits[..., 0])
            or torch.any(qpos > limits[..., 1])
        ):
            raise ValueError("Requested robot initial state violates joint limits.")
        self.robot.set_qpos(qpos, target=False)
        self.robot.set_qpos(qpos, target=True)
        self.robot.clear_dynamics()
        with self.sim.render_frame(capture_camera_images=True):
            observation = self.get_obs()
        return observation, {}

    def policy_observation(self, raw_obs: dict) -> dict:
        """Return two real RGB views and Euler EEF state without expert inputs."""
        result = {
            "states": torch.from_numpy(self.dataset_states()).to(self.device),
            "task_descriptions": [STACK_PROMPT] * self.num_envs,
        }
        for camera, key in (("front", "main_images"), ("wrist", "wrist_images")):
            image = raw_obs["sensor"][camera]["color"][..., :3]
            if image.dtype != torch.uint8:
                image = (
                    (image * (255 if image.max() <= 1 else 1))
                    .clamp(0, 255)
                    .to(torch.uint8)
                )
            result[key] = image.clone()
        return result

    def step_relative(
        self, action: torch.Tensor
    ) -> tuple[EnvObs, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Execute a recorded 7-D EEF command through the expert controller."""
        command = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        if command.shape != (self.num_envs, 7) or not torch.isfinite(command).all():
            raise ValueError("Stack Cube actions must be finite [num_envs, 7] tensors.")
        self._relative_command = command
        try:
            return self.step(ControllerAction(self.robot.get_qpos().clone()))
        finally:
            self._relative_command = None

    def _step_action(self, action):
        current = self.robot.compute_fk(
            self.robot.get_qpos()[:, self.arm_ids], name="arm", to_matrix=True
        )
        if isinstance(action, ControllerAction) and self._relative_command is None:
            planned = action.value
            if not isinstance(planned, torch.Tensor):
                planned = planned["qpos"]
            target = self.robot.compute_fk(
                planned[:, self.arm_ids], name="arm", to_matrix=True
            )
            delta = rotation_vector(target[:, :3, :3] @ current[:, :3, :3].mT)
            gripper = torch.where(
                planned[:, self.hand_ids].mean(-1, keepdim=True) > 0.02,
                1.0,
                -1.0,
            )
            command = torch.cat(
                ((target[:, :3, 3] - current[:, :3, 3]) / 0.5, delta / 0.5, gripper),
                dim=-1,
            )
        else:
            command = self._relative_command
            if command is None:
                command = torch.as_tensor(
                    action, device=self.device, dtype=torch.float32
                )
            if command.shape != (self.num_envs, 7) or not torch.isfinite(command).all():
                raise ValueError(
                    "Stack Cube actions must be finite [num_envs, 7] tensors."
                )
            target = current.clone()
            target[:, :3, 3] += command[:, :3] * 0.5
            delta = Rotation.from_rotvec((command[:, 3:6] * 0.5).cpu().numpy())
            rotation = torch.as_tensor(
                delta.as_matrix(), device=self.device, dtype=current.dtype
            )
            target[:, :3, :3] = rotation @ current[:, :3, :3]
        self._eef_target = target
        self.last_relative_action = command.detach().clone()
        # The base expert buffer retains the original joint target for provenance.
        return action.value if isinstance(action, ControllerAction) else command

    def _advance_physics(self) -> None:
        if self._eef_target is None:
            return super()._advance_physics()
        for _ in range(self.cfg.sim_steps_per_control):
            current = self.robot.compute_fk(
                self.robot.get_qpos()[:, self.arm_ids], name="arm", to_matrix=True
            )
            qpos = self.robot.get_qpos().clone()
            error = torch.cat(
                (
                    self._eef_target[:, :3, 3] - current[:, :3, 3],
                    rotation_vector(
                        self._eef_target[:, :3, :3] @ current[:, :3, :3].mT
                    ),
                ),
                dim=-1,
            )
            jacobian = self.robot.compute_jacobian(
                qpos, end_link_name="panda_tcp", root_link_name="panda_link0"
            )
            identity = torch.eye(6, device=self.device, dtype=qpos.dtype)
            dq = jacobian.mT @ torch.linalg.solve(
                jacobian @ jacobian.mT + 0.01**2 * identity, error.unsqueeze(-1)
            )
            limits = self.robot.get_qpos_limits()[:, self.arm_ids]
            targets = (qpos[:, self.arm_ids] + dq.squeeze(-1)).clamp(
                limits[..., 0], limits[..., 1]
            )
            self.robot.set_qpos(targets, joint_ids=self.arm_ids)
            fingers = torch.where(self.last_relative_action[:, -1:] > 0, 0.04, 0.0)
            self.robot.set_qpos(
                fingers.expand(-1, len(self.hand_ids)), joint_ids=self.hand_ids
            )
            self.sim.update(self.physics_dt, 1, render_final_step=False)

    def dataset_state(self) -> np.ndarray:
        """Return dataset EEF XYZ, Euler XYZ, and signed finger positions.

        The observation point is 0.1034 m from panda_hand, whereas the control
        TCP is at 0.107 m. LeRobot metadata specifies Euler state channels;
        actions use rotation-vector increments.
        """
        return self.dataset_states()[0]

    def dataset_states(self) -> np.ndarray:
        """Return one eight-dimensional EEF observation for each environment."""
        poses = self.robot.get_link_pose("panda_hand", to_matrix=True).cpu().numpy()
        positions = poses[:, :3, 3] + poses[:, :3, :3] @ np.array([0.0, 0.0, 0.1034])
        eulers = Rotation.from_matrix(poses[:, :3, :3]).as_euler("xyz")
        fingers = self.robot.get_qpos()[:, self.hand_ids].cpu().numpy() * [1.0, -1.0]
        return np.concatenate((positions, eulers, fingers), axis=-1).astype(np.float32)

    def stack_metrics(self) -> dict:
        """Measure both stacking relations, release, and rigid-body motion."""
        if self.num_envs != 1:
            raise ValueError("Use batch_stack_metrics() for multiple scenes.")
        return {
            key: value[0].cpu().tolist()
            for key, value in self.batch_stack_metrics().items()
            if key != "failure"
        }

    def batch_stack_metrics(self) -> dict[str, torch.Tensor]:
        """Return per-scene completion and stability without expert state."""
        from rlinf.envs.sim.embodichain.stack_cube_metrics import stack_cube_metrics

        objects = [self.sim.get_rigid_object(uid) for uid in ("blue", "red", "green")]
        return stack_cube_metrics(
            torch.stack(
                [obj.get_local_pose(to_matrix=True)[:, :3, 3] for obj in objects], dim=1
            ),
            self.robot.get_qpos()[:, self.hand_ids],
            torch.stack([obj.body_data.lin_vel.norm(dim=-1) for obj in objects], dim=1),
            torch.stack([obj.body_data.ang_vel.norm(dim=-1) for obj in objects], dim=1),
        )


def evaluate_stack(
    positions: torch.Tensor,
    fingers: torch.Tensor,
    linear: list[float],
    angular: list[float],
) -> dict:
    """Score blue/red/green positions and physical motion without runtime state."""
    from rlinf.envs.sim.embodichain.stack_cube_metrics import stack_cube_metrics

    metrics = stack_cube_metrics(
        positions.unsqueeze(0),
        fingers.unsqueeze(0),
        positions.new_tensor([linear]),
        positions.new_tensor([angular]),
    )
    return {
        key: value[0].cpu().tolist()
        for key, value in metrics.items()
        if key != "failure"
    }


STACK_PROMPT = (
    "Stack the red block on the blue block, then stack the green block on the red block"
)


def build_stack_cube_env(
    panda_urdf: str | Path,
    *,
    seed: int = 0,
    device: str = "cuda:0",
    max_episode_steps: int = 450,
    num_envs: int = 1,
    protocol: str | None = None,
    gym_config_path: str | Path | None = None,
    ignore_terminations: bool = False,
) -> StackCubeExpertEnv:
    """Build scenes with the verified controller and prepared Panda asset.

    The caller owns simulator destruction. The asset must contain the control
    TCP and independently actuated finger joints used by expert collection.
    ``protocol`` selects baseline PPO success or stable evaluation success.
    ``gym_config_path`` selects a native task deployment, including its reward
    terms. EmbodiChain owns reward computation and manager reset. Leaving
    ``protocol`` unset preserves expert collection without policy episode scoring.
    """
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg, get_manager_modules
    from embodichain.lab.gym.utils.registration import (
        discover_task_packages,
        execute_init_hooks,
        get_env_spec,
    )
    from embodichain.utils.utility import load_config

    if protocol not in {None, "isaaclab", "evaluation"}:
        raise ValueError("Stack Cube protocol must be isaaclab or evaluation.")

    asset = Path(panda_urdf).resolve(strict=True)
    robot_xml = ET.parse(asset).getroot()
    if robot_xml.find("link[@name='panda_tcp']") is None:
        raise ValueError("Prepared Panda asset must include panda_tcp.")
    for name in ("panda_finger_joint1", "panda_finger_joint2"):
        joint = robot_xml.find(f"joint[@name='{name}']")
        if joint is None or joint.find("mimic") is not None:
            raise ValueError(
                "Prepared Panda requires independently actuated finger joints."
            )
    torch.set_float32_matmul_precision("highest")
    task = (
        Path(gym_config_path).expanduser().resolve(strict=True)
        if gym_config_path is not None
        else Path(__file__).parent
        / "configs/tasks/manipulation/stack_cube/task.panda.yaml"
    )
    discover_task_packages()
    execute_init_hooks()
    raw = load_config(task)
    cfg = config_to_cfg(
        raw, manager_modules=get_manager_modules(), source_path=str(task)
    )
    cfg.robot.fpath = str(asset)
    if num_envs < 1:
        raise ValueError("num_envs must be positive.")
    cfg.num_envs = num_envs
    cfg.max_episode_steps = max_episode_steps
    cfg.sim_cfg.headless = True
    cfg.sim_cfg.device = device
    cfg.sim_cfg.gpu_id = 0
    cfg.filter_dataset_saving = True
    cfg.seed = seed
    if protocol is not None:
        cfg.ignore_terminations = protocol == "evaluation" or ignore_terminations
    factory = get_env_spec(raw["id"]).task_program_adapter_factory
    env = StackCubeExpertEnv(
        cfg, protocol=protocol, task_program_adapter_factory=factory
    )
    env._replay_no_auto_reset = True
    return env
