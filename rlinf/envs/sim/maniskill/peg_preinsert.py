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

"""Initialize a held peg near the hole with GPU inverse kinematics."""

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PegPreinsertConfig:
    """Randomization in hole axes; gap is measured from the front box surface."""

    gap_m: float = 0.05
    lateral_jitter_m: float = 0.03
    rotation_jitter_deg: float = 3.0

    def __post_init__(self) -> None:
        for name in ("gap_m", "lateral_jitter_m", "rotation_jitter_deg"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Expected finite nonnegative {name}, got {value}.")
        if self.gap_m <= 0 or self.rotation_jitter_deg > 10:
            raise ValueError("Expected positive gap_m and rotation_jitter_deg <= 10.")


def preinsert_kinematics(task: Any) -> Any:
    """Return GPU TCP kinematics owned by the current arm controller."""
    from mani_skill.agents.controllers.utils.kinematics import Kinematics

    if not task.gpu_sim_enabled or task.device.type != "cuda":
        raise ValueError("Preinsert initialization requires GPU simulation and IK.")
    if task.control_mode not in {"pd_ee_delta_pose", "pd_joint_delta_pos"}:
        raise ValueError(f"Unsupported preinsert control mode: {task.control_mode}.")
    arm = task.agent.controller.controllers["arm"]
    if task.control_mode == "pd_ee_delta_pose":
        kinematics = arm.kinematics
    else:
        # A reconfigured articulation owns a new controller and a new solver.
        if not hasattr(arm, "_preinsert_kinematics"):
            arm._preinsert_kinematics = Kinematics(
                task.agent.urdf_path,
                task.agent.ee_link_name,
                task.agent.robot,
                arm.active_joint_indices,
            )
        kinematics = arm._preinsert_kinematics
    if not kinematics.use_gpu_ik:
        raise ValueError("Expected the GPU preinsert kinematics solver.")
    return kinematics


def validate_joint_gpu_controller(env: Any) -> dict[str, Any]:
    """Validate normalized Panda joint deltas and GPU preinsert execution."""
    task = env.unwrapped
    if task.control_mode != "pd_joint_delta_pos":
        raise ValueError(f"Expected pd_joint_delta_pos, got {task.control_mode}.")
    preinsert_kinematics(task)
    cfg = task.agent.controller.controllers["arm"].config
    if (
        not cfg.use_delta
        or cfg.use_target
        or not cfg.normalize_action
        or cfg.interpolate
        or cfg.lower != -0.1
        or cfg.upper != 0.1
    ):
        raise ValueError(
            "Expected normalized joint deltas with +/-0.1 rad bounds, no target accumulation or interpolation."
        )
    return {
        "sim_backend": "gpu",
        "ik_backend": "gpu_pytorch_kinematics_lm",
        "control_mode": task.control_mode,
        "joint_delta_bounds_rad": [cfg.lower, cfg.upper],
        "use_target": cfg.use_target,
        "interpolate": cfg.interpolate,
    }


def initialize_peg_preinsert(
    task: Any, env_idx: torch.Tensor, config: PegPreinsertConfig
) -> None:
    """Set selected environments to a closed-gripper preinsert pose without stepping."""
    from mani_skill.utils.geometry.rotation_conversions import (
        euler_angles_to_matrix,
        matrix_to_quaternion,
    )
    from mani_skill.utils.structs.pose import Pose

    kinematics = preinsert_kinematics(task)
    batch = len(env_idx)
    half_sizes = task.peg_half_sizes[env_idx]
    head_position = torch.zeros((batch, 3), device=task.device)
    head_position[:, 0] = -half_sizes[:, 0] - config.gap_m
    head_position[:, 1:] = (
        torch.rand((batch, 2), device=task.device) * 2 - 1
    ) * config.lateral_jitter_m
    angles = (torch.rand((batch, 3), device=task.device) * 2 - 1) * math.radians(
        config.rotation_jitter_deg
    )
    initial_qpos = task.agent.robot.qpos[env_idx].clone()
    limits = task.agent.robot.get_qlimits()[env_idx, :7]
    grasp_offset = torch.zeros_like(head_position)
    grasp_offset[:, 0] = -0.6 * half_sizes[:, 0]
    head_offset = Pose.create_from_pq(
        head_position, matrix_to_quaternion(euler_angles_to_matrix(angles, "XYZ"))
    )
    root_inv = Pose.create(task.agent.robot.pose.raw_pose[env_idx]).inv()
    for scene_attempt in range(8):
        head = Pose.create(task.box_hole_pose.raw_pose[env_idx]) * head_offset
        peg = head * Pose.create(task.peg_head_offsets.raw_pose[env_idx]).inv()
        tcp = peg * Pose.create_from_pq(grasp_offset, [0, 1, 0, 0])
        qpos, converged = _solve_preinsert_ik(
            kinematics, root_inv * tcp, initial_qpos.clone(), limits
        )
        if converged.all():
            break
        if scene_attempt == 7:
            raise RuntimeError(
                "Preinsert GPU IK found no reachable cabinet pose after 8 attempts "
                f"for environment indices {env_idx[~converged].tolist()}."
            )
        # Keep tip offsets fixed; reject only unreachable cabinet placements.
        box = task.box.pose.raw_pose[env_idx].clone()
        samples = torch.rand((int((~converged).sum()), 3), device=task.device)
        box[~converged, :2] = samples[:, :2] * box.new_tensor(
            [0.1, 0.2]
        ) + box.new_tensor([-0.05, 0.2])
        yaw = math.pi / 2 + (samples[:, 2] * 2 - 1) * math.pi / 8
        box[~converged, 3] = torch.cos(yaw / 2)
        box[~converged, 4:6] = 0
        box[~converged, 6] = torch.sin(yaw / 2)
        task.box.set_pose(Pose.create(box))
    qpos[:, -2:] = half_sizes[:, 1:2] - 0.0003
    task.agent.robot.set_qpos(qpos)
    task.agent.robot.set_qvel(torch.zeros_like(qpos))
    task.peg.set_pose(peg)
    task.peg.set_linear_velocity(torch.zeros_like(head_position))
    task.peg.set_angular_velocity(torch.zeros_like(head_position))


def _solve_preinsert_ik(
    kinematics: Any, target: Any, qpos: torch.Tensor, limits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return GPU joint solutions and a validity mask without changing the scene."""
    from mani_skill.utils.structs.pose import Pose

    batch = qpos.shape[0]
    initial_qpos = qpos.clone()
    desired = target.to_transformation_matrix()
    converged = torch.zeros(batch, dtype=torch.bool, device=qpos.device)
    # Joint-seed retries leave both the scene distribution and episode RNG intact.
    generator = torch.Generator(device=qpos.device).manual_seed(0)
    for retries in (1, 32):
        candidates = initial_qpos[:, None].expand(-1, retries, -1).clone()
        if retries > 1:
            samples = torch.rand(
                (batch, retries, 7), device=qpos.device, generator=generator
            )
            candidates[:, :, :7] = limits[:, None, :, 0] + samples * (
                limits[:, None, :, 1] - limits[:, None, :, 0]
            )
        candidate = candidates.reshape(-1, qpos.shape[-1])
        retry_limits = limits.repeat_interleave(retries, dim=0)
        retry_target = Pose.create(target.raw_pose.repeat_interleave(retries, dim=0))
        # Bound each LM iteration so a distant target cannot jump across limits.
        for _ in range(100):
            solved = kinematics.compute_ik(
                retry_target,
                candidate,
                solver_config={"type": "levenberg_marquardt", "alpha": 0.5},
            )
            candidate[:, :7] += (solved - candidate[:, :7]).clamp(-0.2, 0.2)
            candidate[:, :7] = candidate[:, :7].clamp(
                retry_limits[..., 0], retry_limits[..., 1]
            )
        actual = kinematics.pk_chain.forward_kinematics(candidate[:, :7]).get_matrix()
        actual = actual.reshape(batch, retries, 4, 4)
        position_error = (actual[:, :, :3, 3] - desired[:, None, :3, 3]).norm(dim=-1)
        rotation_error = (
            (actual[:, :, :3, :3] - desired[:, None, :3, :3]).abs().amax(dim=(-1, -2))
        )
        valid = (position_error <= 1e-4) & (rotation_error <= 1e-3)
        distance = (candidates[:, :, :7] - initial_qpos[:, None, :7]).norm(dim=-1)
        best = distance.masked_fill(~valid, torch.inf).argmin(dim=1)
        solved = valid.any(dim=1)
        selected = candidates[torch.arange(batch, device=qpos.device), best]
        qpos[solved & ~converged] = selected[solved & ~converged]
        converged |= solved
        if converged.all():
            break
    return qpos, converged
