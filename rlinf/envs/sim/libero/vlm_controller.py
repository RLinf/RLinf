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

"""Relative LIBERO macro-action controller for a VLM policy.

One model action is a relative Cartesian/gripper command. The controller turns
it into robosuite OSC steps and stops when the target is reached, the arm is
blocked, the task ends, or the native-step budget expires. The motion law is
PD offsets, OSC decoupling, and the Panda gripper command.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from rlinf.models.embodiment.qwen_vlm.protocol import LIBERO_SCHEMA

SCHEMA = LIBERO_SCHEMA
DECIMAL_N = {"x": 1, "y": 1, "z": 1, "roll": 2, "pitch": 2, "yaw": 2, "width": 1}
POS_DELTA_M = 0.05
ROT_DELTA_RAD = 0.5
GRIP_RATE = 0.25
GRIP_SPAN_CM = 8.0
LEAD_S = 0.04
SPAN_M = 0.015
SPAN_RAD = 0.15
DECOUPLE_DAMPING = 0.1
DECOUPLE_MAX = 0.5
DECOUPLE_JOINT_MARGIN_RAD = 0.2
DECOUPLE_ROT_ERR_RAD = 0.1
POS_TOL_M = 0.003
ROT_TOL_RAD = 0.02
REST_MPS = 0.03
REST_RADPS = 0.15
GRIP_STILL_M = 0.0005
GRIP_TOL_CM = 0.5
PROGRESS_M = 0.001
PROGRESS_RAD = 0.005
BLOCKED_STEP_N = 2
STALL_STEP_N = 6
MAX_ACTION_STEP_N = 100


def vector_to_action(row: np.ndarray) -> dict[str, dict[str, float]] | None:
    """Return a LIBERO action, or None when the trailing validity flag is off."""
    values = np.asarray(row, dtype=np.float64).reshape(-1)
    if values.shape[0] != 8 or not np.isfinite(values).all() or values[-1] < 0.5:
        return None
    arm = values[:6]
    return {
        "arm": {name: float(arm[index]) for index, name in enumerate(SCHEMA["arm"])},
        "gripper": {"width": float(values[6])},
    }


def apply_deltas(
    measured: dict[str, float], action: dict[str, dict[str, float]]
) -> dict[str, float]:
    """Add centimeter/radian deltas to the measured arm quantities."""
    targets = dict(measured)
    for name, change in action["arm"].items():
        targets[name] = float(targets[name]) + float(change)
    return targets


def gripper_command(width_delta: float) -> float | None:
    """Map a width delta to the fully-open, fully-closed, or hold target.

    Positive opens (``+1``), negative closes (``-1``), and zero holds.
    """
    if float(width_delta) == 0.0:
        return None
    return float(np.sign(width_delta))


def native_grip_direction(
    opening: float, opening_target: float | None, grip_rate: float = GRIP_RATE
) -> float:
    """Return the Panda command that walks the gripper opening toward its target.

    The command is ``1`` while the opening is above the
    target and ``-1`` while it is below. A zero command holds the gripper.
    """
    if opening_target is None:
        return 0.0
    if opening > opening_target + grip_rate / 2:
        return 1.0
    if opening < opening_target - grip_rate / 2:
        return -1.0
    return 0.0


def pd_offsets(
    position_error: np.ndarray,
    rotation_error: np.ndarray,
    linear_velocity: np.ndarray,
    angular_velocity: np.ndarray,
) -> np.ndarray:
    """Return clipped OSC offsets from the lead/PD law."""
    error = np.concatenate([position_error, rotation_error])
    twist = np.concatenate([linear_velocity, angular_velocity])
    span = np.array([SPAN_M] * 3 + [SPAN_RAD] * 3)
    return np.clip((error - LEAD_S * twist) / span, -1.0, 1.0)


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """Fixed-axis roll, pitch, yaw: Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = math.cos(rpy[0]), math.sin(rpy[0])
    cp, sp = math.cos(rpy[1]), math.sin(rpy[1])
    cy, sy = math.cos(rpy[2]), math.sin(rpy[2])
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def matrix_to_rpy(rot: np.ndarray) -> np.ndarray:
    """Invert :func:`rpy_to_matrix`, giving yaw the shared axis at gimbal lock."""
    sp = min(1.0, max(-1.0, -float(rot[2, 0])))
    if abs(sp) < 1.0 - 1e-9:
        return np.array(
            [
                math.atan2(rot[2, 1], rot[2, 2]),
                math.asin(sp),
                math.atan2(rot[1, 0], rot[0, 0]),
            ]
        )
    return np.array([0.0, math.asin(sp), math.atan2(-rot[0, 1], rot[1, 1])])


class ActionProgress:
    """Reached and blocked/stall stopping for one macro action."""

    def __init__(self) -> None:
        self.least_pos_err = math.inf
        self.least_rot_err = math.inf
        self.stall_step_n = 0
        self.slow_stall_step_n = 0
        self.is_stopping = False

    def update(
        self,
        *,
        pos_err: float,
        rot_err: float,
        is_slow: bool,
        fingers_done: bool,
        opening_reached: bool,
    ) -> str | None:
        """Return ``reached`` or ``stopped``, or None while the arm continues."""
        is_at_rest = is_slow and fingers_done and opening_reached
        arm_reached = pos_err < POS_TOL_M and rot_err < ROT_TOL_RAD
        if is_at_rest and (arm_reached or self.is_stopping):
            return "stopped" if self.is_stopping else "reached"
        if self.is_stopping:
            return None
        if (
            arm_reached
            or pos_err < self.least_pos_err - PROGRESS_M
            or rot_err < self.least_rot_err - PROGRESS_RAD
        ):
            self.least_pos_err = min(self.least_pos_err, pos_err)
            self.least_rot_err = min(self.least_rot_err, rot_err)
            self.stall_step_n = self.slow_stall_step_n = 0
        else:
            self.stall_step_n += 1
            self.slow_stall_step_n = self.slow_stall_step_n + 1 if is_slow else 0
        if not arm_reached and (
            self.slow_stall_step_n >= BLOCKED_STEP_N
            or self.stall_step_n >= STALL_STEP_N
        ):
            self.is_stopping = True
            if is_at_rest:
                return "stopped"
        return None


def _robosuite_env(env: Any) -> Any:
    seen: set[int] = set()
    current = env
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        # OffScreenRenderEnv exposes sim and robots, but _get_observations
        # lives on the wrapped robosuite env.
        if (
            hasattr(current, "_get_observations")
            and hasattr(current, "sim")
            and hasattr(current, "robots")
        ):
            return current
        current = getattr(current, "env", None)
    raise RuntimeError("VLM control requires the robosuite environment.")


def _round_state(values: dict[str, float]) -> dict[str, dict[str, float]]:
    return {
        part: {
            name: round(float(values[name]), DECIMAL_N[name]) + 0.0
            for name in quantities
        }
        for part, quantities in SCHEMA.items()
    }


def _base(scene: Any) -> tuple[np.ndarray, np.ndarray]:
    sim = scene.sim
    body = sim.model.body_name2id("robot0_base")
    return sim.data.body_xpos[body].copy(), sim.data.body_xmat[body].reshape(
        3, 3
    ).copy()


def _tool(observation: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
    return (
        np.asarray(observation["robot0_eef_pos"], dtype=np.float64),
        np.asarray(observation["robot0_eef_quat"], dtype=np.float64),
        float(
            observation["robot0_gripper_qpos"][0]
            - observation["robot0_gripper_qpos"][1]
        ),
    )


def measure_quantities(
    scene: Any, observation: dict[str, np.ndarray]
) -> dict[str, float]:
    """Read tool pose in the robot base frame and the finger gap."""
    import robosuite.utils.transform_utils as transform

    base_pos, base_rot = _base(scene)
    tool_pos, tool_quat, gap = _tool(observation)
    pos_cm = base_rot.T @ (tool_pos - base_pos) * 100.0
    rpy = matrix_to_rpy(base_rot.T @ transform.quat2mat(tool_quat))
    values = dict(zip(SCHEMA["arm"], [*pos_cm, *rpy]))
    values["width"] = gap * 100.0
    return values


def observe_state(env: Any) -> dict[str, dict[str, float]]:
    """Return the rounded arm and gripper state of one LIBERO environment."""
    scene = _robosuite_env(env)
    observation = scene._get_observations()
    return _round_state(measure_quantities(scene, observation))


class MotionController:
    """Closed-loop OSC execution of one relative VLM action."""

    def __init__(
        self, scene: Any, observation: dict[str, np.ndarray], action: dict[str, Any]
    ) -> None:
        import robosuite.utils.transform_utils as transform

        self.scene = scene
        self.gripper = scene.robots[0].gripper
        targets = measure_quantities(scene, observation)
        for quantity, change in action["arm"].items():
            targets[quantity] += float(change)
        base_pos, base_rot = _base(scene)
        self.pos_target = base_pos + base_rot @ (
            np.array([targets["x"], targets["y"], targets["z"]]) / 100.0
        )
        self.quat_target = transform.mat2quat(
            base_rot
            @ rpy_to_matrix(
                np.array([targets["roll"], targets["pitch"], targets["yaw"]])
            )
        )
        self.opening_target = gripper_command(action["gripper"]["width"])
        self.width_target = (
            None
            if self.opening_target is None
            else GRIP_SPAN_CM * (self.opening_target + 1.0) / 2.0
        )
        self.observation = observation
        self.pos, self.quat, self.gap = _tool(observation)
        self.vel, self.ang_vel = self._velocity()
        self.progress = ActionProgress()

    def move(
        self, budget: int
    ) -> tuple[dict[str, np.ndarray], list[list[float]], list[float], bool, str]:
        """Step until reached, stopped, done, budget, or the action cap."""
        actions: list[list[float]] = []
        rewards: list[float] = []
        steps = min(MAX_ACTION_STEP_N, max(int(budget), 0))
        for _ in range(steps):
            native = np.concatenate([self._arm_offsets(), [self._grip_direction()]])
            self.observation, reward, task_done, _info = self.scene.step(native)
            actions.append(native.tolist())
            rewards.append(float(reward))
            if task_done:
                return self.observation, actions, rewards, True, "done"
            outcome = self._step_outcome()
            if outcome is not None:
                return self.observation, actions, rewards, False, outcome
        done = len(actions) >= budget
        return (
            self.observation,
            actions,
            rewards,
            done,
            "budget" if done else "timeout",
        )

    def _arm_offsets(self) -> np.ndarray:
        import robosuite.utils.transform_utils as transform

        rotation = transform.quat_multiply(
            self.quat_target, transform.quat_inverse(self.quat)
        )
        if rotation[3] < 0.0:
            rotation = -rotation
        rot_err = transform.quat2axisangle(rotation)
        offsets = pd_offsets(
            self.pos_target - self.pos, rot_err, self.vel, self.ang_vel
        )
        if (
            np.linalg.norm(rot_err) > DECOUPLE_ROT_ERR_RAD
            or self._joint_margin() < DECOUPLE_JOINT_MARGIN_RAD
        ):
            return offsets
        return self._decouple(offsets)

    def _joint_margin(self) -> float:
        robot = self.scene.robots[0]
        sim = self.scene.sim
        joint_positions = sim.data.qpos[robot._ref_joint_pos_indexes]
        lower_limits, upper_limits = sim.model.jnt_range[robot._ref_joint_indexes].T
        return float(
            np.min(
                np.minimum(
                    joint_positions - lower_limits, upper_limits - joint_positions
                )
            )
        )

    def _decouple(self, offsets: np.ndarray) -> np.ndarray:
        from robosuite.utils.control_utils import opspace_matrices

        ctrl = self.scene.robots[0].controller
        if not getattr(ctrl, "uncoupling", False):
            return offsets
        lambda_full, lambda_pos, lambda_ori, _ = opspace_matrices(
            ctrl.mass_matrix, ctrl.J_full, ctrl.J_pos, ctrl.J_ori
        )
        block_inertia = np.zeros((6, 6))
        block_inertia[:3, :3], block_inertia[3:, 3:] = lambda_pos, lambda_ori
        scale = np.array([POS_DELTA_M] * 3 + [ROT_DELTA_RAD] * 3)
        coupling = np.linalg.solve(lambda_full, block_inertia) * (
            scale[None, :] / scale[:, None]
        )
        coupling_inverse = np.linalg.solve(
            coupling.T @ coupling + DECOUPLE_DAMPING**2 * np.eye(6), coupling.T
        )
        twist = np.concatenate([self.vel, self.ang_vel])
        decoupled = coupling_inverse @ offsets + float(ctrl.kd[0] / ctrl.kp[0]) * (
            np.eye(6) - coupling_inverse
        ) @ (twist / scale)
        correction = np.clip(decoupled - offsets, -DECOUPLE_MAX, DECOUPLE_MAX)
        return np.clip(offsets + correction, -1.0, 1.0)

    def _grip_direction(self) -> float:
        opening = (
            0.0
            if self.opening_target is None
            else float(self.gripper.current_action[0])
        )
        return native_grip_direction(opening, self.opening_target)

    def _velocity(self) -> tuple[np.ndarray, np.ndarray]:
        sim = self.scene.sim
        site = self.scene.robots[0].controller.eef_name
        return (
            np.array(sim.data.get_site_xvelp(site), dtype=np.float64),
            np.array(sim.data.get_site_xvelr(site), dtype=np.float64),
        )

    def _step_outcome(self) -> str | None:
        import robosuite.utils.transform_utils as transform

        previous_gap = self.gap
        self.pos, self.quat, self.gap = _tool(self.observation)
        self.vel, self.ang_vel = self._velocity()
        rotation = transform.quat_multiply(
            self.quat_target, transform.quat_inverse(self.quat)
        )
        if rotation[3] < 0.0:
            rotation = -rotation
        rot_err = float(np.linalg.norm(transform.quat2axisangle(rotation)))
        pos_err = float(np.linalg.norm(self.pos_target - self.pos))
        is_slow = (
            np.linalg.norm(self.vel) < REST_MPS
            and np.linalg.norm(self.ang_vel) < REST_RADPS
        )
        fingers_done = (
            self.width_target is None
            or abs(self.gap - previous_gap) < GRIP_STILL_M
            or abs(self.gap * 100.0 - self.width_target) < GRIP_TOL_CM
        )
        opening = float(self.gripper.current_action[0])
        opening_reached = (
            self.opening_target is None
            or abs(opening - self.opening_target) < GRIP_RATE / 2
        )
        was_stopping = self.progress.is_stopping
        outcome = self.progress.update(
            pos_err=pos_err,
            rot_err=rot_err,
            is_slow=is_slow,
            fingers_done=fingers_done,
            opening_reached=opening_reached,
        )
        if self.progress.is_stopping and not was_stopping:
            self.pos_target = self.pos.copy()
            self.quat_target = self.quat.copy()
        return outcome


def execute_action(
    env: Any,
    action: dict[str, dict[str, float]] | None,
    *,
    native_budget: int,
) -> dict[str, Any]:
    """Execute one macro action inside the simulator process."""
    if action is None:
        return {
            "observation": None,
            "reward": 0.0,
            "success": False,
            "done": True,
            "outcome": "parse",
            "native_steps": 0,
            "native_actions": [],
        }
    scene = _robosuite_env(env)
    observation = scene._get_observations()
    if int(native_budget) <= 0:
        return {
            "observation": observation,
            "reward": 0.0,
            "success": False,
            "done": True,
            "outcome": "budget",
            "native_steps": 0,
            "native_actions": [],
            "state": observe_state(env),
        }
    controller = MotionController(scene, observation, action)
    observation, actions, rewards, done, outcome = controller.move(int(native_budget))
    success = bool(done) and sum(rewards) > 0.0
    return {
        "observation": observation,
        "reward": 1.0 if success else 0.0,
        "success": success,
        "done": done,
        "outcome": outcome,
        "native_steps": len(actions),
        "native_actions": actions,
        "state": _round_state(measure_quantities(scene, observation)),
    }
