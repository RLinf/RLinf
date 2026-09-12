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

"""Tasks scored by where the tool is: reach a fixture pose and stay there.

:class:`CartesianTarget` is the whole task on its own, and the base of every
task set up around a fixture -- a hole, a bottle, a bin. It owns the pose the
reward measures against, the workspace around it, and what has to happen
between episodes. A task built on it adds only what it does before the arm
goes back to waiting: take a peg clear of its hole, let go of a cap.

A task asks an arm for what it wants, through the verbs every arm offers:
hold where you are, get clear of what you are touching, go back to where you
wait. How that is done belongs to the arm, which is why the same task runs on
one driven by tool poses and one driven by joint targets.

Poses in configs are ``[x, y, z, rx, ry, rz]`` with xyz Euler angles, in the
frame the arm reports ``tcp_pose`` in. Poses sent to the arm are ``xyz`` plus
an ``xyzw`` quaternion.
"""

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import cycle
from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.robotics.parts.arms.base import Home
from rlinf.utils.logging import get_logger

from .base import Evaluation, ResetContext, Task, TaskConfig
from .requirements import Needs, Parts, Reading
from .workspace import Workspace

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rlinf.envs.real.policy import Applied


def _pose_array(value: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    return None if value is None else np.asarray(value, dtype=np.float64)


@dataclass
class CartesianTargetConfig(TaskConfig):
    """Settings for :class:`CartesianTarget`."""

    target_ee_pose: Sequence[float] = (0.5, 0.0, 0.1, -3.14, 0.0, 0.0)
    """Fixture pose: what the reward measures against, and the centre of the
    orientation window."""

    reset_ee_pose: Optional[Sequence[float]] = (0.0,) * 6
    """Pose the arm rests at between episodes."""

    reward_threshold: Sequence[float] = (0.0,) * 6
    """Per-axis success tolerance. Only the position entries are scored."""

    ee_pose_limit_min: Optional[Sequence[float]] = (0.0,) * 6
    """Lowest pose a policy may command."""

    ee_pose_limit_max: Optional[Sequence[float]] = (0.0,) * 6
    """Highest pose a policy may command."""

    enable_random_reset: bool = False
    """Perturb the rest pose at the start of each episode."""

    random_xy_range: float = 0.0
    """Largest rest-position perturbation along x and y, in metres."""

    random_rz_range: float = 0.0
    """Largest rest-yaw perturbation around the target's, in radians."""

    joint_reset_qpos: Optional[Sequence[float]] = None
    """Joint configuration of the periodic joint reset. ``None`` never resets
    the joints."""

    joint_reset_cycle: int = 20000
    """Episodes between two joint resets."""

    reset_joint_qpos: Optional[Sequence[float]] = None
    """Joint configuration the arm waits at between episodes, for an arm that
    cannot be sent a tool pose. ``None`` leaves such an arm without a place to
    wait. Not to be confused with ``joint_reset_qpos`` above, which is the
    configuration the joints unwind to every ``joint_reset_cycle`` episodes."""

    random_joint_noise: float = 0.02
    """Largest perturbation of each rest joint when randomising, in radians,
    for an arm that waits at a configuration."""

    enable_gripper_penalty: bool = True
    """Charge a gripper change, so a policy does not chatter the gripper."""

    def __post_init__(self) -> None:
        self.target_ee_pose = _pose_array(self.target_ee_pose)
        self.reward_threshold = _pose_array(self.reward_threshold)
        for name in ("reset_ee_pose", "ee_pose_limit_min", "ee_pose_limit_max"):
            if getattr(self, name) is None:
                raise ValueError(f"{type(self).__name__} needs {name!r}.")
            setattr(self, name, _pose_array(getattr(self, name)))


@dataclass
class FixtureConfig(CartesianTargetConfig):
    """A target with its workspace and rest pose laid out around it.

    Left unset, the workspace reaches ``clip_*_range`` from the target on each
    axis and 0.01 rad in roll and pitch, and the arm rests
    ``clip_z_range_high`` above the target. Setting ``ee_pose_limit_*`` or
    ``reset_ee_pose`` outright replaces the derived value.
    """

    target_ee_pose: Sequence[float] = (0.0,) * 6
    reset_ee_pose: Optional[Sequence[float]] = None
    reward_threshold: Sequence[float] = (0.01, 0.01, 0.01, 0.2, 0.2, 0.2)
    ee_pose_limit_min: Optional[Sequence[float]] = None
    ee_pose_limit_max: Optional[Sequence[float]] = None
    enable_random_reset: bool = True

    clip_x_range: float = 0.05
    """Workspace half-width along x, in metres."""

    clip_y_range: float = 0.05
    """Workspace half-width along y, in metres."""

    clip_z_range_low: float = 0.0
    """How far below the target the workspace reaches, in metres."""

    clip_z_range_high: float = 0.1
    """How far above the target the workspace reaches, in metres."""

    clip_rz_range: float = np.pi / 6
    """Largest yaw away from the target's, in radians."""

    def __post_init__(self) -> None:
        target = np.asarray(self.target_ee_pose, dtype=np.float64)
        if self.ee_pose_limit_min is None:
            self.ee_pose_limit_min = target - np.array(
                [
                    self.clip_x_range,
                    self.clip_y_range,
                    self.clip_z_range_low,
                    0.01,
                    0.01,
                    self.clip_rz_range,
                ]
            )
        if self.ee_pose_limit_max is None:
            self.ee_pose_limit_max = target + np.array(
                [
                    self.clip_x_range,
                    self.clip_y_range,
                    self.clip_z_range_high,
                    0.01,
                    0.01,
                    self.clip_rz_range,
                ]
            )
        if self.reset_ee_pose is None:
            self.reset_ee_pose = target + np.array(
                [0.0, 0.0, self.clip_z_range_high, 0.0, 0.0, 0.0]
            )
        super().__post_init__()


def reach_target(
    tcp_pose: np.ndarray,
    target: np.ndarray,
    threshold: np.ndarray,
    *,
    dense: bool,
    gain: float = 500.0,
) -> Evaluation:
    """Score how close the tool's position is to a target's.

    The tool is in the zone when every position axis is within its threshold,
    which scores 1. Outside it, the sparse reward is 0 and the dense one
    ``exp(-gain * |d|^2)`` for the position error ``d``.
    """
    delta = np.abs(np.asarray(tcp_pose[:3], dtype=np.float64) - target[:3])
    if np.all(delta <= threshold[:3]):
        return Evaluation(reward=1.0, in_zone=True)
    reward = float(np.exp(-gain * np.sum(np.square(delta)))) if dense else 0.0
    return Evaluation(reward=reward, in_zone=False)


def release_and_back_off(parts: Parts, context: ResetContext) -> None:
    """Open the end effector and back away from what it held.

    The object gets 5 s to settle before the tool takes 3 cm of clearance,
    and 2 s more before it takes 2 cm more.
    """
    arm = parts.arm()
    context.action.release(parts)
    arm.hold()
    time.sleep(5)
    arm.clear(distance=0.03, rate_hz=context.rate_hz)
    time.sleep(2)
    arm.clear(distance=0.02, rate_hz=context.rate_hz)


class CartesianTarget(Task):
    """Bring the tool to a fixture pose and hold it there.

    The reward is 1 while the tool's position is within
    :attr:`CartesianTargetConfig.reward_threshold` of the target on every
    axis. Orientation is not scored; the workspace keeps it near the target's.
    """

    CONFIG = CartesianTargetConfig

    config: CartesianTargetConfig

    def __init__(self, config: Optional[CartesianTargetConfig] = None) -> None:
        super().__init__(config)
        self._logger = get_logger()
        self._resets = cycle(range(self.config.joint_reset_cycle))
        next(self._resets)

    def requirements(self) -> Mapping[str, Needs]:
        """An arm that reports its tool pose."""
        return {"arm": Needs(observes=frozenset({"tcp_pose"}))}

    @property
    def workspace(self) -> Workspace:
        """The configured pose limits, with the window centred on the target."""
        return Workspace(
            low=self.config.ee_pose_limit_min,
            high=self.config.ee_pose_limit_max,
            target_euler=self.config.target_ee_pose[3:],
        )

    def rest_pose(self) -> np.ndarray:
        """The configured rest pose, ``xyz`` plus an ``xyzw`` quaternion."""
        pose = self.config.reset_ee_pose
        return np.concatenate([pose[:3], R.from_euler("xyz", pose[3:]).as_quat()])

    def home(self, parts: Parts, context: ResetContext) -> None:
        """Go to where the arm waits, and let it settle."""
        parts.arm().go_home(self.request(context))
        time.sleep(1.0)

    def request(self, context: ResetContext, pose: Optional[np.ndarray] = None) -> Home:
        """Where the arm should wait, in both spellings a robot might take.

        An arm driven by tool poses uses the rest pose; one driven by joint
        targets uses the rest configuration. Randomising perturbs whichever
        of the two the arm reads.
        """
        target = self.rest_pose() if pose is None else np.array(pose, dtype=float)
        qpos = self.config.reset_joint_qpos
        if self.config.enable_random_reset:
            xy, rz = self.config.random_xy_range, self.config.random_rz_range
            target[:2] += context.rng.uniform(-xy, xy, 2)
            euler = self.config.target_ee_pose[3:].copy()
            euler[-1] += context.rng.uniform(-rz, rz)
            target[3:] = R.from_euler("xyz", euler).as_quat()
            if qpos is not None:
                noise = self.config.random_joint_noise
                jittered = np.asarray(qpos, dtype=float) + context.rng.uniform(
                    -noise, noise, size=len(qpos)
                )
                limits = context.action.joint_limits()
                if limits is not None:
                    jittered = np.clip(jittered, *limits)
                qpos = list(jittered)
        return Home(
            pose=target,
            qpos=qpos,
            rate_hz=context.rate_hz,
        )

    def reset(self, parts: Parts, context: ResetContext) -> None:
        """Return to rest."""
        self.go_to_rest(parts, context)

    def go_to_rest(
        self,
        parts: Parts,
        context: ResetContext,
        pose: Optional[np.ndarray] = None,
    ) -> None:
        """Bring the arm to rest for the next episode.

        Every ``joint_reset_cycle`` episodes, or when the reset's options ask
        for ``joint_reset``, the joints first unwind to ``joint_reset_qpos``.
        The arm then goes to where it waits, perturbed when randomising. A
        hand goes back to its resting pose, and any fault the motion latched
        is cleared.

        Args:
            parts: The bound parts.
            context: The env's generator, action layout, and rate.
            pose: Rest pose to use instead of :meth:`rest_pose`.
        """
        arm = parts.arm()
        self.reset_joints_if_due(parts, context)
        arm.go_home(self.request(context, pose))
        context.action.rest_end_effectors(parts)
        arm.clear_errors()

    def reset_joints_if_due(self, parts: Parts, context: ResetContext) -> None:
        """Return the joints to ``joint_reset_qpos`` when a joint reset is due.

        One is due every ``joint_reset_cycle`` episodes, and whenever the
        reset's options ask for ``joint_reset``.
        """
        due = bool(context.options.get("joint_reset", False))
        if next(self._resets) == 0:
            self._logger.info(
                "Number of resets reached %d, resetting joints to initial position.",
                self.config.joint_reset_cycle,
            )
            due = True
        if due:
            parts.arm().unwind(self.config.joint_reset_qpos)

    def evaluate(self, reading: Reading, applied: "Applied") -> Evaluation:
        """Score the tool's distance to the target."""
        return reach_target(
            reading.arm()["tcp_pose"],
            self.config.target_ee_pose,
            self.config.reward_threshold,
            dense=self.config.use_dense_reward,
        )
