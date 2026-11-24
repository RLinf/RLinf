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

"""Task-object state of a LIBERO environment, used to compare ART states."""

import math

import numpy as np

ART_PHASES = ("free", "contact", "grasped")
# Physical size of one unit along each ART cluster coordinate.
ART_POSITION_UNIT_M = 0.01
ART_ANGLE_UNIT_RAD = math.radians(10)
ART_TIME_UNIT_STEPS = 20


def _target_object(env):
    """Return the first object of interest with a free joint."""
    core = env.env
    for candidate in core.obj_of_interest:
        obj = core.objects_dict.get(candidate)
        if obj is not None and obj.joints:
            joint = obj.joints[0]
            if np.asarray(core.sim.data.get_joint_qpos(joint)).shape == (7,):
                return candidate, obj, joint
    raise ValueError(f"No movable free-joint object in {core.obj_of_interest}.")


def object_state(env) -> dict:
    """Read the target object, its goal, the gripper, and the contact phase.

    Runs in the environment process. The goal is the first other object of
    interest; a task without one uses the target position. Positions are world
    coordinates in meters and the quaternion is MuJoCo ``wxyz``.
    """
    core = env.env
    name, obj, joint = _target_object(env)
    qpos = core.sim.data.get_joint_qpos(joint)
    goal_position = qpos[:3]
    goal = next((other for other in core.obj_of_interest if other != name), None)
    if goal is not None:
        model = core.objects_dict.get(goal, core.fixtures_dict.get(goal))
        if model is None:
            raise KeyError(f"Goal {goal!r} is neither a LIBERO object nor a fixture.")
        goal_position = core.sim.data.body_xpos[
            core.sim.model.body_name2id(model.root_body)
        ]
    gripper = core.robots[0].gripper
    eef_site = core.robots[0].eef_site_id
    grasped = bool(core._check_grasp(gripper, obj))
    contact = bool(core.check_contact(gripper, obj))
    return {
        "object_position": qpos[:3].copy(),
        "object_quaternion": qpos[3:].copy(),
        "goal_position": np.array(goal_position),
        "eef_position": core.sim.data.site_xpos[eef_site].copy(),
        "eef_rotation": core.sim.data.site_xmat[eef_site].reshape(3, 3).copy(),
        "phase": "grasped" if grasped else "contact" if contact else "free",
    }


def cluster_features(
    state: dict, gripper_qpos: np.ndarray, remaining_steps: float
) -> np.ndarray:
    """Map an ``object_state`` to ART cluster coordinates.

    One unit is 1 cm for the gripper-to-object and object-to-goal offsets and
    the finger opening, approximately 10 degrees for either full orientation, and 20
    environment steps for the remaining episode budget. Positions are relative,
    so different initial layouts in the same configuration coincide.
    """
    quaternion = np.asarray(state["object_quaternion"], dtype=float)
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    object_rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    # Chordal distance is 2*sin(relative_angle/2)/angle_unit, without angle wraps.
    rotation_scale = math.sqrt(2) * ART_ANGLE_UNIT_RAD
    return np.concatenate(
        [
            (state["eef_position"] - state["object_position"]) / ART_POSITION_UNIT_M,
            (state["object_position"] - state["goal_position"]) / ART_POSITION_UNIT_M,
            object_rotation.ravel() / rotation_scale,
            np.asarray(state["eef_rotation"]).ravel() / rotation_scale,
            [(gripper_qpos[0] - gripper_qpos[1]) / ART_POSITION_UNIT_M],
            [remaining_steps / ART_TIME_UNIT_STEPS],
        ]
    )
