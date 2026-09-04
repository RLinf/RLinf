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

"""Fake SDKs for the remaining robots: GimArm, Turtle2 and DOSW1."""

from __future__ import annotations

import types
from typing import Any

import numpy as np

from ._fakes import module

ARM_DOF = 6


def pinocchio() -> types.ModuleType:
    """Return the subset of Pinocchio needed for kinematics tests."""

    class Transform:
        def __init__(self):
            self.translation = np.array([0.3, 0.0, 0.4])
            self.rotation = np.eye(3)

    class Model:
        nv = ARM_DOF
        nframes = 3

        def __init__(self):
            self.frames = [
                types.SimpleNamespace(name=name)
                for name in ("universe", "link6", "gripper_center")
            ]

        def getFrameId(self, name):  # noqa: N802 - the vendor spells it this way
            for index, frame in enumerate(self.frames):
                if frame.name == name:
                    return index
            return self.nframes - 1

    class Data:
        def __init__(self):
            self.oMf = [Transform() for _ in range(Model.nframes)]

    class Quaternion:
        def __init__(self, _rotation):
            pass

        def coeffs(self):
            return np.array([0.0, 0.0, 0.0, 1.0])

    return module(
        "pinocchio",
        neutral=lambda model: np.zeros(model.nv),
        forwardKinematics=lambda *_a: None,
        updateFramePlacement=lambda *_a: None,
        computeFrameJacobian=lambda *_a, **_k: np.zeros((6, ARM_DOF)),
        Quaternion=Quaternion,
        LOCAL_WORLD_ALIGNED="LOCAL_WORLD_ALIGNED",
        Model=Model,
        Data=Data,
        _fake_model=Model,
        _fake_data=Data,
    )


def gim_arm() -> dict[str, types.ModuleType]:
    """Return a ``gim_arm_control`` module that reports a resting arm."""
    pin = pinocchio()

    class Reading:
        def __init__(self):
            self.position = [0.0] * ARM_DOF
            self.velocity = [0.0] * ARM_DOF
            self.torque = [0.0] * ARM_DOF
            # From the momentum observer, when the SDK provides one.
            self.external_torque = [0.0] * ARM_DOF
            self.gripper_position = 0.0
            self.has_fault = False

    class GimArmController:
        # The travel the driver normalises a gripper command against.
        gripper_open_position = 0.0
        gripper_closed_position = 1.0

        def __init__(self, config):
            self.config = config
            self.started = False
            self.mode = None
            self.targets: list[Any] = []

        def start(self, return_to_zero=False):
            self.started = True
            return True

        def stop(self):
            self.started = False

        def set_mode(self, mode):
            self.mode = mode

        def get_dof(self):
            return ARM_DOF

        def get_reading(self):
            return Reading() if self.started else None

        def set_feedforward_target(self, target, dq, ddq):
            self.targets.append((target, dq, ddq))

        def set_gripper(self, position):
            self.targets.append(("gripper", position))

    class ButterworthFilter:
        def __init__(self, _cutoff, _dt, dof):
            self._dof = dof

        def process(self, value):
            """Pass the signal through: the control loop calls this each tick."""
            return np.asarray(value, dtype=np.float64)

        __call__ = process
        filter = process

    control = module(
        "gim_arm_control",
        GimArmController=GimArmController,
        ButterworthFilter=ButterworthFilter,
        ControllerConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
        # The modes GimArmRobotConfig.control_mode documents, keyed the way
        # the driver looks them up: ControlMode[mode.upper()].
        ControlMode={
            name: name
            for name in (
                "IDLE",
                "GRAVITY_COMP",
                "MOMENTUM_OBSERVER",
                "POSITION",
                "TORQUE",
            )
        },
    )
    loader = module(
        "gim_arm_control.utils.urdf_loader",
        get_urdf_path=lambda *_a, **_k: "/tmp/gim_arm.urdf",
        load_arm6_model=lambda _path: (pin.Model(), pin.Data()),
    )
    utils = module("gim_arm_control.utils")
    utils.urdf_loader = loader
    control.utils = utils
    return {
        "pinocchio": pin,
        "gim_arm_control": control,
        "gim_arm_control.utils": utils,
        "gim_arm_control.utils.urdf_loader": loader,
    }


def turtle2() -> dict[str, types.ModuleType]:
    """Return a ``turtle2_basic`` controller for a resting dual-arm robot."""

    class Cameras:
        def check_cam1(self, timeout=0.5):
            return True

        def check_cam2(self, timeout=0.5):
            return True

        def check_cam3(self, timeout=0.5):
            return True

        def get_cam1_data(self):
            return np.zeros((48, 64, 3), dtype=np.uint8)

        get_cam2_data = get_cam1_data
        get_cam3_data = get_cam1_data

    REST = [0.3, 0.0, 0.2, 0.0, 1.0, 0.0, 0.0]

    class Turtle2Controller:
        """Record arm targets and expose them as the current state."""

        def __init__(self, *_args, **_kwargs):
            self.cam = Cameras()
            self.commanded: list[tuple] = []
            self._pose = [list(REST), list(REST)]

        def chassis_set_current_pose_as_virtual_zero(self):
            return True

        def arms_data(self):
            return [list(pose) for pose in self._pose]

        def arms_joint_data(self):
            return [[0.0] * 7] * 2

        def arms_cur_data(self):
            return [[0.0] * 7] * 2

        def head_data(self):
            return [0.0, 0.0]

        def lift_data(self):
            return 0.0

        def chassis_pose_data(self):
            return [0.0, 0.0, 0.0]

        def arms_control(self, left, right):
            self.commanded.append((list(left), list(right)))
            self._pose = [list(left), list(right)]

        def reset_arms(self):
            self._pose = [list(REST), list(REST)]
            return True

    controller = module(
        "turtle2_basic.turtle2_controller.Turtle2Controller",
        Turtle2Controller=Turtle2Controller,
    )
    package = module("turtle2_basic")
    inner = module("turtle2_basic.turtle2_controller")
    inner.Turtle2Controller = controller
    package.turtle2_controller = inner
    return {
        "turtle2_basic": package,
        "turtle2_basic.turtle2_controller": inner,
        "turtle2_basic.turtle2_controller.Turtle2Controller": controller,
    }


def airbot() -> dict[str, types.ModuleType]:
    """Return an ``airbot_sdk`` module that reports both arms at rest."""

    JOINTS = 7

    class Arm:
        """Represent one arm and record its commands."""

        def __init__(self):
            self.commanded: list[Any] = []

        def set_target_joint_q(self, *args, **_kwargs):
            self.commanded.append(args)

        def set_target_end(self, *args, **_kwargs):
            self.commanded.append(("end", args))

        def get_current_joint_q(self):
            return [0.0] * JOINTS

        def get_current_end(self):
            return 0.0

    class AirbotRobot:
        """Report both arms at rest with seven joints each."""

        def __init__(self, *_args, **_kwargs):
            self.running = True
            self.use_lead_arms = False
            self.config_ = types.SimpleNamespace()
            self.commanded: list[Any] = []
            self.left_arm = Arm()
            self.right_arm = Arm()
            self.left_lead_arm = Arm()
            self.right_lead_arm = Arm()

        def left_get_joint(self):
            return [0.0] * JOINTS

        right_get_joint = left_get_joint
        lead_left_get_joint = left_get_joint
        lead_right_get_joint = left_get_joint

        def left_get_end(self):
            return 0.0

        right_get_end = left_get_end

        def left_get_pose(self):
            # xyz + euler: DOSW1Arm hands this straight back as tcp_pose, and
            # declares it six wide.
            return [0.3, 0.0, 0.2, 0.0, 0.0, 0.0]

        right_get_pose = left_get_pose

        def left_go_joint(self, joint, gripper, interp=None):
            """Command one arm; a real one moves, so the readings follow."""
            self.commanded.append(("left", list(joint), gripper))

        def right_go_joint(self, joint, gripper, interp=None):
            self.commanded.append(("right", list(joint), gripper))

        def fk(self, joint, *_args, **_kwargs):
            """Forward kinematics: a pose for a joint vector."""
            return [0.3, 0.0, 0.2, 0.0, 1.0, 0.0, 0.0]

        def shutdown(self):
            self.running = False

        def close(self):
            self.running = False

    robot = module("airbot_sdk.Airbot", AirbotRobot=AirbotRobot)
    config = module(
        "airbot_sdk.configs.config",
        DosW1Config=lambda **kwargs: types.SimpleNamespace(**kwargs),
    )
    configs = module("airbot_sdk.configs")
    configs.config = config
    package = module("airbot_sdk")
    package.Airbot = robot
    package.configs = configs
    return {
        "airbot_sdk": package,
        "airbot_sdk.Airbot": robot,
        "airbot_sdk.configs": configs,
        "airbot_sdk.configs.config": config,
    }


def modules(**_: Any) -> dict[str, types.ModuleType]:
    """Return the remaining fake robot SDKs keyed by import name."""
    made: dict[str, types.ModuleType] = {}
    made.update(gim_arm())
    made.update(turtle2())
    made.update(airbot())
    return made
