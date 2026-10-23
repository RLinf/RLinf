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
        # The modes GimArmEnvConfig.control_mode documents, keyed the way
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


def r2() -> dict[str, types.ModuleType]:
    """Return fake ``rframe`` and R2 ROS2 message/service modules."""
    waist_21 = (
        "J_WaistPitch_1",
        "J_WaistPitch_2",
        "J_WaistPitch_3",
        "J_WaistRoll",
        "J_WaistYaw",
    )
    waist_20 = (
        "J_WaistLift",
        "J_WaistPitch",
        "J_WaistRoll",
        "J_WaistYaw",
    )
    neck_and_arms = (
        "J_NeckYaw",
        "J_NeckPitch",
        "J_ArmL_1",
        "J_ArmL_2",
        "J_ArmL_3",
        "J_ArmL_4",
        "J_ArmL_5",
        "J_ArmL_6",
        "J_ArmL_7",
        "J_ArmR_1",
        "J_ArmR_2",
        "J_ArmR_3",
        "J_ArmR_4",
        "J_ArmR_5",
        "J_ArmR_6",
        "J_ArmR_7",
    )
    gripper_joints = ("GripperL", "GripperR")
    hand_joints = (
        "HandL_Little",
        "HandL_Ring",
        "HandL_Middle",
        "HandL_Index",
        "HandL_ThumbBend",
        "HandL_ThumbRot",
        "HandR_Little",
        "HandR_Ring",
        "HandR_Middle",
        "HandR_Index",
        "HandR_ThumbBend",
        "HandR_ThumbRot",
    )

    def _make_pose(x: float = 0.3, y: float = 0.0, z: float = 0.2) -> Any:
        return types.SimpleNamespace(
            position=types.SimpleNamespace(x=float(x), y=float(y), z=float(z)),
            orientation=types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )

    def _make_twist(
        vx: float = 0.0, vy: float = 0.0, vz: float = 0.0, wz: float = 0.0
    ) -> Any:
        return types.SimpleNamespace(
            linear=types.SimpleNamespace(x=float(vx), y=float(vy), z=float(vz)),
            angular=types.SimpleNamespace(x=0.0, y=0.0, z=float(wz)),
        )

    class TCP:
        def __init__(self) -> None:
            self.tcp_pose = [_make_pose(0.3, 0.2, 0.2), _make_pose(0.3, -0.2, 0.2)]
            self.tcp_twist = [_make_twist(), _make_twist()]

    class NaviState:
        def __init__(self) -> None:
            self.pose = _make_pose(0.0, 0.0, 0.0)
            self.velocity = _make_twist()

    class HealthState:
        def __init__(self) -> None:
            self.is_healthy = True
            self.fault_level = 0

    def _srv_cls(**req_defaults: Any) -> type:
        class _Request:
            def __init__(self) -> None:
                for k, v in req_defaults.items():
                    setattr(
                        self,
                        k,
                        list(v) if isinstance(v, (list, tuple)) else v,
                    )

        class _Response:
            def __init__(self, **kwargs: Any) -> None:
                self.success = True
                self.message = ""
                for k, v in kwargs.items():
                    setattr(self, k, v)

        return type("Service", (), {"Request": _Request, "Response": _Response})

    class ChannelType:
        READER = 0
        WRITER = 1
        RPC_CLIENT = 2
        RPC_SERVER = 3

    class ChannelConfig:
        def __init__(self) -> None:
            self.topic_name = ""
            self.channel_type = ChannelType.READER
            self.enable_shm = False

    class HybridChannel:
        def __init__(self, session: Session, config: ChannelConfig) -> None:
            self.session = session
            self.config = config
            self.topic = config.topic_name
            self.closed = False

        def register_typed_callback(self, msg_cls: Any, callback: Any) -> None:
            self.session.callbacks.setdefault(self.topic, []).append(
                (msg_cls, callback)
            )
            self.session.deliver_initial(self.topic, msg_cls, callback)

        def send(self, msg: Any) -> None:
            self.session.published.append((self.topic, msg))
            self.session.handle_send(self.topic, msg)

        def call(self, req: Any, timeout: float = 5.0) -> Any:
            self.session.service_calls.append((self.topic, req, timeout))
            return self.session.handle_call(self.topic, req)

        def close(self) -> None:
            self.closed = True

    class Session:
        default_body_dof: int = 21
        default_end_effector_type: str = "gripper"

        def __init__(self, node_name: str = "r2") -> None:
            self.node_name = node_name
            self.closed = False
            self.callbacks: dict[str, list[tuple[Any, Any]]] = {}
            self.published: list[tuple[str, Any]] = []
            self.service_calls: list[tuple[str, Any, float]] = []
            waist = waist_20 if type(self).default_body_dof == 20 else waist_21
            ee_type = type(self).default_end_effector_type
            ee = (
                gripper_joints
                if ee_type == "gripper"
                else (hand_joints if ee_type == "hand" else ())
            )
            active_names = waist + neck_and_arms + ee
            self.joints: dict[str, float] = dict.fromkeys(active_names, 0.0)
            if "GripperL" in self.joints:
                self.joints["GripperL"] = 1.0
                self.joints["GripperR"] = 1.0
            self.velocities: dict[str, float] = dict.fromkeys(active_names, 0.0)
            self.efforts: dict[str, float] = dict.fromkeys(active_names, 0.0)
            self.left_pose = _make_pose(0.3, 0.2, 0.2)
            self.right_pose = _make_pose(0.3, -0.2, 0.2)
            self.base_pose = [0.0, 0.0, 0.0]
            self.base_vel = [0.0, 0.0, 0.0]
            self.mode = 2

        def create_hybrid_channel(self, config: ChannelConfig) -> HybridChannel:
            return HybridChannel(self, config)

        def _build_joint_msg(self, msg_cls: Any) -> Any:
            msg = msg_cls()
            names = list(self.joints.keys())
            msg.name = names
            msg.position = [self.joints[n] for n in names]
            msg.velocity = [self.velocities.get(n, 0.0) for n in names]
            msg.effort = [self.efforts.get(n, 0.0) for n in names]
            return msg

        def _build_tcp_msg(self, msg_cls: Any) -> Any:
            msg = msg_cls()
            msg.tcp_pose = [self.left_pose, self.right_pose]
            msg.tcp_twist = [_make_twist(), _make_twist()]
            return msg

        def _build_navi_msg(self, msg_cls: Any) -> Any:
            msg = msg_cls()
            msg.pose = _make_pose(self.base_pose[0], self.base_pose[1], 0.0)
            msg.velocity = _make_twist(
                vx=self.base_vel[0], vy=self.base_vel[1], wz=self.base_vel[2]
            )
            return msg

        def deliver_initial(self, topic: str, msg_cls: Any, callback: Any) -> None:
            if topic == "/Motion/JointState":
                callback(self._build_joint_msg(msg_cls))
            elif topic == "/Motion/TCP":
                callback(self._build_tcp_msg(msg_cls))
            elif topic == "/navigation/state":
                callback(self._build_navi_msg(msg_cls))
            elif "camera" in topic:
                width, height = (1280, 720) if "zed" in topic else (640, 480)
                msg = msg_cls()
                if "depth" in topic:
                    msg.height = height
                    msg.width = width
                    msg.encoding = "32FC1"
                    msg.data = bytes(height * width * 4)
                elif "compressed" in topic:
                    msg.format = "jpeg"
                    msg.data = bytes(height * width * 3)
                else:
                    msg.height = height
                    msg.width = width
                    msg.encoding = "rgb8"
                    msg.data = bytes(height * width * 3)
                callback(msg)
            else:
                callback(msg_cls())

        def _notify(self, topic: str, builder: Any) -> None:
            for msg_cls, cb in self.callbacks.get(topic, []):
                cb(builder(msg_cls))

        def handle_send(self, topic: str, msg: Any) -> None:
            if topic == "/Motion/ServoJ":
                names = list(getattr(msg, "name", []))
                positions = list(getattr(msg, "position", []))
                velocities = list(getattr(msg, "velocity", []))
                efforts = list(getattr(msg, "effort", []))
                for idx, name in enumerate(names):
                    if idx < len(positions):
                        self.joints[name] = float(positions[idx])
                    if idx < len(velocities):
                        self.velocities[name] = float(velocities[idx])
                    if idx < len(efforts):
                        self.efforts[name] = float(efforts[idx])
                self._notify("/Motion/JointState", self._build_joint_msg)
            elif topic == "/teleop/chassis_Twist":
                vx = float(msg.linear.x)
                vy = float(msg.linear.y)
                wz = float(msg.angular.z)
                self.base_vel = [vx, vy, wz]
                self.base_pose[0] += vx * 0.05
                self.base_pose[1] += vy * 0.05
                self.base_pose[2] += wz * 0.05
                self._notify("/navigation/state", self._build_navi_msg)

        def handle_call(self, topic: str, req: Any) -> Any:
            if topic == "/Motion/SwitchController":
                self.mode = int(getattr(req, "mode", 2))
                return types.SimpleNamespace(success=True)
            if topic == "/Motion/MoveJ":
                names = list(getattr(req, "joint_names", []))
                targets = list(getattr(req, "target_positions", []))
                for idx, name in enumerate(names):
                    if idx < len(targets):
                        self.joints[name] = float(targets[idx])
                self._notify("/Motion/JointState", self._build_joint_msg)
                return types.SimpleNamespace(success=True)
            if topic == "/Motion/MoveC":
                arm = int(getattr(req, "arm", 0))
                pose = getattr(req, "target_pose", None)
                if pose is not None:
                    if arm == 0:
                        self.left_pose = pose
                    else:
                        self.right_pose = pose
                self._notify("/Motion/TCP", self._build_tcp_msg)
                return types.SimpleNamespace(success=True)
            if topic == "/Motion/ForwardKine":
                return types.SimpleNamespace(
                    success=True, tcp_pose=[self.left_pose, self.right_pose]
                )
            if topic == "/Motion/InverseKine":
                left_q = [self.joints[f"J_ArmL_{i}"] for i in range(1, 8)]
                right_q = [self.joints[f"J_ArmR_{i}"] for i in range(1, 8)]
                return types.SimpleNamespace(
                    success=True, joint_positions=left_q + right_q
                )
            return types.SimpleNamespace(success=True)

        def close(self) -> None:
            self.closed = True

    rframe_mod = module(
        "rframe",
        init=lambda *_a, **_k: None,
        shutdown=lambda *_a, **_k: None,
        Session=Session,
        ChannelConfig=ChannelConfig,
        ChannelType=ChannelType,
        HybridChannel=HybridChannel,
    )
    r2_motion_msg = module("r2_motion_msgs.msg", TCP=TCP)
    r2_motion_srv = module(
        "r2_motion_msgs.srv",
        SwitchController=_srv_cls(mode=2),
        MoveJ=_srv_cls(
            joint_names=[], target_positions=[], duration=1.0, max_velocity=[]
        ),
        MoveC=_srv_cls(arm=0, target_pose=None, duration=1.0),
        SetLoad=_srv_cls(arm=0, mass=0.0, com=[0.0, 0.0, 0.0]),
        ForwardKine=_srv_cls(joint_positions=[]),
        InverseKine=_srv_cls(tcp_pose=[], reference_joint_positions=[]),
    )
    r2_motion_pkg = module("r2_motion_msgs", msg=r2_motion_msg, srv=r2_motion_srv)

    nav_msg = module("navigation_msgs.msg", NaviState=NaviState)
    nav_srv = module("navigation_msgs.srv", NavigationControl=_srv_cls(command=0))
    nav_pkg = module("navigation_msgs", msg=nav_msg, srv=nav_srv)

    task_srv = module(
        "task_msgs.srv",
        AssignTask=_srv_cls(
            task_type="navigation", target_x=0.0, target_y=0.0, target_yaw=0.0
        ),
    )
    task_pkg = module("task_msgs", srv=task_srv)

    sm_srv = module("state_machine_msgs.srv", SwitchMode=_srv_cls(mode=0))
    sm_pkg = module("state_machine_msgs", srv=sm_srv)

    diag_msg = module("rdiagnostic_msgs.msg", HealthState=HealthState)
    diag_pkg = module("rdiagnostic_msgs", msg=diag_msg)

    std_srvs_srv = module("std_srvs.srv", Trigger=_srv_cls())
    std_srvs_pkg = module("std_srvs", srv=std_srvs_srv)

    return {
        "rframe": rframe_mod,
        "r2_motion_msgs": r2_motion_pkg,
        "r2_motion_msgs.msg": r2_motion_msg,
        "r2_motion_msgs.srv": r2_motion_srv,
        "navigation_msgs": nav_pkg,
        "navigation_msgs.msg": nav_msg,
        "navigation_msgs.srv": nav_srv,
        "task_msgs": task_pkg,
        "task_msgs.srv": task_srv,
        "state_machine_msgs": sm_pkg,
        "state_machine_msgs.srv": sm_srv,
        "rdiagnostic_msgs": diag_pkg,
        "rdiagnostic_msgs.msg": diag_msg,
        "std_srvs": std_srvs_pkg,
        "std_srvs.srv": std_srvs_srv,
    }


def modules(**_: Any) -> dict[str, types.ModuleType]:
    """Return the remaining fake robot SDKs keyed by import name."""
    made: dict[str, types.ModuleType] = {}
    made.update(gim_arm())
    made.update(turtle2())
    made.update(airbot())
    made.update(r2())
    return made
