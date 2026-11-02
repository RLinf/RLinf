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
from typing import Any, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

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


class R2Controller:
    """A simulated R2 motion controller behind the robot's RFrame/ROS 2 interface.

    It publishes ``/robotic/joint_states`` at 100 Hz from its own thread, as
    the robot does, tracks ``/robotic/servoj_data`` frames only while it is in
    ServoJ mode with no fault, and answers the services the driver calls.
    Tests drive faults through :meth:`set_main_state`, :meth:`report_health`,
    :attr:`error_code`, :attr:`service_errors`, :attr:`send_fails`, and
    :attr:`shut_down`.
    """

    WAIST = ("J_AnklePitch", "J_Knee", "J_HipPitch", "J_HipRoll", "J_HipYaw")
    LEFT = tuple(f"J_LeftArm{i}" for i in range(1, 8))
    RIGHT = tuple(f"J_RightArm{i}" for i in range(1, 8))
    GRIPPERS = ("J_LeftGripper", "J_RightGripper")
    HANDS = tuple(
        f"Hand{side}_{finger}"
        for side in ("L", "R")
        for finger in ("Little", "Ring", "Middle", "Index", "ThumbBend", "ThumbRot")
    )

    def __init__(self) -> None:
        import threading

        self.lock = threading.Lock()
        self.run_mode = 0
        self.state = 2
        self.error_code = 0
        self.main_state = 4
        #: Error code a service answers with, by service name.
        self.service_errors: dict[str, int] = {}
        #: Set to mimic Ctrl-C: RFrame's ROS 2 handler tears the transport
        #: down, so rclpy reports shutdown and every send fails.
        self.shut_down = False
        #: Set to make every send fail while ROS 2 stays up.
        self.send_fails = False
        #: Clear to make clearing faults leave the controller's fault in place.
        self.clearable = True
        #: Set to publish no joint states, as a robot whose motion node is down.
        self.silent = False
        self.frames: list[tuple[float, list[float], float, float]] = []
        self.hand_frames: list[Any] = []
        self.base_commands: list[list[float]] = []
        self.base_pose = [0.0, 0.0, 0.0]
        self.base_velocity = [0.0, 0.0, 0.0]
        self.calls: list[tuple[str, Any]] = []
        self.readers: dict[str, list[tuple[Any, Any]]] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.configure()

    def configure(
        self, *, body_dof: int = 21, end_effector_type: str = "gripper"
    ) -> None:
        """Choose the hardware variant the joint states describe."""
        neck = ("J_NeckYaw", "J_NeckPitch") if body_dof == 21 else ("J_NeckPitch",)
        effector = {"gripper": self.GRIPPERS, "hand": self.HANDS}.get(
            end_effector_type, ()
        )
        self.neck = neck
        self.joints = dict.fromkeys(
            self.WAIST + self.LEFT + self.RIGHT + neck + effector, 0.0
        )

    def start(self) -> None:
        import threading

        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._publish_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    @property
    def tracking(self) -> bool:
        """Whether ServoJ frames move the joints."""
        return self.run_mode == 2 and self.error_code == 0 and self.main_state != 7

    def deliver(self, topic: str, build: Any) -> None:
        for msg_cls, callback in list(self.readers.get(topic, [])):
            callback(build(msg_cls))

    def joint_state(self, msg_cls: Any) -> Any:
        msg = msg_cls()
        with self.lock:
            msg.name = list(self.joints)
            msg.position = list(self.joints.values())
        msg.velocity = [0.0] * len(msg.name)
        msg.effort = [0.0] * len(msg.name)
        return msg

    def tcp(self, _msg_cls: Any) -> Any:
        from rlinf.robotics.parts.arms.r2 import compute_r2_arm_fk_and_jacobian

        with self.lock:
            waist = [self.joints[j] for j in self.WAIST]
            arms = {
                "left": [self.joints[j] for j in self.LEFT],
                "right": [self.joints[j] for j in self.RIGHT],
            }
        poses = {}
        for side, q in arms.items():
            p, rpy, _, _ = compute_r2_arm_fk_and_jacobian(side, q, waist)
            quat = R.from_euler("xyz", rpy).as_quat()
            poses[f"pose_{side[0]}"] = types.SimpleNamespace(
                position=types.SimpleNamespace(x=p[0], y=p[1], z=p[2]),
                orientation=types.SimpleNamespace(
                    x=quat[0], y=quat[1], z=quat[2], w=quat[3]
                ),
            )
        return types.SimpleNamespace(**poses)

    def navi(self, _msg_cls: Any) -> Any:
        vx, vy, wz = self.base_velocity
        return types.SimpleNamespace(
            current_pose=types.SimpleNamespace(
                x=self.base_pose[0], y=self.base_pose[1], theta=self.base_pose[2]
            ),
            velocity=types.SimpleNamespace(
                linear=types.SimpleNamespace(x=vx, y=vy, z=0.0),
                angular=types.SimpleNamespace(x=0.0, y=0.0, z=wz),
            ),
        )

    @staticmethod
    def image(topic: str, msg_cls: Any) -> Any:
        width, height = (1280, 720) if "/head/" in topic else (640, 480)
        msg = msg_cls()
        msg.height, msg.width = height, width
        if "depth" in topic:
            msg.encoding = "32FC1"
            msg.data = bytes(height * width * 4)
        elif "compressed" in topic:
            msg.format = "jpeg"
            msg.data = bytes(height * width * 3)
        else:
            msg.encoding = "rgb8"
            msg.data = bytes(height * width * 3)
        return msg

    def _publish_loop(self) -> None:
        while not self._stop.wait(0.01):
            if not self.silent:
                self.deliver("/robotic/joint_states", self.joint_state)

    def subscribed(self, topic: str, msg_cls: Any, callback: Any) -> None:
        """Hand a new reader what the robot would publish next."""
        if topic == "/robotic/joint_states":
            if not self.silent:
                callback(self.joint_state(msg_cls))
            self.start()
        elif topic == "/system_state":
            callback(types.SimpleNamespace(current_main_state=self.main_state))
        elif topic == "/robotic/tcp_data":
            callback(self.tcp(msg_cls))
        elif topic == "/navi/state":
            callback(self.navi(msg_cls))
        elif topic.startswith("/robby/sensors/camera/"):
            callback(self.image(topic, msg_cls))

    def receive(self, topic: str, msg: Any) -> None:
        import time

        if topic == "/robotic/servoj_data":
            joint = list(msg.joint)
            with self.lock:
                self.frames.append(
                    (time.monotonic(), joint, msg.gripper_l, msg.gripper_r)
                )
                if self.tracking:
                    order = self.WAIST + self.LEFT + self.RIGHT + self.neck
                    for name, value in zip(order, joint, strict=False):
                        self.joints[name] = float(value)
                    for name, value in zip(
                        self.GRIPPERS, (msg.gripper_l, msg.gripper_r), strict=True
                    ):
                        if name in self.joints:
                            self.joints[name] = float(value)
            self.deliver("/robotic/tcp_data", self.tcp)
        elif topic == "/robotic/servo_ry_hand_data":
            self.hand_frames.append(msg)
            with self.lock:
                for names, ratios in (
                    (self.HANDS[:6], msg.left_finger_open_ratio),
                    (self.HANDS[6:], msg.right_finger_open_ratio),
                ):
                    for name, value in zip(names, ratios, strict=True):
                        if name in self.joints:
                            self.joints[name] = float(value)
        elif topic == "/chassis/cmd_vel":
            velocity = [float(msg.linear.x), float(msg.linear.y), float(msg.angular.z)]
            self.base_commands.append(velocity)
            self.base_velocity = velocity
            for axis in range(3):
                self.base_pose[axis] += velocity[axis] * 0.05
            self.deliver("/navi/state", self.navi)

    def set_main_state(self, main_state: int) -> None:
        """Publish a ``/system_state`` main state, as the state manager does."""
        self.main_state = int(main_state)
        self.deliver(
            "/system_state",
            lambda cls: types.SimpleNamespace(current_main_state=self.main_state),
        )

    def report_health(
        self, code_str: str, severity: int = 4, message: str = ""
    ) -> None:
        """Publish one ``/diagnostic`` HealthState carrying a single error."""
        self.deliver(
            "/diagnostic",
            lambda cls: types.SimpleNamespace(
                severity_level=severity,
                errors=[
                    types.SimpleNamespace(
                        severity_level=severity, code_str=code_str, message=message
                    )
                ],
            ),
        )

    def call(self, service: str, req: Any) -> Any:
        self.calls.append((service, req))
        error = types.SimpleNamespace(
            error_code=self.service_errors.get(service, 0), message="all ok"
        )
        if service == "/robotic/switch_mode_cmd":
            if self.main_state == 7:
                error = types.SimpleNamespace(error_code=401, message="estop")
                return types.SimpleNamespace(success=False, error_code=error)
            self.run_mode = int(req.mode)
        elif service == "/robotic/get_system_state_cmd":
            return types.SimpleNamespace(
                state=self.state,
                message="IDLE",
                error_code=types.SimpleNamespace(
                    error_code=self.error_code, message=""
                ),
                run_mode=self.run_mode,
            )
        elif service in ("/clear_faults", "/robotic/release_stop"):
            if self.main_state != 7 and self.clearable:
                self.error_code = 0
                self.state = 2
        return types.SimpleNamespace(success=True, error_code=error)


def r2(controller: Optional[R2Controller] = None) -> dict[str, types.ModuleType]:
    """Return fake R2 SDK modules wired to ``controller``, or to a fresh one.

    The controller is reachable as ``rframe.controller``. Services go through
    rclpy clients, as on the robot.
    """
    controller = controller if controller is not None else R2Controller()

    class HandlerType:
        READER = 0
        WRITER = 1

    class ChannelType:
        HYBRID = 0

    class ChannelConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class HybridChannel:
        def __init__(self, config: ChannelConfig) -> None:
            self.topic = config.topic

        def register_typed_callback(self, msg_cls: Any, callback: Any) -> None:
            controller.readers.setdefault(self.topic, []).append((msg_cls, callback))
            controller.subscribed(self.topic, msg_cls, callback)

        def send(self, msg: Any) -> None:
            if controller.shut_down or controller.send_fails:
                raise AttributeError("'int' object has no attribute 'code'")
            controller.receive(self.topic, msg)

        def close(self) -> None:
            for entry in list(controller.readers.get(self.topic, [])):
                controller.readers[self.topic].remove(entry)

    class App:
        def init(self) -> None:
            pass

    class _Future:
        def __init__(self, result: Any) -> None:
            self._result = result

        def done(self) -> bool:
            return True

        def result(self) -> Any:
            return self._result

    class _Client:
        def __init__(self, name: str) -> None:
            self.name = name

        def wait_for_service(self, timeout_sec: float = 0.0) -> bool:
            return True

        def call_async(self, req: Any) -> _Future:
            return _Future(controller.call(self.name, req))

    class _Node:
        def create_client(self, srv_cls: Any, name: str) -> _Client:
            return _Client(name)

        def destroy_node(self) -> None:
            controller.stop()

    def _service(**fields: Any) -> type:
        def _request(self: Any) -> None:
            self.__dict__.update(fields)

        request = type("Request", (), {"__init__": _request})
        return type("Service", (), {"Request": request})

    def _message(name: str) -> type:
        return type(name, (), {})

    rframe_mod = module(
        "rframe",
        App=App,
        ChannelConfig=ChannelConfig,
        ChannelType=ChannelType,
        HandlerType=HandlerType,
        HybridChannel=HybridChannel,
        controller=controller,
    )
    r2_msg = module(
        "r2_motion_msgs.msg",
        TCP=_message("TCP"),
        ServoJ=_message("ServoJ"),
        ServoRYHandDual=_message("ServoRYHandDual"),
    )
    r2_srv = module(
        "r2_motion_msgs.srv",
        GetSystemState=_service(),
        SwitchMode=_service(mode=0),
        SetAxisEnable=_service(),
        SetGripperParam=_service(),
    )
    nav_msg = module("navigation_msgs.msg", NaviState=_message("NaviState"))
    sm_msg = module("state_machine_msgs.msg", SystemState=_message("SystemState"))
    diag_msg = module("rdiagnostic_msgs.msg", HealthState=_message("HealthState"))
    diag_srv = module("rdiagnostic_msgs.srv", ClearFaults=_service())
    std_srv = module("std_srvs.srv", Trigger=_service(), SetBool=_service(data=False))
    system_srv = module("system_msgs.srv", PowerCtrl=_service(cmd=0))
    rclpy_mod = module(
        "rclpy",
        ok=lambda: not controller.shut_down,
        init=lambda *_a, **_k: None,
        create_node=lambda _name: _Node(),
        spin_until_future_complete=lambda *_a, **_k: None,
    )
    return {
        "rframe": rframe_mod,
        "rclpy": rclpy_mod,
        "r2_motion_msgs": module("r2_motion_msgs", msg=r2_msg, srv=r2_srv),
        "r2_motion_msgs.msg": r2_msg,
        "r2_motion_msgs.srv": r2_srv,
        "navigation_msgs": module("navigation_msgs", msg=nav_msg),
        "navigation_msgs.msg": nav_msg,
        "state_machine_msgs": module("state_machine_msgs", msg=sm_msg),
        "state_machine_msgs.msg": sm_msg,
        "rdiagnostic_msgs": module("rdiagnostic_msgs", msg=diag_msg, srv=diag_srv),
        "rdiagnostic_msgs.msg": diag_msg,
        "rdiagnostic_msgs.srv": diag_srv,
        "std_srvs": module("std_srvs", srv=std_srv),
        "std_srvs.srv": std_srv,
        "system_msgs": module("system_msgs", srv=system_srv),
        "system_msgs.srv": system_srv,
    }


def modules(**_: Any) -> dict[str, types.ModuleType]:
    """Return the remaining fake robot SDKs keyed by import name."""
    made: dict[str, types.ModuleType] = {}
    made.update(gim_arm())
    made.update(turtle2())
    made.update(airbot())
    made.update(r2())
    return made
